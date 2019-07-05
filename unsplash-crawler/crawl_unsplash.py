#!/usr/bin/env python3
"""
crawl_unsplash.py - Downloads data/images from unsplash.com

Usage:
  crawl_unsplash.py (--collections=FILE | --download_images=QUALITY)
  crawl_unsplash.py -h | --help

Options:
  --collections=FILE              FILE consisting of newline separated collection ids to process
  --download_images=QUALITY       QUALITY must be one of raw, full, regular, small, thumbnail

"""
import concurrent.futures
import json
import os
import re
import shutil
import sys
import time
import traceback
import urllib.request

from docopt import docopt
from retry import retry
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from sqlitedict import SqliteDict

INITIAL_STATE_RE = re.compile(r'<script>window.__INITIAL_STATE__ = (.*?);</script>')
DB_PATH = 'photos/unsplash.sqlite'

class WebDriver:
    """wrapper for webdriver to allow context manager usage"""

    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument('window-size=1450x841')
        options.add_argument('headless')
        options.add_argument('disable-gpu')
        options.add_argument('no-sandbox')
        driver = webdriver.Chrome(chrome_options=options)
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.quit()


def _fetch_initial_state(url):
    """Fetch json "window.__INITIAL_STATE__" from page, convert to dict, and return it"""

    @retry(WeirdPage, delay=0.5, backoff=2, max_delay=6, tries=10)
    def get_html(url):
        try:
            with urllib.request.urlopen(url) as response:
                html = response.read().decode('utf-8')
                return html
        except Exception:
            raise WeirdPage

    html = get_html(url)
    match_obj = INITIAL_STATE_RE.search(html)
    match_str = match_obj.group(1)
    initial_state = json.loads(match_str)
    return initial_state


def _simplify_tags(data):
    return [t['title'] for t in data]


def _extract_collection_metadata(initial_state, target_id):
    """Extract collection metadata from `initial_state` returned by `_fetch_initial_state`"""
    collections = initial_state['entities']['collections']
    coll_keep_keys = set(['title', 'description', 'curated', 'featured', 'total_photos', 'tags'])
    coll_data = collections[target_id]
    for k in list(coll_data.keys()):
        if k not in coll_keep_keys:
            del coll_data[k]
        if k == 'tags':
            coll_data[k] = _simplify_tags(coll_data[k])

    other_colls = [coll for coll in collections.keys() if coll is not target_id]
    return coll_data, other_colls


def _extract_image_metadata(initial_state, target_id):
    """Extract image metadata from `initial_state` returned by `fetch_initial_state`"""
    images = initial_state['entities']['photos']
    image_keep_keys = set([
        'urls',
        'description',
        'tags',
        'likes',
        'views',
        'height',
        'width',
        'location',
    ])
    image_data = images[target_id]
    for k in list(image_data.keys()):
        if k not in image_keep_keys:
            del image_data[k]
        if k == 'tags':
            image_data[k] = _simplify_tags(image_data[k])

    other_images = [img for img in images if img is not target_id]
    other_colls = list(initial_state['entities']['collections'].keys())
    return image_data, other_images, other_colls


def _fetch_image_list(coll_id, total_photos):
    """Fetch collection image id list from unsplash and return it"""
    time.sleep(0.5)
    with WebDriver() as web_driver:
        driver = web_driver.driver
        driver.get('https://unsplash.com/collections/' + coll_id)

        if total_photos > 20:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            driver.execute_script("window.scrollTo(0, 0)")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            try:
                button = driver.find_element_by_xpath('//button[text()="Load more photos"]')
            except NoSuchElementException:
                raise WeirdPage
            driver.execute_script('window.scrollTo(0, ' + str(button.location['y'] - 100) + ')')
            ActionChains(driver).move_to_element(button).perform()
            button.click()
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")

        num_no_increase = 0
        num_found = 0

        @retry(StillLoading, delay=0.5, backoff=2, max_delay=6)
        def find_image_ids():
            nonlocal num_no_increase, num_found
            images = driver.find_elements_by_xpath('//figure/div/a')
            if len(images) == num_found:
                num_no_increase += 1
            elif len(images) < num_found:
                raise WeirdPage
            else:
                num_no_increase = 0

            num_found = len(images)
            if num_found / total_photos < 0.9:
                driver.execute_script("window.scrollTo(0, 0)")
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                if num_no_increase < 4:
                    raise StillLoading
                else:
                    raise WeirdPage

            print(num_found, total_photos)
            img_ids = [img.get_attribute('href').split('/')[-1] for img in images]
            return img_ids

        image_ids = find_image_ids()
    return image_ids


def _fetch_collection_metadata(coll_id):
    """Fetch collection metadata from unsplash and return it"""
    initial_state = _fetch_initial_state('https://unsplash.com/collections/' + coll_id)
    coll_metadata, other_colls = _extract_collection_metadata(initial_state, coll_id)
    return coll_metadata, other_colls


def _fetch_image_metadata(image_id):
    """Fetch image metadata from unsplash and return it"""
    initial_state = _fetch_initial_state('https://unsplash.com/photos/' + image_id)
    image_metadata, other_images, other_colls = _extract_image_metadata(initial_state, image_id)
    return image_metadata, other_images, other_colls


def fill_collection_data(coll_id, colls_data, images_data):
    """
    Populate the `colls_data` and `images_data` sqlite tables with information relating to
    collection `coll_id` (collection metadata, collection image list, images metadata, and stubs
    for related collections and images encountered.
    """

    data = colls_data.get(coll_id, {})
    if 'total_photos' not in data:
        metadata, other_colls = _fetch_collection_metadata(coll_id)
        data.update(metadata)
        colls_data[coll_id] = data
        for coll in other_colls:
            if coll not in colls_data:
                colls_data[coll] = {}

    total_photos = data['total_photos']
    if 'image_ids' not in data or len(data['image_ids']) / total_photos < 0.9:
        image_ids = _fetch_image_list(coll_id, total_photos)
        data['image_ids'] = image_ids
        colls_data[coll_id] = data

    if 'filled_image_metadata' not in data:
        image_ids = data['image_ids']
        for image_id in image_ids:
            fill_image_metadata(image_id, colls_data, images_data)
        data['filled_image_metadata'] = True
        colls_data[coll_id] = data


def fill_image_metadata(image_id, colls_data, images_data):
    """
    Populate the `colls_data` and `images_data` sqlite tables with information relating to
    image `image_id` (image metadata, and stubs for related collections and images encountered.
    """

    data = images_data.get(image_id, {})
    if not data:
        metadata, other_images, other_colls = _fetch_image_metadata(image_id)
        data.update(metadata)
        images_data[image_id] = data
        for img in other_images:
            if img not in images_data:
                images_data[img] = {}
        for coll in other_colls:
            if coll not in colls_data:
                colls_data[coll] = {}


def process_collections_file(collections_file):
    with SqliteDict(DB_PATH, tablename='collections', autocommit=True) as colls_data:
        with SqliteDict(DB_PATH, tablename='images', autocommit=True) as images_data:
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                futures = {}
                for ln in open(collections_file):
                    coll_id = ln.strip()
                    futures[executor.submit(fill_collection_data, coll_id, colls_data,
                                            images_data)] = coll_id
                for future in concurrent.futures.as_completed(futures):
                    coll_id = futures[future]
                    try:
                        future.result()
                        print("Filled data for collection: ", coll_id)
                    except WeirdPage:
                        print('WeirdPage')
                    except Exception as e:
                        tb = ''.join(traceback.format_exception(None, e, e.__traceback__))
                        print('%r generated an exception: %s' % (coll_id, tb))


def _try_download_url(url, download_path):
    with urllib.request.urlopen(url) as response:
        if response.code == 200:
            with open(download_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)


def download_images(image_quality='regular'):
    download_dir = os.path.join('photos', image_quality)
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    with SqliteDict(DB_PATH, tablename='images', flag='r') as images_data:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            for image_id in images_data:
                download_path = os.path.join(download_dir, image_id + '.jpg')
                if not os.path.exists(download_path):
                    if 'urls' in images_data[image_id]:
                        if image_quality in images_data[image_id]['urls']:
                            url = images_data[image_id]['urls'][image_quality]
                            executor.submit(_try_download_url, url, download_path)


def main():
    args = docopt(__doc__, help=True)
    if args['--collections']:
        collections_file = args['--collections']
        process_collections_file(collections_file)

    elif args['--download_images']:
        quality = args['--download_images']
        if quality not in ['raw', 'full', 'regular', 'small', 'thumbnail']:
            print(__doc__, file=sys.stderr)
            sys.exit(1)
        download_images(image_quality=quality)


class WeirdPage(Exception):
    pass


class StillLoading(Exception):
    pass


if __name__ == '__main__':
    main()
