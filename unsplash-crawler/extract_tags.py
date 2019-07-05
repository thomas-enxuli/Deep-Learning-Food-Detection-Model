#!/usr/bin/env python3
"""
extract_tags.py - extract image tag data from unsplash.sqlite file

Usage:
  crawl_unsplash.py <unsplash_sqlite_file> <output_sqlite_file>
"""
from collections import defaultdict
from docopt import docopt
from sqlitedict import SqliteDict

MIN_IMAGE_THRESHOLD = 50


def main():
    args = docopt(__doc__, help=True)
    input_file = args['<unsplash_sqlite_file>']
    output_file = args['<output_sqlite_file>']
    input_data = SqliteDict(input_file, tablename='images', flag='r')
    tag_data = defaultdict(list)
    for img, img_data in input_data.items():
        if 'tags' in img_data and 'urls' in img_data:
            for tag in img_data['tags']:
                tag_data[tag] += [img]
    with SqliteDict(output_file, tablename='tags', autocommit=True) as output_data:
        output_data = {'unsplash': {}}
        for tag in tag_data:
            img_list = tag_data[tag]
            if len(img_list) >= MIN_IMAGE_THRESHOLD:
                output_data['unsplash'][tag] = {
                    'pos': img_list,
                    'neg': [],
                    'unk': [],
                }


if __name__ == '__main__':
    main()
