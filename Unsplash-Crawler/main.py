# -*- coding: utf-8 -*-
# @Author  : YunWei.Chen
# @Site    : https://chen.yunwei.space

import os
import json
import threading
import requests
from bs4 import BeautifulSoup
import sqlite3
from config import Config

basedir = os.path.abspath(os.path.dirname(__file__))  # 程序文件所在目录
lock = threading.Lock()  # 线程锁
client_id = None
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
    'accept-version': 'v1',
    'viewport-width': Config.IMAGE_WIDTH or '1920'  # 屏幕宽度
}

# 获取首页HTML代码
print('\n正在连接主站：https://unsplash.com')
r = requests.get('https://unsplash.com', headers=headers)

# 对HTML进行解析
soup = BeautifulSoup(r.text, 'html.parser')

print('\n正在分析和获取Client-ID......')
# 从JS文件中获取Client-ID 并加入到 header 的 authorization 字段中
for script in soup.find_all('script'):
    if script.get('src'):
        # 获取JS文件
        r = requests.get('https://unsplash.com' + script.get('src'), headers=headers)
        if r.text.find('UNSPLASH_APP_ID') > 0:
            # 从JS文件中截取Client-ID字符串
            start_find = r.text.index('UNSPLASH_APP_ID')
            start_id = r.text.index('"', start_find) + 1
            end_id = r.text.index('"', start_id)
            client_id = r.text[start_id:end_id]
            if client_id:
                # 将 Client-ID 加入到请求头中
                headers["authorization"] = 'Client-ID ' + client_id
                break

if client_id:
    print('\n提取到Client-ID为：', client_id)
else:
    print('\n提取Client-ID失败，请检查相关错误')
    exit()

# 连接sqlite数据库
db = sqlite3.connect(os.path.join(basedir, 'UnsplashPhotos.sqlite'), check_same_thread=False)

# 创建游标
cursor = db.cursor()

# 创建数据库Photo表
cursor.execute('CREATE TABLE IF NOT EXISTS photos'
               '(id INTEGER PRIMARY KEY AUTOINCREMENT,'
               'photo_id VARCHAR(64),'
               'photo_likes INTEGER DEFAULT 0,'
               'raw_url VARCHAR(255),'
               'full_url VARCHAR(255),'
               'normal_url VARCHAR(255),'
               'author_name VARCHAR(64),'
               'author_username VARCHAR(64),'
               'photo_name VARCHAR(128),'
               'downloaded INTEGER DEFAULT 0)')

# 创建数据库Status表
cursor.execute('CREATE TABLE IF NOT EXISTS status'
               '(id INTEGER PRIMARY KEY AUTOINCREMENT,'
               'done INTEGER DEFAULT 0,'
               'page INTEGER DEFAULT 1)')

base_url = 'https://unsplash.com/napi/photos?page='
page_num = 1
base_url_end = '&per_page=30'

if Config.GET_TYPE == 'latest':
    base_url_end += '&order_by=latest'
elif Config.GET_TYPE == 'oldest':
    base_url_end += '&order_by=oldest'
elif Config.GET_TYPE == 'popular':
    base_url_end += '&order_by=popular'
else:
    print('\nGET_TYPE 参数设置错误，请检查config.py文件中的配置参数')
    exit()


def change_url(url, width):
    new_url = url[0:url.index('&w=') + 3]
    new_url += str(width)
    new_url += url[url.index('&fit='):]
    return new_url


def save_url_to_db(photo_list):
    for photo in photo_list:
        photo_id = photo['id']
        photo_likes = int(photo['likes'])
        raw_url = photo['urls']['raw']
        full_url = photo['urls']['full']
        normal_url = change_url(photo['urls']['regular'], Config.IMAGE_WIDTH)
        author_name = photo['user']['name']
        author_username = photo['user']['username']
        downloaded = 0
        cursor.execute('INSERT INTO photos('
                       'photo_id,'
                       'photo_likes,'
                       'raw_url,'
                       'full_url,'
                       'normal_url,'
                       'author_name,'
                       'author_username,'
                       'downloaded) VALUES (?,?,?,?,?,?,?,?);', (photo_id,
                                                                 photo_likes,
                                                                 raw_url,
                                                                 full_url,
                                                                 normal_url,
                                                                 author_name,
                                                                 author_username,
                                                                 downloaded))


url_is_done = False

cursor.execute('SELECT * FROM status')  # 查询数据库
download_status = cursor.fetchall()  # 获取查询结果

if len(download_status) < 1:
    cursor.execute('INSERT INTO status(done,page) VALUES (?,?)', (0, 1))
    db.commit()
else:
    if download_status[0][1] > 0:
        url_is_done = True
    else:
        url_is_done = False
        page_num = int(download_status[0][2])

if not url_is_done:
    if len(download_status) < 1:
        print('\n正在批量获取图片下载连接和写入数据库，可随时中断任务，再次运行程序时将继续任务。')
    else:
        print('\n继续批量获取图片下载连接和写入数据库，可随时中断任务，再次运行程序时将继续任务。')
    print('视网络情况、开启的下载线程数和图片数量，这将花费一些时间，请耐心等候......\n')
    if Config.GET_COUNT > 0:
        cursor.execute('SELECT * FROM photos')  # 查询数据库
        all_photos = cursor.fetchall()  # 获取查询结果
        get_num = len(all_photos)


        def get_url_with_count():
            global get_num, page_num
            while get_num < Config.GET_COUNT:
                lock.acquire()
                tmp_page_num = page_num
                page_num += 1
                lock.release()
                try:
                    r_c = requests.get(base_url + str(tmp_page_num) + base_url_end, headers=headers)
                except:
                    try:
                        r_c = requests.get(base_url + str(tmp_page_num) + base_url_end, headers=headers)
                    except:
                        print('获取第 ', tmp_page_num, ' 页的图片URL失败！')
                        continue
                r_c_list = json.loads(r_c.text)
                if type(r_c_list) is list:
                    lock.acquire()
                    try:
                        save_url_to_db(r_c_list)
                        cursor.execute('UPDATE status SET page=? WHERE id=1;', (tmp_page_num + 1,))
                        db.commit()  # 写入数据库
                        print('在数据库中添加了 ' + str(len(r_c_list)) + ' 条记录')
                        get_num += len(r_c_list)
                    finally:
                        lock.release()
                else:
                    break


        t = []
        for n in range(Config.THREAD_COUNT):
            t_tmp = threading.Thread(target=get_url_with_count)
            t_tmp.start()
            t.append(t_tmp)
        for th in t:
            th.join()
    else:
        def get_url_all():
            global page_num
            while True:
                lock.acquire()
                tmp_page_num = page_num
                page_num += 1
                lock.release()
                try:
                    r_a = requests.get(base_url + str(tmp_page_num) + base_url_end, headers=headers)
                except:
                    try:
                        r_a = requests.get(base_url + str(tmp_page_num) + base_url_end, headers=headers)
                    except:
                        print('获取第 ', tmp_page_num, ' 页的图片URL失败！')
                        continue
                r_a_list = json.loads(r_a.text)
                if type(r_a_list) is list:
                    lock.acquire()
                    try:
                        save_url_to_db(r_a_list)
                        cursor.execute('UPDATE status SET page=? WHERE id=1;', (tmp_page_num + 1,))
                        db.commit()  # 写入数据库
                        print('在数据库中添加了 ' + str(len(r_a_list)) + ' 条记录')
                    finally:
                        lock.release()
                else:
                    break


        t = []
        for n in range(Config.THREAD_COUNT):
            t_tmp = threading.Thread(target=get_url_all)
            t_tmp.start()
            t.append(t_tmp)
        for th in t:
            th.join()
    if Config.GET_COUNT > 0:
        cursor.execute('SELECT * FROM photos')  # 查询数据库
        tmp_photos = cursor.fetchall()  # 获取查询结果
        if len(tmp_photos) > Config.GET_COUNT:
            del_id = tmp_photos[Config.GET_COUNT - 1][0]
            cursor.execute('DELETE FROM photos WHERE id>?;', (del_id,))
            db.commit()
    cursor.execute('SELECT * FROM photos')  # 查询数据库
    all_photos = cursor.fetchall()  # 获取查询结果
    print('\n已成功获取 ' + str(len(all_photos)) + ' 张图片下载地址并写入数据库，准备开始下载图片......')
else:
    print('\n图片地址已存在于数据库中，将继续完成数据库中未完成的下载任务......')

save_path = os.path.join(basedir, Config.SAVE_DIR)  # 图片文件保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

if Config.GET_SIZE == 'raw':
    size_url = 3
elif Config.GET_SIZE == 'full':
    size_url = 4
elif Config.GET_SIZE == 'normal':
    size_url = 5
else:
    print('\nGET_SIZE 参数设置错误，请检查config.py文件中的配置参数')
    exit()

cursor.execute('SELECT * FROM photos WHERE downloaded=0;')  # 查询数据库
photos = cursor.fetchall()  # 获取查询结果

all_photos = len(photos)
has_error = False
download_num = 0
print('\n开始下载图片，可随时中断任务，再次运行程序时将继续任务。')


def download_photo():
    global download_num, all_photos, has_error
    while download_num < all_photos:
        lock.acquire()
        try:
            photo_info = photos[download_num]
            download_num += 1
            print('正在下载第 ' + str(download_num) + ' 张图片......')
        except:
            print('从数据库中获取图片下载地址失败！')
            has_error = True
            lock.release()
            continue
        finally:
            lock.release()
        try:
            r_img = requests.get(photo_info[size_url], headers=headers)
            if r_img.status_code == 200:
                image_type = r_img.headers['Content-Type'].split('/')[-1]
                tmp_name = photo_info[3].split('/')[-1]
                if tmp_name.find('.') > 0:
                    image_name = tmp_name
                else:
                    image_name = tmp_name + '.' + image_type
                with open(os.path.join(save_path, image_name), 'wb') as img:
                    img.write(r_img.content)
                    lock.acquire()
                    try:
                        cursor.execute('UPDATE photos SET photo_name=?, downloaded=1 WHERE id=?;',
                                       (image_name, photo_info[0]))
                        db.commit()  # 写入数据库
                    finally:
                        lock.release()
            else:
                has_error = True
                print('下载数据库中记录id为：' + str(photo_info[0]) + ' 的图片出错')
        except:
            has_error = True
            print('下载数据库中记录id为：' + str(photo_info[0]) + ' 的图片出错')


t = []
for n in range(Config.THREAD_COUNT):
    t_tmp = threading.Thread(target=download_photo)
    t_tmp.start()
    t.append(t_tmp)
for th in t:
    th.join()

if has_error:
    print('\n下载图片完成，但仍有图片未完成下载，请再次执行下载程序或检查错误。')
else:
    print('\nWell done! 所有图片均已下载完成')

db.close()
