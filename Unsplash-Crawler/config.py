# -*- coding: utf-8 -*-
# @Author  : YunWei.Chen
# @Site    : https://chen.yunwei.space


# 配置相关下载参数
class Config:
    # GET_TYPE 参数，定义获取图片的顺序，可选参数与说明：
    # latest：按时间从现在到以前的顺序获取图片
    # oldest：按时间从以前到现在的顺序获取图片
    # popular：按图片的热度顺序获取图片
    GET_TYPE = 'popular'

    # GET_COUNT 参数，设定获取图片的数量，设置为 0 时将获取所有可获取的图片
    GET_COUNT = 0

    # GET_SIZE 参数(注意大小写)，定义获取图片的尺寸，可选参数与说明：
    # raw：包含Exif信息的全尺寸原图，此类图片的容量很大
    # full：全尺寸分辨率的图片，去除了Exif信息并且对内容进行了压缩，图片容量适中
    # normal：普通尺寸的图片，去除了Exif信息，并且对分辨率和内容进行了压缩，图片容量较小；配合IMAGE_WIDTH参数进行使用
    GET_SIZE = 'normal'

    # IMAGE_WIDTH 参数(注意是字符串)，定义下载图片的最大宽度（像素）
    # 配合 GET_SIZE='normal' 参数使用时可下载设定分辨率大小的图片（当设定分辨率大于原图分辨率时，以原图分辨率下载）
    IMAGE_WIDTH = '1920'

    # 图片保存的目录名称
    SAVE_DIR = 'Photos'

    # 开启下载线程数量
    THREAD_COUNT = 4
