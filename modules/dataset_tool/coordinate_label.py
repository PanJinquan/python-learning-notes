# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : coordinate_label.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-25 15:39:36
"""





def convert_yolo(size, box):
    '''
    YOLO label data
    [xmin,xmax,ymin,ymax] convert to [x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    :param size:
    :param box:
    :return:
    '''
    # box=[xmin,xmax,ymin,ymax]
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def convert_ssd(size, box):
    '''
    SSD label data
    [x y width height]
    :param size:
    :param box:
    :return:
    '''
    # box=[xmin,xmax,ymin,ymax]
    dw = 1
    dh = 1
    x = box[0]
    y = box[2]
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def convert_mmdet(size, box):
    '''
    mmdetection label data
    将[xmin,xmax,ymin,ymax]转为[xmin,ymin,xmax,ymax]
    :param size:
    :param box:
    :return:
    '''
    (w, h) = size
    xmin, xmax, ymin, ymax = box
    return [w, h, xmin, ymin, xmax, ymax]






