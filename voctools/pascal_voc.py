# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : pascal_voc.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-09 20:39:21
"""
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from utils import file_processing, image_processing


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
    xmin, xmax, ymin, ymax = box
    return [xmin, ymin, xmax, ymax]


def get_annotation(annotations_file, classes, coordinatesType="SSD"):
    '''

    :param annotations_file:
    :param classes:
    :param coordinatesType: 坐标类型：SSD,YOLO,MMDET格式
    :return:
    '''
    tree = ET.parse(annotations_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    rects = []
    class_name = []
    class_id = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # b=[xmin,xmax,ymin,ymax]
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        if coordinatesType == "SSD":
            rect = convert_ssd((w, h), b)
        elif coordinatesType == "YOLO":
            rect = convert_yolo((w, h), b)
        elif coordinatesType == "MMDET":
            rect = convert_mmdet((w, h), b)
        else:
            print("Error:coordinatesType={},must be SSD or YOLO".format(coordinatesType))
        rects.append(rect)
        class_name.append(cls)
        class_id.append(cls_id)
    return rects, class_name, class_id
