# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : face_body_tools.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-21 20:34:05
"""

from utils import json_utils
from modules.dataset_tool import coordinate_label


def parse_annotation(annotations_file, class_names):
    json_data = json_utils.load_config(annotations_file)
    boxList = json_data["boxList"]
    return boxList


def read_line_image_label(line_image_label):
    '''
    line_image_label:[image_id,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id,...]
    :param line_image_label: 
    :return: 
    '''
    line_image_label = line_image_label.strip().split()
    image_id = line_image_label[0]
    boxes_nums = int(line_image_label[1])
    box = []
    label = []
    for i in range(boxes_nums):
        x = float(line_image_label[2 + 5 * i])
        y = float(line_image_label[3 + 5 * i])
        w = float(line_image_label[4 + 5 * i])
        h = float(line_image_label[5 + 5 * i])
        c = int(line_image_label[6 + 5 * i])
        if w <= 0 or h <= 0:
            continue
        box.append([x, y, x + w, y + h])
        label.append(c)
    return image_id, box, label


def get_annotation(annotations_file, classes, image_shape, coordinatesType="SSD"):
    '''
    label data format：
    SSD  = [label_id,x,y,w,h]
    YOLO = [label_id,x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    MMDET= [img_width,img_height,label_id,x,y,w,h]
    :param annotations_file:
    :param classes:
    :param coordinatesType: 坐标类型：SSD,YOLO,MMDET格式
    :return:
    '''
    boxList = parse_annotation(annotations_file, classes)

    rects = []
    class_name = []
    class_id = []
    h, w, d = image_shape
    for item in boxList:
        cls = item["label"]
        cls_id = classes.index(cls)
        xmin = float(item["xtl"])
        xmax = float(item["xbr"])
        ymin = float(item["ytl"])
        ymax = float(item["ybr"])
        b = [xmin, xmax, ymin, ymax]
        if coordinatesType == "SSD":
            rect = coordinate_label.convert_ssd((w, h), b)
        elif coordinatesType == "YOLO":
            rect = coordinate_label.convert_yolo((w, h), b)
        elif coordinatesType == "MMDET":
            rect = coordinate_label.convert_mmdet((w, h), b)
        else:
            raise "Error:coordinatesType={},must be SSD,YOLO or MMDET".format(coordinatesType)
        rects.append(rect)
        class_name.append(cls)
        class_id.append(cls_id)
    return rects, class_name, class_id


def convert_to_linedataset(image_path, boxList, classes):
    '''
    :param image_path:
    :param boxList:
    :param classes:
    :return:line_image_label:[image_path,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id,...]
    '''
    boxes_nums = len(boxList)
    line_image_label = [image_path, boxes_nums]
    for item in boxList:
        name = item["label"]
        label = classes.index(name)
        xmin = float(item["xtl"])
        xmax = float(item["xbr"])
        ymin = float(item["ytl"])
        ymax = float(item["ybr"])
        w = xmax - xmin
        h = ymax - ymin
        # box = [xmin, ymin, xmax, ymax]
        # box = [int(float(b)) for b in box]
        rect = [xmin, ymin, w, h]
        rect = [int(r) for r in rect]
        line_image_label += rect
        line_image_label += [label]
    return line_image_label
