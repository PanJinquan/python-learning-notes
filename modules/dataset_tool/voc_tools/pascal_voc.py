# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : pascal_voc.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-09 20:39:21
"""
import xml.etree.ElementTree as ET
from modules.dataset_tool import coordinate_label

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
VOC_CLASSES_BG = ["BACKGROUND"] + VOC_CLASSES


def get_annotation(annotations_file, classes, minAreaTH=0, coordinatesType="SSD", remove_difficult=False):
    '''
    label data format：
    coordinatesType:
    SSD  = [label_id,x,y,w,h]
    xywh = [label_id,x,y,w,h]
    xyxy = [label_id,xmin,ymin,xmax,ymax]
    xxyy = [label_id,xmin,xmax,ymin,ymax]
    YOLO = [label_id,x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    MMDET= [img_width,img_height,label_id,x,y,w,h]


    :param annotations_file:
    :param classes:
    :param coordinatesType: 坐标类型：SSD,YOLO,MMDET,xywh格式
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
        if cls not in classes:
            continue
        if int(difficult) == 1 and remove_difficult:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # b=[xmin,xmax,ymin,ymax]
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        area = (b[1] - b[0]) * (b[3] - b[2])
        if minAreaTH > 0 and area < minAreaTH:
            print("area< minAreaTH:{}<{}".format(area, minAreaTH))
            continue
        if coordinatesType == "SSD" or coordinatesType == "xywh":
            rect = coordinate_label.convert_ssd((w, h), b)
        elif coordinatesType == "YOLO":
            rect = coordinate_label.convert_yolo((w, h), b)
        elif coordinatesType == "MMDET":
            rect = coordinate_label.convert_mmdet((w, h), b)
        elif coordinatesType == "xyxy":
            [xmin,xmax,ymin,ymax]=b
            rect=[xmin,ymin,xmax,ymax]
        elif coordinatesType == "xxyy":
            rect=b
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
    for box, cls in zip(boxList, classes):
        line_image_label += box
        line_image_label += [cls]
    return line_image_label
