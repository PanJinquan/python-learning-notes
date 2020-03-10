# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : vocDemo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-09 19:10:16
"""
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from utils import file_processing, image_processing
from voctools import pascal_voc

# for SSD  label，the first label is BACKGROUND：
# classes = ["BACKGROUND", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# for YOLO label,ignore the BACKGROUND
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# for wall
# classes = ["PCwall"]
# classes=["BACKGROUND",'PCwall']
classes=["BACKGROUND",'face']
print("class_name:{}".format(classes))


def pascal_voc_test(annotations_dir, image_dir, class_names, coordinatesType="SSD", show=True):
    '''

    :param annotations_dir:
    :param image_dir:
    :param class_names:
    :param coordinatesType:
    :param show:
    :return:
    '''
    annotations_list = file_processing.get_files_list(annotations_dir, postfix=["*.xml"])
    print("have {} annotations files".format(len(annotations_list)))
    for i, annotations_file in enumerate(annotations_list):
        name_id = os.path.basename(annotations_file)[:-len(".xml")]
        image_name = name_id + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print("no image_dict:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        rects, class_name, class_id = pascal_voc.get_annotation(annotations_file, class_names, coordinatesType)
        if len(rects) == 0 or len(class_name) == 0 or len(class_id) == 0:
            print("no class in annotations:{}".format(annotations_file))
        if show:
            image = image_processing.read_image(image_path)
            image_processing.show_image_rects_text("image_dict", image, rects, class_name)


if __name__ == "__main__":
    # annotations_dir = './dataset/VOC/Annotations'
    # image_dir = "./dataset/VOC/JPEGImages"

    annotations_dir = '/media/dm/dm2/project/dataset/face/Annotations'
    image_dir = "/media/dm/dm2/project/dataset/face/JPEGImages"
    pascal_voc_test(annotations_dir, image_dir, classes, coordinatesType="SSD", show=True)
