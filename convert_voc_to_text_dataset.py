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
import random
from os import listdir, getcwd
from os.path import join
from utils import file_processing, image_processing
from modules.dataset_tool import pascal_voc, comment


# for SSD  label，the first label is BACKGROUND：
# classes = ["BACKGROUND","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# for YOLO label,ignore the BACKGROUND
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# for wall
# classes = ["PCwall"]
# classes = ["BACKGROUND","person", "dog"]
# print("class_name:{}".format(classes))


def convert_voc_to_textdataset_for_annotation(annotations_list, image_dir, label_out_dir, class_names, coordinatesType,
                                              image_type='.jpg', labelType="class_id", show=True):
    '''
    coordinatesType:
    SSD  = [label_id,x,y,w,h]
    xywh = [label_id,x,y,w,h]
    xyxy = [label_id,xmin,ymin,xmax,ymax]
    xxyy = [label_id,xmin,xmax,ymin,ymax]
    YOLO = [label_id,x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    MMDET= [img_width,img_height,label_id,x,y,w,h]
    :param annotations_list:annotations列表
    :param image_dir:图片所在路径
    :param label_out_dir:输出label目录
    :param class_names:
    :param image_type:图片的类型，如.jpg ,.png
    :param labelType:class_name,class_id
    :param show:
    :return:
    '''
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)
    name_id_list = []
    nums = len(annotations_list)
    for i, annotations_file in enumerate(annotations_list):
        name_id = os.path.basename(annotations_file)[:-len(".xml")]
        image_name = name_id + image_type
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print("no image:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        out_file = os.path.join(label_out_dir, name_id + ".txt")
        rects, class_name, class_id = pascal_voc.get_annotation(annotations_file, class_names,
                                                                coordinatesType=coordinatesType)
        if labelType == "class_name":
            label = class_name
        elif labelType == "class_id":
            label = class_id
        content_list = [[c] + r for c, r in zip(label, rects)]
        name_id_list.append(name_id)
        file_processing.write_data(out_file, content_list, mode='w')
        if show:
            image = image_processing.read_image(image_path)
            # rect_image = image_processing.get_rects_image(image,rects)
            # save_root=DATASET_ROOT+"/trainval_faces"
            # image_processing.save_image_lable_dir(save_root, rect_image, class_name,i)
            image_processing.show_image_rects_text("image", image, rects, class_name)
        if i % 10 == 0 or i == len(annotations_list) - 1:
            print("processing image:{}/{}".format(i, len(annotations_list) - 1))
    return name_id_list


def text_dataset_for_annotation(annotations_dir, image_dir, label_out_dir, out_train_val_path, class_names,
                                coordinatesType,
                                shuffle=True, labelType="class_id", show=True):
    '''
    :param annotations_dir:
    :param image_dir:
    :param label_out_dir:
    :param out_train_val_path:
    :param class_names:
    :param labelType:class_name,class_id
    :param show:
    :return:
    '''
    annotations_list = file_processing.get_files_list(annotations_dir, postfix=["*.xml"])
    print("have {} annotations files".format(len(annotations_list)))
    if shuffle:
        seeds = 100  # 固定种子,只要seed的值一样，后续生成的随机数都一样
        random.seed(seeds)
        random.shuffle(annotations_list)

    # 分割成train和val数据集
    factor = 1.0
    train_num = int(factor * len(annotations_list))
    train_annotations_list = annotations_list[:train_num]
    val_annotations_list = annotations_list[train_num:]

    # 转换label数据
    print("doing train data .....")
    train_image_id = convert_voc_to_textdataset_for_annotation(train_annotations_list, image_dir, label_out_dir,
                                                               class_names, coordinatesType,
                                                               image_type=".jpg", labelType=labelType, show=show)
    print("doing val data .....")
    val_image_id = convert_voc_to_textdataset_for_annotation(val_annotations_list, image_dir, label_out_dir,
                                                             class_names, coordinatesType,
                                                             image_type=".jpg", labelType=labelType, show=show)
    print("done...ok!")

    # 保存图片id数据
    train_id_path = os.path.join(out_train_val_path, "train.txt")
    val_id_path = os.path.join(out_train_val_path, "val.txt")
    comment.save_id(train_id_path, train_image_id, val_id_path, val_image_id)


def convert_voc_to_textdataset_for_image(image_list, annotations_dir, label_out_dir, class_names, coordinatesType,
                                         labelType="class_id",
                                         show=True):
    '''
    label data format：
    coordinatesType:
    SSD  = [label_id,x,y,w,h]
    xywh = [label_id,x,y,w,h]
    xyxy = [label_id,xmin,ymin,xmax,ymax]
    xxyy = [label_id,xmin,xmax,ymin,ymax]
    YOLO = [label_id,x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    MMDET= [img_width,img_height,label_id,x,y,w,h]

    :param image_list: 图片列表
    :param annotations_dir: 图片对应annotations所在目录
    :param label_out_dir: label输出目录
    :param class_names:
    :param coordinatesType: 坐标类型：SSD,YOLO,MMDET格式
    :param show: 显示
    :return:
    '''
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)
    name_id_list = []
    nums = len(image_list)
    for i, image_path in enumerate(image_list):
        name_id = os.path.basename(image_path)[:-len(".jpg")]
        ann_name = name_id + '.xml'
        annotations_file = os.path.join(annotations_dir, ann_name)

        if not os.path.exists(image_path):
            print("no image:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        out_file = os.path.join(label_out_dir, name_id + ".txt")
        rects, class_name, class_id = pascal_voc.get_annotation(annotations_file, class_names, minAreaTH=500,
                                                                coordinatesType=coordinatesType)
        if len(rects) == 0 or len(class_name) == 0 or len(class_id) == 0:
            print("no class in annotations:{}".format(annotations_file))
            continue
        if labelType == "class_name":
            label = class_name
        elif labelType == "class_id":
            label = class_id
        content_list = [[c] + r for c, r in zip(label, rects)]
        name_id_list.append(name_id)
        file_processing.write_data(out_file, content_list, mode='w')
        if show:
            image = image_processing.read_image(image_path)
            image_processing.show_image_rects_text("image", image, rects, class_name)
        if i % 10 == 0 or i == len(image_list) - 1:
            print("processing image:{}/{}".format(i, len(image_list) - 1))
    return name_id_list


def text_dataset_for_image(annotations_dir, image_dir, label_out_dir, out_train_val_path, class_names,
                           coordinatesType, shuffle=True, labelType="class_id", show=True):
    '''
    label data format：
    SSD  = [label_id,x,y,w,h]
    YOLO = [label_id,x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    MMDET= [img_width,img_height,label_id,x,y,w,h]
    :param annotations_dir:
    :param image_dir:
    :param label_out_dir:
    :param out_train_val_path:
    :param class_names:
    :param coordinatesType: 坐标类型：SSD,YOLO,MMDET格式
    :param show:
    :return:
    '''
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])
    print("have {} images".format(len(image_list)))
    if shuffle:
        seeds = 100  # 固定种子,只要seed的值一样，后续生成的随机数都一样
        random.seed(seeds)
        random.shuffle(image_list)

    # 分割成train和val数据集
    factor = 0.95
    train_num = int(factor * len(image_list))
    train_image_list = image_list[:train_num]
    val_image_list = image_list[train_num:]

    # 转换label数据
    print("doing train data .....")
    train_image_id = convert_voc_to_textdataset_for_image(train_image_list, annotations_dir, label_out_dir, class_names,
                                                          coordinatesType, labelType=labelType, show=show)
    print("doing val data .....")
    val_image_id = convert_voc_to_textdataset_for_image(val_image_list, annotations_dir, label_out_dir, class_names,
                                                        coordinatesType, labelType=labelType, show=show)
    print("done...ok!")

    # 保存图片id数据
    train_id_path = os.path.join(out_train_val_path, "train.txt")
    val_id_path = os.path.join(out_train_val_path, "val.txt")
    comment.save_id(train_id_path, train_image_id, val_id_path, val_image_id)


def label_test(image_dir, filename, class_names=None):
    basename = os.path.basename(filename)[:-len('.txt')] + ".jpg"
    image_path = os.path.join(image_dir, basename)
    image = image_processing.read_image(image_path)
    data = file_processing.read_data(filename, split=" ")
    label_list, rect_list = file_processing.split_list(data, split_index=1)
    label_list = [l[0] for l in label_list]
    if class_names:
        name_list = file_processing.decode_label(label_list, class_names)
    else:
        name_list=label_list
    show_info = ["id:" + str(n)for n in name_list]
    rgb_image=image_processing.show_image_rects_text("object2", image, rect_list, show_info,color=(0,0,255),drawType="text",waitKey=1)
    rgb_image=image_processing.resize_image(rgb_image,900)
    image_processing.cv_show_image("object2",rgb_image)


def batch_label_test(label_dir, image_dir, classes):
    file_list = file_processing.get_files_list(label_dir, postfix=[".txt"])
    for filename in file_list:
        label_test(image_dir, filename, class_names=classes)


if __name__ == "__main__":
    # classes = ["BACKGROUND", 'face']
    # classes = ["BACKGROUND", 'PCwall']
    # classes = ["BACKGROUND","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # print("class_name:{}".format(classes))
    # DATASET_ROOT="/media/dm/dm2/project/dataset/face/wider_face_voc/"
    # annotations_dir = './dataset/VOC/Annotations'
    # label_out_dir = './dataset/VOC/label'
    # image_dir = "./dataset/VOC/JPEGImages"
    # out_train_val_path = "./data/voc"  # 输出 train/val 文件
    #
    # annotations_dir='/media/dm/dm2/project/dataset/VOCdevkit/VOC2007/Annotations'
    # label_out_dir= '/media/dm/dm2/project/dataset/VOCdevkit/VOC2007/label'
    # image_dir="/media/dm/dm2/project/dataset/VOCdevkit/VOC2007/JPEGImages"
    # out_train_val_path= "/media/dm/dm2/project/dataset/VOCdevkit/VOC2007"# 输出 train/val 文件

    # annotations_dir = '/media/dm/dm2/project/dataset/VOC_wall/Annotations'
    # label_out_dir = '/media/dm/dm2/project/dataset/VOC_wall/label'
    # image_dir = "/media/dm/dm2/project/dataset/VOC_wall/JPEGImages"
    # out_train_val_path = "/media/dm/dm2/project/dataset/VOC_wall"  # 输出 train/val 文件

    # classes = list(range(1, 50, 1))
    # classes = [str(i) for i in classes]
    classes = ["BACKGROUND","body", 'face']
    DATASET_ROOT = "/media/dm/dm2/project/dataset/face_recognition/NVR/face/NVR-Teacher2/"
    annotations_dir = DATASET_ROOT + 'Annotations'
    label_out_dir = DATASET_ROOT + 'label'
    image_dir = DATASET_ROOT + "JPEGImages"
    out_train_val_path = DATASET_ROOT  # 输出 train/val 文件

    # annotations_dir = '/media/dm/dm2/project/dataset/face/Annotations'
    # label_out_dir = '/media/dm/dm2/project/dataset/face/label'
    # image_dir = "/media/dm/dm2/project/dataset/face/JPEGImages"
    # out_train_val_path = "/media/dm/dm2/project/dataset/face/"  # 输出 train/val 文件

    coordinatesType = "SSD"
    show = True
    labelType="class_name"
    text_dataset_for_annotation(annotations_dir, image_dir, label_out_dir, out_train_val_path, classes, coordinatesType,labelType=labelType,
                                show=show)
    # text_dataset_for_image(annotations_dir, image_dir, label_out_dir, out_train_val_path, classes, coordinatesType,
    #                        show=show)

    batch_label_test(label_out_dir, image_dir, classes=None)
