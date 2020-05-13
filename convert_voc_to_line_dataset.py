# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : vocDemo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-09 19:10:16
"""
import os
import random
from utils import file_processing, image_processing
from modules.dataset_tool import face_body
from modules.dataset_tool.voc_tools import pascal_voc


# for SSD  label，the first label is BACKGROUND：
# classes = ["BACKGROUND","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# for YOLO label,ignore the BACKGROUND
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# for wall
# classes = ["PCwall"]
# classes = ["BACKGROUND","person", "dog"]
# print("class_name:{}".format(classes))


def convert_voc_to_linedataset(annotations_dir, image_list, class_names, out_filename, coordinatesType, show=True):
    '''
    label data format：
    SSD  = [label_id,x,y,w,h]
    YOLO = [label_id,x_center/img_width ,y_center/img_height ,width/img_width ,height/img_height]
    MMDET= [img_width,img_height,label_id,x,y,w,h]

    line_image_label:[image_path,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id,...]
    :param annotations_dir: 图片对应annotations所在目录
    :param image_list: 图片列表
    :param class_names:
    :param out_filename
    :param coordinatesType: 坐标类型：SSD,YOLO,MMDET格式
    :param show: 显示
    :return:
    '''
    with open(out_filename, mode='w', encoding='utf-8') as f:
        for i, image_path in enumerate(image_list):
            image_name = os.path.basename(image_path)
            name_id = image_name[:-len(".jpg")]
            ann_name = name_id + '.xml'
            annotations_file = os.path.join(annotations_dir, ann_name)

            if not os.path.exists(image_path):
                print("no image_dict:{}".format(image_path))
                continue
            if not os.path.exists(annotations_file):
                print("no annotations:{}".format(annotations_file))
                continue
            rects, class_name, class_id = pascal_voc.get_annotation(annotations_file,
                                                                    class_names,
                                                                    minAreaTH=0,
                                                                    coordinatesType=coordinatesType)
            if len(rects) == 0 or len(class_name) == 0 or len(class_id) == 0:
                print("no class in annotations:{}".format(annotations_file))
                continue
            line_image_label = pascal_voc.convert_to_linedataset(image_name, rects, class_id)
            # contents = face_body_tools.convert_pyramidbox_data(image_path, boxList, classes)
            contents_line = " ".join('%s' % id for id in line_image_label)
            f.write(contents_line + "\n")

            if show:
                image = image_processing.read_image(image_path)
                image_processing.show_image_rects_text("image_dict", image, rects, class_name)
            if i % 10 == 0 or i == len(image_list) - 1:
                print("processing image_dict:{}/{}".format(i, len(image_list) - 1))


def linedataset_for_image(shuffle=True):
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
    show = False
    coordinatesType = "SSD"
    # PCwall
    # train_filename = "/media/dm/dm2/project/dataset/VOC_wall/train.txt"
    # val_filename = "/media/dm/dm2/project/dataset/VOC_wall/val.txt"
    # annotations_dir = '/media/dm/dm2/project/dataset/VOC_wall/Annotations'
    # image_dir = "/media/dm/dm2/project/dataset/VOC_wall/JPEGImages"

    # VOC
    # DATA_ROOT="/media/dm/dm2/project/dataset/VOCdevkit/VOC2007/"
    # annotations_dir=DATA_ROOT+'Annotations'
    # image_dir=DATA_ROOT+"JPEGImages"
    # train_filename = DATA_ROOT+"train.txt"
    # val_filename = DATA_ROOT+"val.txt"

    # widerface
    DATA_ROOT = "/media/dm/dm2/project/dataset/face/wider_face_voc/"
    annotations_dir = DATA_ROOT + 'Annotations'
    image_dir = DATA_ROOT + "JPEGImages"
    train_filename = DATA_ROOT + "train.txt"
    val_filename = DATA_ROOT + "val.txt"

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
    convert_voc_to_linedataset(annotations_dir, train_image_list, classes, train_filename, coordinatesType,
                               show)
    print("doing val data .....")
    convert_voc_to_linedataset(annotations_dir, val_image_list, classes, val_filename, coordinatesType,
                               show)
    print("done...ok!")
    # test
    linedataset_test(train_filename, classes, image_dir=image_dir, show=True)


def linedataset_test(filename, classes, image_dir=None, show=True):
    '''
    label_data:[image_path,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id]
    :param filename:
    :param image_dir:
    :param show:
    :return:
    '''
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            image_id, box, label = face_body.read_line_image_label(line)
            if show:
                if image_dir:
                    image_path = os.path.join(image_dir, image_id)
                else:
                    image_path = image_id
                image = image_processing.read_image(image_path)
                name_list = file_processing.decode_label(label, classes)
                image_processing.show_image_bboxes_text("image_dict", image, box, name_list)


if __name__ == "__main__":
    # classes = ["BACKGROUND", 'face']
    # classes = ["BACKGROUND", 'PCwall']
    # classes = ["BACKGROUND","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # classes=pascal_voc.VOC_CLASSES_BG

    # classes=["BACKGROUND","face"]
    # print("class_name:{}".format(classes))
    # linedataset_for_image()
    classes = ["BACKGROUND", 'face', "body"]
    DATASET_ROOT = "/media/dm/dm2/XMC/FaceDataset/NVR/NVR-Teacher2/"
    # annotations_dir = DATASET_ROOT + 'Annotations'
    # label_out_dir = DATASET_ROOT + 'label'
    image_dir = DATASET_ROOT + "image_dict"

    filename = DATASET_ROOT + "teacher_data_anno.txt"
    linedataset_test(filename, classes, image_dir)
