# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : face_body_tesy.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-21 11:26:48
"""

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
from modules.dataset_tool import face_body, comment


def face_body_linedataset_for_image(shuffle=True):
    annotations_dir = '/media/dm/dm2/project/dataset/face_body/face_body_dataset/Annotations'
    image_dir = "/media/dm/dm2/project/dataset/face_body/face_body_dataset/images"
    train_filename = "/media/dm/dm2/project/dataset/face_body/face_body_dataset/train.txt"
    val_filename = "/media/dm/dm2/project/dataset/face_body/face_body_dataset/val.txt"

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
    convert_to_linedataset(annotations_dir, train_image_list, classes, train_filename)
    print("doing val data .....")
    convert_to_linedataset(annotations_dir, val_image_list, classes, val_filename)

    print("done...ok!")
    linedataset_test(train_filename, image_dir=image_dir, show=True)
    # annotations_list = file_processing.get_files_list(annotations_dir, postfix=["*.json"])


def convert_to_linedataset(annotations_dir, image_list, class_names, out_filename):
    '''
    label_data:[image_path,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id]
    :param annotations_dir:
    :param image_dir:
    :param class_names:
    :param out_filename:
    :return:
    '''

    print("image_list nums:{}".format(len(image_list)))
    with open(out_filename, mode='w', encoding='utf-8') as f:
        for i, image_path in enumerate(image_list):
            image_name = os.path.basename(image_path)
            name_id = image_name[:-len(".jpg")]
            annotations_file = os.path.join(annotations_dir, name_id + ".json")
            if not os.path.exists(image_path):
                print("no image:{}".format(image_path))
                continue
            if not os.path.exists(annotations_file):
                print("no annotations:{}".format(annotations_file))
                continue
            boxList = face_body.parse_annotation(annotations_file, class_names)
            if not boxList:
                print("no class in annotations:{}".format(annotations_file))
                continue
            line_image_label = face_body.convert_to_linedataset(image_name, boxList, classes)
            # contents = face_body_tools.convert_pyramidbox_data(image_path, boxList, classes)
            contents_line = " ".join('%s' % id for id in line_image_label)
            f.write(contents_line + "\n")
            if i % 10 == 0 or i == len(image_list) - 1:
                print("processing image:{}/{}".format(i, len(image_list)-1))


def linedataset_test(filename, image_dir=None, show=True):
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
                image_processing.show_image_bboxes_text("image", image, box, label)


def convert_facebody_to_textdataset(image_list, annotations_dir, label_out_dir, class_names, coordinatesType,
                                    show=False):
    '''
    label data format：
    SSD  = [label_id,x,y,w,h]
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
        ann_name = name_id + '.json'
        annotations_file = os.path.join(annotations_dir, ann_name)

        if not os.path.exists(image_path):
            print("no image:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        out_file = os.path.join(label_out_dir, name_id + ".txt")
        # rects, class_name, class_id = pascal_voc.get_annotation(annotations_file, class_names, coordinatesType)
        image = image_processing.read_image(image_path)
        image_shape = image.shape
        rects, class_name, class_id = face_body.get_annotation(annotations_file, class_names, image_shape,
                                                               coordinatesType)
        if len(rects) == 0 or len(class_name) == 0 or len(class_id) == 0:
            print("no class in annotations:{}".format(annotations_file))
            continue
        content_list = [[c] + r for c, r in zip(class_id, rects)]
        name_id_list.append(name_id)
        file_processing.write_data(out_file, content_list, mode='w')
        if show:
            image = image_processing.read_image(image_path)
            image_processing.show_image_rects_text("image", image, rects, class_name)
        if i % 10 == 0 or i == len(image_list) - 1:
            print("processing image:{}/{}".format(i, len(image_list) - 1))
    return name_id_list


def face_body_for_image(shuffle=False):
    out_train_val_path = "/media/dm/dm2/project/dataset/face_body/SSD"  # 输出 train/val 文件

    annotations_dir = '/media/dm/dm2/project/dataset/face_body/face_body_dataset/Annotations'
    # face_body_test(annotations_dir, image_dir, classes, coordinatesType="SSD", show=True)
    label_out_dir = "/media/dm/dm2/project/dataset/face_body/SSD/label"
    image_dir = "/media/dm/dm2/project/dataset/face_body/SSD/trainval"
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

    train_image_id = convert_facebody_to_textdataset(train_image_list, annotations_dir, label_out_dir, classes,
                                                     coordinatesType="SSD",
                                                     show=False)
    val_image_id = convert_facebody_to_textdataset(val_image_list, annotations_dir, label_out_dir, classes,
                                                   coordinatesType="SSD",
                                                   show=False)
    print("done...ok!")
    # # 保存图片id数据
    train_id_path = os.path.join(out_train_val_path, "train.txt")
    val_id_path = os.path.join(out_train_val_path, "val.txt")
    comment.save_id(train_id_path, train_image_id, val_id_path, val_image_id)
    batch_label_test(label_out_dir, image_dir, classes)


def face_body_test(annotations_dir, image_dir, class_names, show=True):
    '''
    :param annotations_dir:
    :param image_dir:
    :param class_names:
    :param show:
    :return:
    '''
    annotations_list = file_processing.get_files_list(annotations_dir, postfix=["*.json"])
    print("have {} annotations files".format(len(annotations_list)))
    for i, annotations_file in enumerate(annotations_list):
        name_id = os.path.basename(annotations_file)[:-len(".json")]
        image_name = name_id + ".jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print("no image:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        boxList = face_body.get_annotation(annotations_file, class_names)
        if not boxList:
            print("no class in annotations:{}".format(annotations_file))
            continue
        if show:
            image = image_processing.read_image(image_path)
            # image_processing.show_image_rects_text("image", image, rects, class_name)
            image_processing.show_boxList("image", boxList, image)


def label_test(image_dir, filename, class_names):
    basename = os.path.basename(filename)[:-len('.txt')] + ".jpg"
    image_path = os.path.join(image_dir, basename)
    image = image_processing.read_image(image_path)
    data = file_processing.read_data(filename, split=" ")
    label_list, rect_list = file_processing.split_list(data, split_index=1)
    label_list = [l[0] for l in label_list]
    name_list = file_processing.decode_label(label_list, class_names)
    image_processing.show_image_rects_text("object2", image, rect_list, name_list)


def batch_label_test(label_dir, image_dir, classes):
    file_list = file_processing.get_files_list(label_dir, postfix=[".txt"])
    for filename in file_list:
        label_test(image_dir, filename, class_names=classes)


if __name__ == "__main__":
    classes = ["BACKGROUND", "脸", '上半身']
    print("class_name:{}".format(classes))
    # convert_face_body_for_image()
    face_body_linedataset_for_image()
