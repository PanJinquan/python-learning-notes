# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : ramdom_resize.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-09-04 21:30:09
"""
import os
import random
from utils import image_processing, file_processing


def random_resize(image, range_size):
    # random.seed(100)
    r = int(random.uniform(range_size[0], range_size[1]))
    # size = (r, r)
    image = image_processing.resize_image(image, resize_height=r, resize_width=r)
    return image


def random_resize_image_list(image_dir, out_dir, range_size):
    image_list, label_list = file_processing.get_files_labels(image_dir, postfix=['*.jpg'])
    print("hace {} images".format(len(image_list)))
    for image_path, lable in zip(image_list, label_list):
        basename = os.path.basename(image_path)
        image = image_processing.read_image(image_path)
        re_image = random_resize(image, range_size)
        print("image_dict shape:{}".format(re_image.shape))
        # out_path = os.path.join(out_dir, lable, basename)
        out_path = file_processing.create_dir(out_dir, lable, basename)
        image_processing.save_image(out_path, re_image)


def resize_image_list(image_dir, out_dir, resize_height=None, resize_width=None):
    image_list, label_list = file_processing.get_files_labels(image_dir, postfix=['*.jpg',"*.JPG"])
    print("hace {} images".format(len(image_list)))
    for image_path, lable in zip(image_list, label_list):
        basename = os.path.basename(image_path)
        image = image_processing.read_image(image_path,
                                            resize_height=resize_height,
                                            resize_width=resize_width)
        print("image_dict shape:{}".format(image.shape))
        # out_path = os.path.join(out_dir, lable, basename)
        out_path = file_processing.create_dir(out_dir, lable, basename)
        image_processing.save_image(out_path, image)


if __name__ == "__main__":
    # image_dir = "/media/dm/dm/FaceDataset/X4/X4_Face50_Alig/val"
    # out_dir = "/media/dm/dm/FaceDataset/X4/X4_Face50_low_25_40/val"
    image_dir = "/media/dm/dm1/FaceRecognition/dataset/honghe/test/honghe/portrait"
    out_dir = "/media/dm/dm1/FaceRecognition/dataset/honghe/test/honghe/portrait2"
    range_size = [40, 40]
    # random_resize_image_list(image_dir, out_dir, range_size)
    resize_image_list(image_dir, out_dir, resize_height=600, resize_width=None)

