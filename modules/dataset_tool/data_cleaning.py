# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : data_cleaning.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-17 12:45:18
"""

import os, sys

sys.path.append("/home/panjinquan/project/python-learning-notes")
import time
from utils import file_processing, sample_statistics
import pandas as pd
import PIL.Image as Image


def data_cleaning(src_image_dir, dest_image_dir, nums_threashold=0):
    '''
    clear data
    :param src_image_dir:
    :return:
    '''
    image_list, image_label = file_processing.get_files_labels(src_image_dir, postfix=["*.jpg"])
    p = sample_statistics.count_data_info_pd(image_label, plot=False)
    for i, (image_path, label) in enumerate(zip(image_list, image_label)):
        name = os.path.basename(image_path)
        # label_nums = image_label.count(label)
        label_nums = p[label]
        if label_nums < nums_threashold:
            continue
        out_path = os.path.join(dest_image_dir, label)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, name)
        file_processing.copyfile(image_path, out_path)
        if i % 100 == 0 or i == len(image_list) - 1:
            print("processing:{}/{}".format(i, len(image_list)))
        time.sleep(0.01)


def move_merge_dirs(source_root, dest_root):
    '''
    move source_root file to dest_root
    :param source_root:
    :param dest_root:
    :return:
    '''
    for path, dirs, files in os.walk(source_root, topdown=False):
        dest_dir = os.path.join(
            dest_root,
            os.path.relpath(path, source_root)
        )
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        else:
            id = os.path.basename(dest_dir)
            print("have exists same dir:{} ".format(id))
        for filename in files:
            os.rename(
                os.path.join(path, filename),
                os.path.join(dest_dir, filename)
            )


def isValidImage(images_list, sizeTh=1000, isRemove=False):
    ''' 去除不存的文件和文件过小的文件列表
    :param images_list:
    :param sizeTh: 文件大小阈值,单位：字节B，默认1000B
    :param isRemove: 是否在硬盘上删除被损坏的原文件
    :return:
    '''
    i = 0
    while i < len(images_list):
        path = images_list[i]
        # 判断文件是否存在
        if not (os.path.exists(path)):
            print(" non-existent file:{}".format(path))
            images_list.pop(i)
            continue
        # 判断文件是否为空
        f_size = os.path.getsize(path)
        if f_size < sizeTh:
            print(" empty file:{}".format(path))
            if isRemove:
                os.remove(path)
                print(" info:----------------remove image:{}".format(path))
            images_list.pop(i)
            continue
        # 判断图像文件是否损坏
        try:
            Image.open(path).verify()
        except:
            print(" damaged image:{}".format(path))
            if isRemove:
                os.remove(path)
                print(" info:----------------remove image:{}".format(path))
            images_list.pop(i)
            continue
        i += 1
    return images_list


if __name__ == "__main__":
    # image_dir = "/data/yehongjiang/douban/celebs_add_movies"
    # # dest_image_dir = "/data/panjinquan/douban/celebs_add_movies"
    # image_dir = "/media/dm/dm2/project/dataset/face_recognition/celebs_add_movies/val_face"
    # dest_image_dir = "/media/dm/dm2/project/dataset/face_recognition/celebs_add_movies/Asian_Faces"
    # data_cleaning(image_dir, dest_image_dir, nums_threashold=10)
    file_dir = '/media/dm/dm1/project/backup/log/1498491877124107'
    images_list = file_processing.get_files_list(file_dir, postfix=['*.jpg'])
    isValidImage(images_list, sizeTh=1000, isRemove=False)
