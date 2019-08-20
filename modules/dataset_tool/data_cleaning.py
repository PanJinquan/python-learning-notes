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


def data_cleaning(src_image_dir, dest_image_dir, min_nums=0):
    '''
    clear sample data in src_image_dir,which sample nums must be more than min_num
    :param src_image_dir:
    :param dest_image_dir: 
    :param min_nums: 
    :return: 
    '''

    image_list, image_label = file_processing.get_files_labels(src_image_dir, postfix=["*.jpg"])
    p = sample_statistics.count_data_info_pd(image_label, plot=False)
    for i, (image_path, label) in enumerate(zip(image_list, image_label)):
        name = os.path.basename(image_path)
        # label_nums = image_label.count(label)
        label_nums = p[label]
        if label_nums < min_nums:
            continue
        out_path = os.path.join(dest_image_dir, label)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, name)
        file_processing.copyfile(image_path, out_path)
        if i % 100 == 0 or i == len(image_list) - 1:
            print("processing:{}/{}".format(i, len(image_list)))
        time.sleep(0.01)


def move_merge_dirs(source_dir, dest_dir, merge_same=False):
    '''
    move and merge files, move/merge files from source_dir to dest_dir, set merge_same True or False to ignore the same dir
    :param source_dir:
    :param dest_dir:
    :return:
    '''
    for path, dirs, files in os.walk(source_dir, topdown=False):
        dest_path = os.path.join(
            dest_dir,
            os.path.relpath(path, source_dir)
        )
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        else:
            id = os.path.basename(dest_path)
            print("have exists same dir:{}, merge_same:{}".format(id, merge_same))
            if not merge_same:
                continue
        for filename in files:
            os.rename(
                os.path.join(path, filename),
                os.path.join(dest_path, filename)
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
    source_dir = '/media/dm/dm1/git/python-learning-notes/dataset/dataset2'
    dest_dir = '/media/dm/dm1/git/python-learning-notes/dataset/dataset'
    move_merge_dirs(source_dir, dest_dir, merge_same=False)
    # images_list = file_processing.get_files_list(file_dir, postfix=['*.jpg'])
    # isValidImage(images_list, sizeTh=1000, isRemove=False)
