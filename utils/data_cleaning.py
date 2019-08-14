# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : data_cleaning.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-17 12:45:18
"""

import os, sys
import time
from utils import file_processing, sample_statistics, image_processing
import pandas as pd
import os

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



if __name__ == "__main__":
    # image_dir = "/data/yehongjiang/douban/celebs_add_movies"
    # dest_image_dir = "/data/panjinquan/douban/celebs_add_movies"
    image_dir = "/media/dm/dm2/project/dataset/face_recognition/celebs_add_movies/val_face"
    dest_image_dir = "/media/dm/dm1/project/dataset/face_recognition/celebs_add_movies/Asian_Faces"
    # data_cleaning(image_dir, dest_image_dir, nums_threashold=10)
    source_root = '/media/dm/dm1/project/python-learning-notes/dataset/dataset'
    dest_root = '/media/dm/dm1/project/python-learning-notes/dataset/dataset2'
    move_merge_dirs(source_root, dest_root)
