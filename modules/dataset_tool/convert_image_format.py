# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : convert_image_format.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-09-23 16:55:28
"""

import os
import numpy as np
import shutil
import os.path
from utils import file_processing, image_processing


def convert_image_format(image_dir, dest_dir, resize_width=None, dest_format='.jpg'):
    image_id = file_processing.get_sub_directory_list(image_dir)
    for id in image_id:
        image_list = file_processing.get_files_list(os.path.join(image_dir, id),
                                                    postfix=['*.jpg', "*.jpeg", '*.png', "*.JPG"])
        print("processing :{}".format(id))
        for src_path in image_list:
            basename = os.path.basename(src_path).split('.')[0]
            image = image_processing.read_image_gbk(src_path, resize_width=resize_width)
            dest_path = file_processing.create_dir(dest_dir, id, basename + dest_format)
            file_processing.create_file_path(dest_path)
            image_processing.save_image(dest_path, image)


if __name__ == "__main__":
    resize_width = 500
    # image_dir = '/media/dm/dm1/FaceDataset/lexue/lexue1/val-src'
    # dest_dir = '/media/dm/dm1/FaceDataset/lexue/lexue1/val-src2'
    # image_dir = '/media/dm/dm1/FaceDataset/lexue/lexue2/facebank'
    # dest_dir = '/media/dm/dm1/FaceDataset/lexue/lexue2/facebank2'
    image_dir = "/media/dm/dm1/FaceDataset/lexue/lexue_teacher"
    dest_dir = "/media/dm/dm1/FaceDataset/lexue/lexue_teacher2"
    convert_image_format(image_dir, dest_dir, resize_width)
