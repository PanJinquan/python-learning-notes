# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : rename_tool.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-08-09 18:14:37
"""
import os
import os.path
from utils import file_processing


def rename_image_dir(image_list, prefix="ID_"):
    for image_path in image_list:
        dirname = os.path.dirname(image_path)
        label = image_path.split(os.sep)[-2]
        # basename=os.path.basename(image_path)
        index = 0
        newName = prefix + label + '_{}.jpg'.format(index)
        newpath = os.path.join(dirname, newName)
        while os.path.exists(newpath):
            index += 1
            newName = prefix + label + '_{}.jpg'.format(index)
            newpath = os.path.join(dirname, newName)

        print(image_path)
        print(newName)
        os.rename(image_path, newpath)


if __name__ == '__main__':
    # dir = '/media/dm/dm/project/dataset/face_recognition/NVR/face/NVRS/trainval'
    dataset_dir = "/media/dm/dm/project/dataset/face_recognition/X4/X4_Face20/"
    image_list = file_processing.get_files_list(dataset_dir, postfix=['*.jpg'])
    rename_image_dir(image_list)
