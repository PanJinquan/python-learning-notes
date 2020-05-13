# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import argparse
import cv2
import os
import numpy as np
import skimage.data
import skimage.transform
from utils import image_processing, file_processing
import datetime

import torch
import torchvision.models as models
from collections import OrderedDict
import glob


def get_prefix_files(file_dir, prefix):
    """
    :param file_dir:
    :param prefix: "best*"
    :return:
    """
    file_list = glob.glob(os.path.join(file_dir, prefix))
    return file_list


def remove_prefix_files(file_dir, prefix):
    """
    :param file_dir:
    :param prefix: "best*"
    :return:
    """
    file_list = get_prefix_files(file_dir, prefix)
    for file in file_list:
        file_processing.remove_file(file)


if __name__ == "__main__":
    file_dir = "/media/dm/dm2/FaceRecognition/anti-spoofing/torch-anti-spoofing-pipeline/work_space/multi_mobilenet_v2_color_ir_train_20200511140154/model"
    remove_prefix_files(file_dir, prefix="best*")
