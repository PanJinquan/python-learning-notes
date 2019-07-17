# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : comment.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-25 15:48:15
"""

import os
from utils import file_processing


def save_id(train_id_path, train_id, val_id_path, val_id):
    if not os.path.exists(os.path.dirname(train_id_path)):
        os.makedirs(os.path.dirname(train_id_path))
    if not os.path.exists(os.path.dirname(val_id_path)):
        os.makedirs(os.path.dirname(val_id_path))

    # 保存图片id数据
    file_processing.write_list_data(train_id_path, train_id, mode="w")
    file_processing.write_list_data(val_id_path, val_id, mode="w")
    print("train num:{},save path:{}".format(len(train_id), train_id_path))
    print("val   num:{},save path:{}".format(len(val_id), val_id_path))
