# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : dmai_test_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-15 11:55:32
"""
import os
import numpy as np
import itertools
from utils import file_processing, image_processing
import tqdm
import PIL.Image as Image
from pathlib import Path


def create_pair_data(image_dir, pair_num=0):
    '''
    get image_dir image_dict list,combinations image_dict
    :param image_dir:
    :return:
    '''
    _ID = True
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg","png"])
    nums = len(image_list)
    print("have {} images and {} combinations".format(nums, nums * (nums - 1) / 2))
    pair_issame = []
    for paths in itertools.combinations(image_list, 2):
        image_path1, image_path2 = paths
        name1 = os.path.basename(image_path1)
        name2 = os.path.basename(image_path2)
        # label1 = image_path1.split(os.sep)[-2]
        # label2 = image_path2.split(os.sep)[-2]
        label1 = name1.split("_")[0]
        label2 = name2.split("_")[0]
        if label1 == label2:
            issame = 1
        else:
            issame = 0
        if _ID:
            # image_id1 = os.path.join(label1, name1)
            # image_id2 = os.path.join(label2, name2)
            pair_issame.append([name1, name2, issame])
        else:
            pair_issame.append([image_path1, image_path2, issame])

    pair_issame = np.asarray(pair_issame)
    pair_issame = pair_issame[np.lexsort(pair_issame.T)]
    if pair_num is None:
        return pair_issame

    pair_issame_0 = pair_issame[pair_issame[:, -1] == "0", :]
    pair_issame_1 = pair_issame[pair_issame[:, -1] == "1", :]
    num_pair_issame_1 = len(pair_issame_1)
    num_pair_issame_0 = len(pair_issame_0)  # pair_issame_0.shape[0]
    select_nums = int(pair_num / 2)
    if select_nums == 0:
        select_nums = num_pair_issame_1
    else:
        if select_nums > num_pair_issame_1:
            raise Exception(
                "pair_nums({}) must be less than num_pair_issame_1({})".format(select_nums, num_pair_issame_1))
    np.random.seed(100)
    index_0 = np.random.permutation(num_pair_issame_0)[:select_nums]  # 打乱后的行号
    index_1 = np.random.permutation(num_pair_issame_1)[:select_nums]  # 打乱后的行号
    pair_issame_0 = pair_issame_0[index_0, :]  # 获取打乱后的训练数据
    pair_issame_1 = pair_issame_1[index_1, :]  # 获取打乱后的训练数据
    pair_issame = np.concatenate([pair_issame_0, pair_issame_1], axis=0)
    print("pair_issame_0 nums:{}".format(len(pair_issame_0)))
    print("pair_issame_1 nums:{}".format(len(pair_issame_1)))

    # image_list1 = pair_issame[:, 0]
    # image_list2 = pair_issame[:, 1]
    # issame_list = pair_issame[:, 2]
    print("have {} pairs".format(len(pair_issame)))
    return pair_issame


if __name__ == "__main__":
    # NVR VAL faceDataset
    dataset = '/media/dm/dm2/XMC/FaceDataset/X4/X4_Face20_Crop/'
    # dataset = '/media/dm/dm1/FaceDataset/X4/DMAI_Alig/'
    # lexue
    image_dir = dataset + "trainval"
    pair_filename = dataset + "x4_pair_data.txt"
    pair_issame = create_pair_data(image_dir, pair_num=0)
    file_processing.write_data(pair_filename, pair_issame, mode='w')
