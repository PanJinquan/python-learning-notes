# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : combinations.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-12 14:21:18
    @url    : https://blog.csdn.net/specter11235/article/details/71189486
"""
import os
import numpy as np
import itertools
from utils import file_processing,image_processing
import tqdm

def get_combinations_pair_data(image_dir, weight=1):
    '''
    get image_dir image list,combinations image
    :param image_dir:
    :return:
    '''
    _ID = True
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])
    nums=len(image_list)
    print("have {} images and {} combinations".format(nums,nums*(nums-1)/2))
    pair_issame = []
    for paths in itertools.combinations(image_list, 2):
        image_path1, image_path2 = paths
        name1 = os.path.basename(image_path1)
        name2 = os.path.basename(image_path2)
        label1 = image_path1.split(os.sep)[-2]
        label2 = image_path2.split(os.sep)[-2]
        if label1 == label2:
            issame = 1
        else:
            issame = 0
        if _ID:
            image_id1 = os.path.join(label1, name1)
            image_id2 = os.path.join(label2, name2)
            pair_issame.append([image_id1, image_id2, issame])
        else:
            pair_issame.append([image_path1, image_path2, issame])

    pair_issame = np.asarray(pair_issame)
    if weight > 0:
        pair_issame = pair_issame[np.lexsort(pair_issame.T)]
        pair_issame_0 = pair_issame[pair_issame[:, -1] == "0", :]
        pair_issame_1 = pair_issame[pair_issame[:, -1] == "1", :]
        num_pair_issame_1 = len(pair_issame_1)
        num_pair_issame_0 = len(pair_issame_0)  # pair_issame_0.shape[0]
        select_nums = num_pair_issame_1 * weight
        per = np.random.permutation(num_pair_issame_0)[:select_nums]  # 打乱后的行号
        pair_issame_0 = pair_issame_0[per, :]  # 获取打乱后的训练数据
        pair_issame = np.concatenate([pair_issame_0, pair_issame_1], axis=0)
        print("pair_issame_0 nums:{}".format(len(pair_issame_0)))
        print("pair_issame_1 nums:{}".format(len(pair_issame_1)))

    # image_list1 = pair_issame[:, 0]
    # image_list2 = pair_issame[:, 1]
    # issame_list = pair_issame[:, 2]
    print("have {} pairs".format(len(pair_issame)))
    return pair_issame


def save_pair_data(filename, content_list):
    file_processing.write_data(filename, content_list, mode='w')



if __name__ == "__main__":
    # NVR VAL faceDataset
    # dataset="/media/dm/dm/project/dataset/face_recognition/NVR/facebank/"
    # image_dir = dataset+"NVR_3_20190605_1005_VAL"
    # pair_issame = get_combinations_pair_data(image_dir)
    # save_pair_data(dataset+"nvr_pair_data.txt",pair_issame)

    # CASIA-FaceV5 faceDataset
    dataset="/media/dm/dm2/project/dataset/face_recognition/CASIA-FaceV5/"
    image_dir = dataset+"CASIA-Faces"
    pair_issame = get_combinations_pair_data(image_dir)
    save_pair_data(dataset+"casia_pair_data.txt",pair_issame)

    #
    # dataset="/media/dm/dm2/project/dataset/face_recognition/celebs_add_movies/"
    # image_dir = dataset+"Asian_Faces"
    # pair_issame = get_combinations_pair_data(image_dir)
    # save_pair_data(dataset+"asian_faces_pair_data.txt",pair_issame)