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
from utils import file_processing, image_processing
import tqdm
import PIL.Image as Image
from pathlib import Path


def get_combinations_pair_data(image_dir, pair_num=0):
    '''
    get image_dir image list,combinations image
    :param image_dir:
    :return:
    '''
    select_nums = int(pair_num / 2)
    _ID = True
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])
    nums = len(image_list)
    print("have {} images and {} combinations".format(nums, nums * (nums - 1) / 2))
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
    pair_issame = pair_issame[np.lexsort(pair_issame.T)]
    pair_issame_0 = pair_issame[pair_issame[:, -1] == "0", :]
    pair_issame_1 = pair_issame[pair_issame[:, -1] == "1", :]
    num_pair_issame_1 = len(pair_issame_1)
    num_pair_issame_0 = len(pair_issame_0)  # pair_issame_0.shape[0]
    if select_nums == 0 or select_nums is None:
        select_nums = num_pair_issame_1
    else:
        if select_nums > num_pair_issame_1:
            raise Exception(
                "pair_nums({}) must be less than num_pair_issame_1({})".format(select_nums, num_pair_issame_1))

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


def get_combinations_pair_data_for_x4_data(image_dir, pair_num=0):
    '''
    get image_dir image list,combinations image
    :param image_dir:
    :return:
    '''
    select_nums = int(pair_num / 2)
    _ID = True
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])
    nums = len(image_list)
    print("have {} images and {} combinations".format(nums, nums * (nums - 1) / 2))
    pair_issame = []
    for paths in itertools.combinations(image_list, 2):
        image_path1, image_path2 = paths
        name1 = os.path.basename(image_path1)
        name2 = os.path.basename(image_path2)
        index1 = name1.find("ID")
        index2 = name2.find("ID")
        if (index1 == -1 and index2 == -1):
            continue
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
    pair_issame = pair_issame[np.lexsort(pair_issame.T)]
    pair_issame_0 = pair_issame[pair_issame[:, -1] == "0", :]
    pair_issame_1 = pair_issame[pair_issame[:, -1] == "1", :]
    num_pair_issame_1 = len(pair_issame_1)
    num_pair_issame_0 = len(pair_issame_0)  # pair_issame_0.shape[0]
    if select_nums == 0 or select_nums is None:
        select_nums = num_pair_issame_1
    else:
        if select_nums > num_pair_issame_1:
            raise Exception(
                "pair_nums({}) must be less than num_pair_issame_1({})".format(select_nums, num_pair_issame_1))

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


def save_pair_data(filename, content_list):
    file_processing.write_data(filename, content_list, mode='w')


def read_pair_data(filename, split=True):
    content_list = file_processing.read_data(filename)
    if split:
        content_list = np.asarray(content_list)
        faces_list1 = content_list[:, :1].reshape(-1)
        faces_list2 = content_list[:, 1:2].reshape(-1)
        # convert to 0/1
        issames_data = np.asarray(content_list[:, 2:3].reshape(-1), dtype=np.int)
        issames_data = np.where(issames_data > 0, 1, 0)
        faces_list1 = faces_list1.tolist()
        faces_list2 = faces_list2.tolist()
        issames_data = issames_data.tolist()
        return faces_list1, faces_list2, issames_data
    return content_list


def convert_image_to_bcolz(pair_filename, image_dir, save_dir, input_size=[112, 112]):
    from torchvision import transforms as trans
    import bcolz
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    faces_list1, faces_list2, issames_data = read_pair_data(pair_filename)
    print("have {} pair".format(len(issames_data)))
    print("have {} pair".format(len(faces_list1)))

    issames_data = np.array(issames_data)
    issames_data = np.where(issames_data > 0, True, False)

    data = bcolz.fill(shape=[len(faces_list1) + len(faces_list2), 3, input_size[0], input_size[1]], dtype=np.float32,
                      rootdir=save_dir, mode='w')
    for i, (face1_path, face2_path, issame) in enumerate(zip(faces_list1, faces_list2, issames_data)):
        # pred_id, pred_scores = faceRec.predict(faces)
        # 或者使用get_faces_embedding()获得embedding，再比较compare_embedding()
        if image_dir:
            face1_path = os.path.join(image_dir, face1_path)
            face2_path = os.path.join(image_dir, face2_path)
        face1 = image_processing.read_image_gbk(face1_path, colorSpace="BGR")
        face2 = image_processing.read_image_gbk(face2_path, colorSpace="BGR")
        face1 = image_processing.resize_image(face1, resize_height=input_size[0], resize_width=input_size[1])
        face2 = image_processing.resize_image(face2, resize_height=input_size[0], resize_width=input_size[1])
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # image_processing.cv_show_image("image",img)
        face1 = Image.fromarray(face1.astype(np.uint8))
        face2 = Image.fromarray(face2.astype(np.uint8))
        data[i * 2, ...] = transform(face1)
        data[i * 2 + 1, ...] = transform(face2)
        if i % 100 == 0:
            print('loading bin', i)

    print(data.shape)
    np.save(str(save_dir) + '_list', issames_data)


if __name__ == "__main__":
    # NVR VAL faceDataset
    dataset = '/media/dm/dm2/XMC/FaceDataset/X4/X4_Face20_Crop/'
    # dataset = '/media/dm/dm2/project/dataset/face_recognition/NVR/face/NVR-Teacher/'
    # dataset = "/media/dm/dm1/project/dataset/face_recognition/NVR/face/NVR1/"
    # dataset = "/media/dm/dm/project/dataset/face_recognition/NVR/face/NVRS/"
    image_dir = dataset + "trainval"
    pair_num=0
    pair_filename = dataset + "x4_pair_data.txt"
    # pair_issame = get_combinations_pair_data(image_dir, pair_num)
    pair_issame = get_combinations_pair_data_for_x4_data(image_dir, pair_num)

    save_pair_data(pair_filename, pair_issame)
    # convert_image_to_bcolz(pair_filename, image_dir, save_dir=dataset + "nvr", input_size=[112, 112])

    # CASIA-FaceV5 faceDataset
    # dataset = "/media/dm/dm2/project/dataset/face_recognition/CASIA-FaceV5/"
    # image_dir = dataset + "CASIA-Faces"
    # pair_filename = dataset + "casia_pair_data6000.txt"
    # pair_issame = get_combinations_pair_data(image_dir, pair_nums=6000)
    # save_pair_data(pair_filename, pair_issame)

    #
    # dataset="/media/dm/dm2/project/dataset/face_recognition/celebs_add_movies/"
    # image_dir = dataset+"Asian_Faces"
    # pair_issame = get_combinations_pair_data(image_dir)
    # save_pair_data(dataset+"asian_faces_pair_data.txt",pair_issame)
    convert_image_to_bcolz(pair_filename, image_dir, save_dir=dataset + "x4", input_size=[112, 112])
