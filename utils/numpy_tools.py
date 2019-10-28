# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : numpy_tools.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-23 12:01:36
"""
import numpy as np
import math


def matching_data_vecror(data, vector):
    '''
    从data中匹配vector向量，查找出现vector的index,如：
    data = [[1., 0., 0.],[0., 0., 0.],[2., 0., 0.],
            [0., 0., 0.],[0., 3., 0.],[0., 0., 4.]]
    # 查找data中出现[0, 0, 0]的index
    data = np.asarray(data)
    vector=[0, 0, 0]
    index =matching_data_vecror(data,vector)
    print(index)
    >>[False  True False  True False False]
    # 实现去除data数组中元素为[0, 0, 0]的行向量
    pair_issame_1 = data[~index, :]  # 筛选数组
    :param data:
    :param vector:
    :return:
    '''
    # index = (data[:, 0] == 0) & (data[:, 1] == 0) & (data[:, 2] == 0)
    row_nums = len(data)
    clo_nums = len(vector)
    index = np.asarray([True] * row_nums)
    for i in range(clo_nums):
        index = index & (data[:, i] == vector[i])
    return index


def set_mat_vecror(data, index, vector):
    '''
    实现将data指定index位置的数据设置为vector
    # 实现将大于阈值分数的point，设置为vector = [10, 10]
    point = [[0., 0.], [1., 1.], [2., 2.],
             [3., 3.], [4., 4.], [5., 5.]]
    point = np.asarray(point) # 每个数据点
    score = np.array([0.7, 0.2, 0.3, 0.4, 0.5, 0.6])# 每个数据点的分数
    score_th=0.5
    index = np.where(score > score_th) # 获得大于阈值分数的所有下标
    vector = [10, 10]                  # 将大于阈值的数据设置为vector
    out = set_mat_vecror(point, index, vector)
    :param data:
    :param index:
    :param vector:
    :return:
    '''
    data[index, :] = vector
    return data


def get_batch(image_list, batch_size):
    '''
    batch size data
    :param image_list:
    :param batch_size:
    :return:
    '''
    sample_num = len(image_list)
    batch_num = math.ceil(sample_num / batch_size)

    for i in range(batch_num):
        start = i * batch_size
        end = min((i + 1) * batch_size, sample_num)
        batch_image = image_list[start:end]
        print("batch_image:{}".format(batch_image))


def gen_range(shape=None,start=None, *args, **kwargs):
    '''create range data'''
    data = np.arange(start, *args, **kwargs)
    if shape:
        data = np.reshape(data, newshape=shape)
    return data


def mat2d_data(data, indexes):
    '''
    get element by indexes
    data=numpy_tools.gen_range((3,3),0,9)
    data=np.mat(data)
    indexes=np.asarray([[1,1],[2,1],[2,2]])
    out=mat2d_data(data, indexes)
    print(data)
    print(out)
    :param data:
    :param indexes:
    :return:
    '''
    out=data[indexes[:,0],indexes[:,1]]
    return out