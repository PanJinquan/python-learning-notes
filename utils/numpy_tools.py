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


def find_max_shape_data(list_data):
    max_shape_data = np.asarray([])
    for data in list_data:
        if len(max_shape_data) < len(data):
            max_shape_data = data
    return max_shape_data


def data_alignment(data):
    '''
    row_stack()函数扩展行，column_stack()函数扩展列
    :param list_data:
    :param align:
    :param extend:
    :return:
    '''
    max_shape_data = find_max_shape_data(data)
    for i in range(len(data)):
        maxdata = np.zeros(shape=max_shape_data.shape, dtype=max_shape_data.dtype) - 1
        shape = data[i].shape
        if len(shape) == 1:
            maxdata[0:shape[0]] = data[i]
        else:
            maxdata[0:shape[0], 0:shape[1]] = data[i]
        data[i] = maxdata
    # data = np.asarray(data)
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


def gen_range(shape=None, start=None, *args, **kwargs):
    '''
    create range data->
    gen_range(shape=(10, 10), start=0, stop=100)
    :param shape:
    :param start:
    :param args:
    :param kwargs:
    :return:
    '''
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
    out = data[indexes[:, 0], indexes[:, 1]]
    return out


def count_sort_list(list_data: list, reverse=True):
    '''
    给定一个非空正整数的数组，按照数组内数字重复出现次数，从高到低排序
    :param list_data:
    :param reverse: True-降序,Fasle-升序
    :return:
    '''
    d = {}
    list_sorted = []
    for i in list_data:
        d[i] = list_data.count(i)
    # 根据字典值的降序排序
    d_sorted = sorted(d.items(), key=lambda x: x[1], reverse=reverse)
    # 输出排序后的数组
    for x in d_sorted:
        for number in range(0, x[1]):
            list_sorted.append(x[0])
    return list_sorted


def remove_list_data(list_data, flag=["", -1]):
    '''
    删除list所有符合条件元素
    :param list_data:
    :param flag:
    :return:
    '''
    for f in flag:
        while f in list_data:
            list_data.remove(f)
    return list_data


def label_alignment(data_list):
    mat = np.asarray(data_list).T
    label_list = []
    for data in mat:
        out = count_sort_list(data.tolist(), reverse=True)
        out = remove_list_data(out, flag=["", -1])
        if out:
            label = out[0]
        else:
            label = -1
        label_list.append(label)
    return label_list


def __print(data, info=""):
    print("-------------------------------------")
    print(info)
    for index in range(len(data)):
        print("{}".format(data[index]))


def rmse(data1, data2):
    '''
    均方差
    :param predictions:
    :param targets:
    :return:
    '''
    return np.sqrt(((data1 - data2) ** 2).mean())


def l2(data1, data2):
    '''
    L2欧式距离
    :param emb1:
    :param emb2:
    :return:返回欧式距离(0,+∞),值越小越相似
    '''
    diff = data1 - data2
    dist = np.sum(np.power(diff, 2), axis=1)
    return dist


def L1_loss(y_true, y_pre):
    return np.sum(np.abs(y_true - y_pre))

def L2_loss(y_true, y_pre):
    return np.sum(np.square(y_true - y_pre))

def norn(x, ord):
    y = np.linalg.norm(x, ord=ord, axis=1, keepdims=True)
    return y


def mean(data):
    return np.mean(data)


def var(data):
    # 求方差
    return np.var(data)


def std(data):
    # 求标准差
    return np.std(data, ddof=1)


def load_data(data_path):
    return np.load(data_path)


# 矩阵拼接


if __name__ == "__main__":
    data1 = np.arange(0, 10)
    data1 = data1.reshape([5, 2])
    y = np.concatenate([data1, data1], 0)
    print(y)
