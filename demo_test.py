# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : demo_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 13:57:29
"""


# 主函数


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


def bad_append(new_item, a_list=[]):
    a_list.append(new_item)
    return a_list


def __set_stages_lr(epoch, lr_stages, lr_list):
    '''
    :param epoch:
    :param lr_stages: [    35, 65, 95, 150]
    :param lr_list:   [0.1, 0.01, 0.001, 0.0001, 0.00001]
    :return:
    '''
    if epoch in lr_stages:
        index = lr_stages.index(epoch) + 1
        lr = lr_list[index]


def __get_lr(epoch, lr_stages, lr_list):
    lr = None
    max_stages = max(lr_stages)
    for index in range(len(lr_stages)):
        if epoch <= lr_stages[index]:
            lr = lr_list[index]
            break
        if epoch > max_stages:
            lr = lr_list[index + 1]
    return lr


import numpy as np


def get_drop_last(num_sample, batch_size):
    r = num_sample % batch_size
    if r:
        num_sample = num_sample + batch_size - r
    return num_sample


if __name__ == "__main__":
    print(get_drop_last(num_sample=4, batch_size=4))
