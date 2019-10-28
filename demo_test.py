# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : demo_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 13:57:29
"""
import time
from utils import debug, numpy_tools
import socket
import os
from utils import plot_utils

# 主函数
import numpy as np


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


if __name__ == "__main__":
    data = numpy_tools.gen_range((3, 3), 0, 9)
    data = np.mat(data)
    indexes = np.asarray([[1, 1], [2, 1], [2, 2]])
    out = mat2d_data(data, indexes)
    print(data)
    print(out)
    dd=np.asarray([0.1, 2.1])
    print(dd.astype(np.int32))
