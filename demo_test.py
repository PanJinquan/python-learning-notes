# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : demo_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 13:57:29
"""
import cv2
import os
import torch
from torch.autograd import Variable
import numpy as np
from utils import debug

from memory_profiler import profile

data = np.asarray([[6, 4, 10],
                  [3, 2, 9],
                  [0, 1, 5]])
data=data[np.lexsort(data.T)]
print(data)
