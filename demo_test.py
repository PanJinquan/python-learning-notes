# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import numpy as np

angle = 150
# angle = np.where(angle > 90, 180 - angle, angle)
print(angle)

iou_mat = np.zeros(shape=(10, 20))
max_index = np.argmax(iou_mat, axis=0)
print("iou_mat.shape:{}".format(iou_mat.shape))
print("max_index.shape:{}".format(max_index.shape))