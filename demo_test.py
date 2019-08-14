# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : demo_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 13:57:29
"""
import time
from utils import debug


import cv2
import numpy as np

# 这里说一下旋转的opencv中为旋转提供的三个要素
# 旋转的中心点（center）
# 旋转角度()
# 旋转后进行放缩
# 我们可以通过cv2.getRotationMatrix2D函数得到转换矩阵

img = cv2.imread('/media/dm/dm2/project/python-learning-notes/dataset/dataset2/A/1.jpg')
rows,cols,_ = img.shape

matrix = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
# 得到变换的矩阵，通过这个矩阵再利用warpAffine来进行变换
# 第一个参数就是旋转中心，元组的形式，这里设置成相片中心
# 第二个参数90，是旋转的角度
# 第三个参数1，表示放缩的系数，1表示保持原图大小

img1 = cv2.warpAffine(img,matrix,(cols,200))

cv2.imshow('img',img)
cv2.imshow('img1',img1)
cv2.waitKey(0)

