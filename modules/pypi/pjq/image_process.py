# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : fun.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-12 19:56:01
"""
import cv2
import ddm


def image_process(image):
    '''
    图像处理
    :return:
    '''
    dst = cv2.GaussianBlur(image, (5, 5), 15)
    return dst
