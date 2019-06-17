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


def read_image(image_path):
    '''
    读取图像
    :return:
    '''
    image = cv2.imread(image_path)
    return image
