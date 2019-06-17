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
if __name__=="__main__":
    image_path="1"+os.getcwd()
    print(image_path)
    path1=os.path.join(image_path,image_path)
    print(path1)