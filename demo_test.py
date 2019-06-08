# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : demo_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 13:57:29
"""
import cv2

image_path="https://farm3.staticflickr.com/2099/1791684639_044827f860_o.jpg"
image=cv2.imread(image_path)

cv2.imshow("ii",image)
cv2.waitKey(0)
