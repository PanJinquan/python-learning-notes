# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : demo_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-07 13:57:29
"""
import cv2
import pjq.image_process as image_process

if __name__=="__main__":
    image_path="./dataset/test_image/1.jpg"
    image = image_process.read_image(image_path)
    cv2.imshow("image",image)
    cv2.waitKey(0)