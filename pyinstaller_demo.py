# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : pyinstaller_demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-16 08:28:21
"""
from utils import image_processing





if __name__=="__main__":
    '''
    cmd:pyinstaller -F -w  pyinstaller_demo.py
    '''
    image_path="E:/git/tools/dataset/VOC/JPEGImages/000002.jpg"
    image=image_processing.read_image(image_path)
    image_processing.cv_show_image("image",image)