# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : env_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-16 12:10:26
"""
import sys
import os

# 打印当前python搜索模块的路径集
print(sys.path)
# 打印当前文件所在路径
print("os.path.dirname(__file__):", os.path.dirname(__file__))
print("os.getcwd():              ", os.getcwd())  # get current work directory：cwd:获得当前工作目录

'''添加相关的路径
sys.path.append(‘你的模块的名称’)。
sys.path.insert(0,’模块的名称’)
'''
# 先添加image_processing所在目录路径
sys.path.append("F:/project/python-learning-notes/utils")
# sys.path.append(os.getcwd())
# 再倒入该包名
import image_processing

#
os.environ["PATH"] += os.pathsep + 'D:/ProgramData/Anaconda3/envs/pytorch-py36/Library/bin/graphviz/'

image_path = "F:/project/python-learning-notes/dataset/test_image/1.jpg"
image = image_processing.read_image(image_path)
image_processing.cv_show_image("image", image)
