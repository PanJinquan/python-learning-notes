# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : main.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-08-16 12:34:55
"""
from utils import json_utils



if __name__=="__main__":
    conf_name="/media/dm/dm1/git/python-learning-notes/modules/yam_config/config.yaml"
    conf=json_utils.load_config(conf_name)
    print(conf)