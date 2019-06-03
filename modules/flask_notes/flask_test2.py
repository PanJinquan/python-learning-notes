# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : flask_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-31 11:40:04
"""

import os
import sys
import logging as logger
from flask import Flask, request, redirect, url_for, render_template, jsonify
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
app = Flask(__name__)

host = '/http://vipkid.api.xmc.dm-ai.cn'
api = '/api/v1/detect/facepp'
url=host+api
image_path='../dataset/test_image/1.jpg'

# 从flask 这个框架中导入到Flask这个类
from flask import Flask

# 初始化一个Flask 对象
# Flask()
# 需要传递一个参数__name__
# 1. 方便flask框架去找寻资源
# 2. 方便flask插件比如Flask-Sqlalchemy 出现错误的时候，好去找寻问题所在的位置
app = Flask(__name__)

# @app.route 是一个装饰器
# @开头，并且在函数的上面，说明是装饰器
# 这个装饰器的作用， 是做一个url与视图函数的映射
# 127.0.0.1:5000/ -> 去请求hello_world这个函数，然后将结果返回给浏览器
@app.route('/')
def hello_world():
    return 'Hello World!'

# 如果当前这个文件作为入口程序运行，那么就会执行app.run()
if __name__ == '__main__':
    # app.run()
    # 启动一个应用服务器， 来接受用户的请求
    # while True:
    #   listen()
    app.run()
