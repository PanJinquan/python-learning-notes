#! /usr/bin/env python
# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : urls.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-03 15:31:55
    @url    : https://www.cnblogs.com/haifeima/p/9915820.html
"""

import flask
import json
import mysql_data

server=flask.Flask(__name__)

@server.route('/',methods=['post'])
def login():
    #登录需要两个参数，name和pwd
    uname=flask.request.values.get('username')# 传参，前面的是变量，括号里面是key
    passwd=flask.request.values.get('password')
    #args只能获取到跟在url后面的参数，所以我们改用values

    if uname and passwd:# 非空为真
        # 需要先写一个导入数据库的函数，例如我写了一个名称为tools的函数（如图），放在另一个python文件中，import tools进行调用。当然也可以直接写在本python文件中，但是显得会累赘。
        sql="SELECT * FROM app_myuser WHERE username='%s' AND passwd='%s';"%(uname,passwd)
        print("sql:{}".format(sql))
        result = mysql_data.my_db(sql)#执行sql
        if result:
            res={"error_code":1000,"mag":"登录成功"}# 接口返回的都是json，所以要这样写。先导入json模块，import json。
        else:
            res = {"error_code": 3001, "mag": "账号或密码错误！"}
    else:
        res={"error_code":3000,"mag":"必填参数未填，请查看接口文档！"}

    return  json.dumps(res,ensure_ascii=False)#防止出现乱码；json.dumps()函数是将字典转化为字符串

server.run(port=8888)