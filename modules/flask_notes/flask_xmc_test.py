# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : flask_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-31 11:40:04
"""

import flask
import json
import datetime

app=flask.Flask(__name__)

'''
{
	"request_id":"path/to/image.jpg",
	"source_id":"path/to/image_id.jpg",
	"result":"result"
}
'''
@app.route('/',methods=['post'])
def login():
    params=flask.request.json #入参是字典json时用它,下面的代码要判断传入的参数是否是json类型
    # 传参，前面的是变量，括号里面是key
    if params:
        request_id = params.get('request_id')
        source_id  = params.get('source_id', 'default source_id')  # 如果没有传。默认值是男
        result     = params.get('result')
        print("request_id:{}".format(request_id))
        print("source_id:{}".format(source_id))
        print("result:{}".format(result))
    return "hi Alian"


def start_flask_server(host='0.0.0.0', port=9000, debug=True):
    print('starting flask server on {}:{}'.format(host, port))
    app.run(host=host, port=port, debug=debug, processes=1)

if __name__=="__main__":
    app.run(port=8888)