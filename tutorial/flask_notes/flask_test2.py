# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : flask_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-31 11:40:04
"""
from flask import Flask
import json
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':

    data='"request_id": "value"'
    result_json=json.dump(data)

    # app.run(host='0.0.0.0', debug=True, port=8000)