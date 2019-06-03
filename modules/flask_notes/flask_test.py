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
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
app = Flask(__name__)


url='https://mockapi.eolinker.com/NCRBltj06bce8e2a6de6dffcf8d635d8b9a3cbfaf8cb457/v1/model/version'
def projects():
    r = requests.get(url)
    print(r.text)

if __name__=="__main__":
    # app.run()
    projects()