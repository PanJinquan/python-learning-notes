# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : json_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-02 10:48:47
"""

'''
url : https://www.runoob.com/python3/python3-json.html

'''

import json

json_path="result.json"
out_json_path="out_result.json"

# 读取数据
with open(json_path, 'r') as f:
    json_data = json.load(f)

# 写入 JSON 数据
with open(out_json_path, 'w') as f:
    json.dump(json_data, f)

print("原始数据：", json_data)

# 将 JSON 对象转换为 Python 字典
# data2 = json.loads(json_data)
# print("data2['name']: ", json_data['name'])
# print("data2['url']: ", json_data['url'])


