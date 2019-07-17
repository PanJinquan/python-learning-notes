# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : list_json.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-09 17:25:46
"""
import json


def read_json_data(json_path):
    # 读取数据
    with open(json_path, 'r') as f:
        json_data = json.load(f)

def write_json_path(out_json_path,json_data):
    # 写入 JSON 数据
    with open(out_json_path, 'w') as f:
        json.dump(json_data, f)


if __name__=="__main__":
    # json_path = "list_result.json"
    out_json_path = "list_result.json"

    # data=[1,2,3,4]
    data=[]
    data.append([1,2,3,4])
    data.append([4,5,6,7])
    dict_data={}
    dict_data["data"]=data
    print(dict_data)
    # jsonArr = json.dumps(dict_data, ensure_ascii=False)
    # print(jsonArr)
    write_json_path(out_json_path,json_data=dict_data)