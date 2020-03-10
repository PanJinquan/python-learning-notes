# -*- coding:utf-8 -*-
from flask import  jsonify
# error code reference: http://wiki.dm-ai.cn/pages/viewpage.action?pageId=30761576
ERROR_MSG = {
    0: {'err_code': 0, 'err_msg': 'OK'},
    1: {'err_code': 1, 'err_msg': 'xxxx'},
    2: {'err_code': 2, 'err_msg': 'xxxx'},
    3: {'err_code': 3, 'err_msg': 'xxxx'},

}

def get_err_msg_dict(err_code):
    assert err_code in ERROR_MSG.keys()
    return ERROR_MSG.get(err_code)

def util_make_response(msg, status_code=200):
    assert (isinstance(msg, dict))
    resp = jsonify(msg)
    resp.status_code = status_code
    return resp
