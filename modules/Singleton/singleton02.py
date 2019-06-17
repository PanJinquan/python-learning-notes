# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : singleton.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-10 10:06:18
"""


def singleton(cls):
    instances = {}
    def _singleton(*args,**kwargs):
        if cls not in instances:
            instances[cls] = cls(*args,**kwargs)
        return instances[cls]
    return _singleton


@singleton
class MySingleton(object):
    '''
    使用装饰器的方法，__init__仅会运行一次：

    '''
    def __init__(self, name):
        print("__init__")
        self.__name = name

    def set_name(self, name):
        self.__name = name

    def get_name(self):
        return self.__name


if __name__ == "__main__":
    s1 = MySingleton("A")
    s2 = MySingleton("B")
    s3 = MySingleton("C")
    s3.set_name("D")

    print(s1,s1.get_name())
    print(s2,s2.get_name())
    print(s3,s3.get_name())
