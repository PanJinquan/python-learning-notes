# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : singleton.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-10 10:06:18
"""

import time
import threading


class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        print("__init__")
        self.__name = None

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"):
                    Singleton._instance = Singleton(*args, **kwargs)
        return Singleton._instance

    def set_name(self, name):
        self.__name = name

    def get_name(self):
        return self.__name


def task(arg):
    obj = Singleton.instance()
    print(obj)


if __name__ == "__main__":
    s1 = Singleton.instance()
    s2 = Singleton.instance()
    s3 = Singleton.instance()
    s3.set_name("D")

    print(s1, s1.get_name())
    print(s2, s2.get_name())
    print(s3, s3.get_name())
