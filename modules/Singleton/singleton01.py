# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : singleton.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-10 10:06:18
    @url    : https://www.cnblogs.com/huchong/p/8244279.html
"""
import threading


class Singleton(object):
    '''
    这是线程安全的单实例方法，为了保证线程安全需要在内部加入锁
    '''
    _instance_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if not hasattr(Singleton, "_instance"):
            with Singleton._instance_lock:
                if not hasattr(Singleton, "_instance"):
                    print("__new__2")
                    Singleton._instance = object.__new__(cls)
        return Singleton._instance

    def __init__(self, name):
        print("__init__")
        self.__name = name

    def set_name(self, name):
        self.__name = name

    def get_name(self):
        return self.__name


if __name__ == "__main__":
    s1 = Singleton("A")
    s2 = Singleton("B")
    s3 = Singleton("C")

    print(s1,s1.get_name())
    print(s2,s2.get_name())
    print(s3,s3.get_name())

