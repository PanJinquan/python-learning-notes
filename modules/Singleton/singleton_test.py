# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : singleton.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-10 10:06:18
"""

from modules.Singleton.singleton02 import MySingleton

if __name__ == "__main__":
    s1 = MySingleton("A")
    s2 = MySingleton("B")
    s3 = MySingleton("C")
    s3.set_name("D")

    print(s1,s1.get_name())
    print(s2,s2.get_name())
    print(s3,s3.get_name())
