# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : person.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-17 13:47:39
"""


class Person():
    classes = None

    def __init__(self, name):
        self.name = name

    def print_person(self):
        print("name:{}".format(self.name))
        print("age :{}".format(self.age))
        print("classes :{}".format(self.classes))
