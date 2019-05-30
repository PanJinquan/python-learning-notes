# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : wrapper_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-30 09:10:47
"""


def decorator(func):
	def wrapper(*args, **kwargs):  # 包裹器包裹住新的业务逻辑
		func(*args, **kwargs)
		print("done...")
	return wrapper  # 返回包裹器

@decorator
def print_name(value, *args, **kwargs):
    print(value)
    print(args)
    print(kwargs)

if __name__=="__main__":
    print_name('name')  # 输出 Find Name: David Coco
    print("-------------------------")
    print_name(10,11,12,name="pjq",age=21)  # 输出 Find Name: David Coco


