# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : thread_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 08:55:50
"""

import threading
def task1(a,b):
    return (a+b)

def task2(a,b):
    return (a+b)
if __name__=="__main__":
    task1_thread = threading.Thread(target=task1,args=(1,2))
    task2_thread = threading.Thread(target=task2, args=(2,2))

    # 开启线程执行
    task1_thread.start()
    task2_thread.start()

    # 等待线程执行完毕
    task1_thread.join()
    task2_thread.join()

    task1_thread.get