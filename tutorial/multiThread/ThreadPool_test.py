# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : threadpool_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-30 17:26:31
"""


from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import time


def task(n):
    print(n)
    time.sleep(2)


if __name__=="__main__":
    nums_thread=5
    pool = ThreadPool(processes=nums_thread)
    for i in range(500):
        # pool.apply_async(func=task,args=(i,)) # 异步
        pool.apply(func=task,args=(i,))     #同步
    print('test')
    pool.close()
    pool.join()