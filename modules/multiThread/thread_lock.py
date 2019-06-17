# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : thread_lock.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-10 15:51:49
"""
import time
import threading
from modules.multiThread.thread_operate import customThread

# 创建线程锁
threadLock = threading.Lock()
num = 0


def myThread():
    threadLock.acquire()
    global num
    num = num + 1
    time.sleep(1)
    num = num - 1
    threadLock.release()
    return num


if __name__ == "__main__":
    thread_nums = 5
    thread_pool = []
    for i in range(thread_nums):
        p = customThread(thread_id=str(i), func=myThread, args=())
        # 执行线程
        p.start()
        thread_pool.append(p)
    while True:
        for p in thread_pool:
            p.join()
            print(p.get_result())
