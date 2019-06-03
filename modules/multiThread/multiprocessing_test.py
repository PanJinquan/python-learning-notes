# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : multiprocessing_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-30 17:59:14
"""
#coding=utf-8
import multiprocessing
import time
from multiprocessing import Pool

def do(n) :
  #获取当前线程的名字
  name = multiprocessing.current_process().name
  print(name,'starting')
  print("worker ", n)
  return



if __name__ == '__main__' :
    '''
    url:https://blog.csdn.net/seetheworld518/article/details/49639651
    '''
    num_pro=5 # nums of multiprocessing
    pro_list = []
    for i in range(1,num_pro):
        p = multiprocessing.Process(target=do, args=(i,))
        pro_list.append(p)
        p.start()
    # join()方法表示等待子进程结束以后再继续往下运行，通常用于进程间的同步。
    for p in pro_list:
        p.join()
    print("Process end.")


#
# def test(p):
#     print(p)
#     time.sleep(3)
#
#
# if __name__=="__main__":
#     pool = Pool(processes=10)
#     for i  in range(500):
#         ''' https://blog.csdn.net/brucewong0516/article/details/85788202
#          （1）循环遍历，将500个子进程添加到进程池（相对父进程会阻塞'
#          （2）每次执行2个子进程，等一个子进程执行完后，立马启动新的子进程。（相对父进程不阻塞'
#         '''
#         pool.apply_async(test, args=(i,))   #维持执行的进程总数为10，当一个进程执行完后启动一个新进程.
#     print('test')
#     pool.close()
#     pool.join()

#
# if __name__ == "__main__":
#     pool = Pool(processes=10)
#     for i in range(500):
#         '''
#          （1）遍历500个可迭代对象，往进程池放一个子进程'
#          （2）执行这个子进程，等子进程执行完毕，再往进程池放一个子进程，再执行。（同时只执行一个子进程）
#           for循环执行完毕，再执行print函数'
#         '''
#         pool.apply(test, args=(i,))  # 维持执行的进程总数为10，当一个进程执行完后启动一个新进程.
#     print('test')
#     pool.close()
#     pool.join()

