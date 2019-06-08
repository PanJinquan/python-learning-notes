# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : queue_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 18:02:16
    @url:     https://www.cnblogs.com/xiangsikai/p/8185031.html
"""
import threading,time
import queue

# 最多存入10个
q = queue.Queue(maxsize=10)

def producer(name):
    count = 1
    while True:
        #　生产一块骨头
        q.put("骨头 %s" % count )
        print("生产了骨头",count)
        count +=1
        time.sleep(0.3)

def consumer(name):
    while True:
        print("%s 取到[%s] 并且吃了它" %(name, q.get()))
        time.sleep(1)
        # 告知这个任务执行完了
        q.task_done()

# 生成线程
p = threading.Thread(target=producer,args=("德国骨科",))
c = threading.Thread(target=consumer,args=("陈狗二",))
# d = threading.Thread(target=consumer,args=("吕特黑",))

# 执行线程
p.start()
c.start()
# d.start()