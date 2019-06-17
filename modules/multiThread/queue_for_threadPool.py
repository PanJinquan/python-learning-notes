# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : queue_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 18:02:16
    @url:     https://www.cnblogs.com/xiangsikai/p/8185031.html
"""
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor

# 队列最多存入10个
queue_container = queue.Queue(maxsize=10)
# 最多产生的数据量
MAX_NUMS = 20
# 创建线程锁
threadLock = threading.Lock()


def producer(input_queue, max_nums):
    '''
    :param input_queue:
    :param max_nums:
    :return:
    '''
    count = 0
    while True:
        count += 1
        # 　下载图片
        image_path = str(count) + ".jpg"
        input_queue.put(image_path)
        print("下载图片：{},放入队列中".format(image_path))
        # time.sleep()
        if count == max_nums:
            break
    print("下载完成....................")


def consumer(image_path, thread_id):
    '''
    :param image_path:
    :param id:
    :return:
    '''
    # threadLock.acquire()  # 加锁
    print("id:{},从队列取出图片，并处理图片：{}".format(thread_id, image_path))
    time.sleep(1)
    # 用于通知queue_container.join()可以继续干其他事啦
    queue_container.task_done()
    # threadLock.release()  # 释放锁


def consumer_thread(max_workers):
    '''
    :return:
    '''
    executor = ThreadPoolExecutor(max_workers)
    count = 0
    task_list = []
    timeout = None
    while True:
        try:
            image_path = queue_container.get(timeout=timeout)
            count += 1
            task = executor.submit(consumer, image_path=image_path, thread_id=count)
            task_list.append(task)
        except Exception as e:
            print("get timeout...:{}".format(e))
        for task in task_list:
            if task.done():
                # result=task.result()
                task_list.remove(task)
        if task_list:
            timeout = 0.5
        else:
            timeout = None

        print("剩余任务:{}".format(len(task_list)))
        # queue_container.task_done()


if __name__ == "__main__":
    p = threading.Thread(target=producer, args=(queue_container, MAX_NUMS))
    # 执行线程
    p.start()
    consumer_thread(max_workers=2)
