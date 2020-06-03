# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : custom_queue.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-14 08:40:16
"""

import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor

threadLock = threading.Lock()
# threadLock.acquire()  # 加锁
# threadLock.release()  # 释放锁

class CustomQueue():
    '''
    自定义队列
    '''

    def __init__(self, maxsize):
        '''
        初始化
        :param maxsize:
        '''
        self.receive_queue = queue.Queue(maxsize=maxsize)
        self.send_queue = queue.Queue(maxsize=maxsize)
        self.count = 0
        self.task_nums = 0

    def receiver(self, data, func):
        '''
        接收器，接收接收队列数据，并对数据进行处理
        :param data: 接收的数据
        :param func: receiver接收数据处理回调函数
        :return:
        '''
        self.count += 1
        result = func(data)
        self.receive_queue.put(result)
        print("接收到第{}个数据，并存入receive_queue队列中data={},qsize={}".format(self.count, result, self.receive_queue.qsize()))

    @staticmethod
    def receiver_process(data):
        '''
        接收数据处理
        :param data:
        :return:
        '''
        # TODO 这里进行接收数据处理
        return data

    def start_consumer(self, max_workers, func):
        '''
        开启接收线程
        :param max_workers:
        :param func 消费数据处理回调函数
        :return:
        '''
        receive_task = threading.Thread(target=self.consumer_thread_pool, args=(max_workers, func))
        receive_task.start()

    def consumer_thread_pool(self, max_workers, func):
        '''
        消费线程池
        :param max_workers
        :param func 消费数据处理回调函数
        :return:
        '''
        executor = ThreadPoolExecutor(max_workers)
        task_list = []
        timeout = None
        while True:
            try:
                data = self.receive_queue.get(timeout=timeout)
                task = executor.submit(func, data)
                task_list.append(task)
            except Exception as e:
                print("get timeout...:{}".format(e))
            for task in task_list:
                if task.done():
                    # 将返回数据存入发送消息队列
                    print("将处理数据存入send_queue队列:qsize={}".format(self.send_queue.qsize()))
                    self.send_queue.put(task.result())
                    task_list.remove(task)
            if task_list:
                timeout = 0.5
            else:
                # 如果线程任务已经全部做完，则receive_queue阻塞
                timeout = None
            self.task_nums = len(task_list)
            print("剩余任务:{}".format(self.task_nums))
            # queue_container.task_done()

    @staticmethod
    def consumer_process(data):
        '''
        消费数据处理函数,
        注意，如果这里有线程共享的变量，需要加锁保证安全
        :param data:由线程池传入的数据
        :return:
        '''
        # TODO 这里进行消费数据处理
        # threadLock.acquire()  # 加锁
        print("从队列取出数据，并处理：{}".format(data))
        # threadLock.release()  # 释放锁
        return data

    def start_sender(self, func):
        '''
        开启发送线程，并发送数据进行处理
        :param func 发送数据处理回调函数
        :return:
        '''
        send_task = threading.Thread(target=self.sender, args=(func,))
        send_task.start()

    def sender(self, func):
        '''
        发送数据进行处理
        :param func 发送数据处理回调函数
        :return:None
        '''
        count = 1
        while True:
            # 取出队列数据
            print("等待发送第{}个信息".format(count))
            data = self.send_queue.get()
            result = func(data)
            print("发送数据是result:{}".format(result))
            print("已经发送第{}个信息,剩余:{}".format(count, self.send_queue.qsize()))
            count += 1

    @staticmethod
    def sender_process(data):
        '''
        发送数据处理函数
        :param data:
        :return:
        '''
        # TODO 这里进行发送数据处理
        return data

    def get_task_nums(self):
        '''
        获得当前任务数量状态
        :return:
        '''
        return self.task_nums


def receiver_process(data):
    '''
    接收数据处理
    :param data:
    :return:
    '''
    # TODO 这里进行接收数据处理
    data = data + " receiver_process"
    time.sleep(0.5)
    return data


def consumer_process(data):
    '''
    消费数据处理函数,
    注意，如果这里有线程共享的变量，需要加锁保证安全
    :param data:由线程池传入的数据
    :return:
    '''
    # TODO 这里进行消费数据处理
    # threadLock.acquire()  # 加锁
    print("从队列取出数据，并处理：{}".format(data))
    data = data + " consumer_process"
    time.sleep(2)
    # threadLock.release()  # 释放锁
    return data


def sender_process(data):
    '''
    发送数据处理函数
    :param data:
    :return:
    '''
    # TODO 这里进行发送数据处理
    # time.sleep(1)
    data = data + " sender_process"
    return data


def create_data():
    '''
    产生数据
    :return:
    '''
    maxsize = 5
    q = CustomQueue(maxsize=maxsize)
    q.start_sender(func=sender_process)
    q.start_consumer(max_workers=2, func=consumer_process)
    count = 0
    max_nums = 20
    while True:
        count += 1
        image_path = str(count) + ".jpg"
        q.receiver(image_path, func=receiver_process)
        if count == max_nums:
            break
    print("数据产生完毕....................")


if __name__ == "__main__":
    p = threading.Thread(target=create_data)
    # 执行线程
    p.start()
    print("__main__主线程")
