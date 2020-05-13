# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : thread_operate.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 11:04:59
"""


from utils import file_processing,image_processing
import threading
import time

threadLock = threading.Lock()#创建线程锁

class cvThread(threading.Thread):
    '''
    CV多线程
    '''

    def __init__(self, thread_id, func, args=()):
        '''
        :param thread_id:
        :param func:
        :param args:
        '''
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        '''
        重载Thread的方法run
        :return:
        '''
        print("Starting thread_id:{} ".format(self.thread_id))
        # 获得锁，成功获得锁定后返回True， 可选的timeout参数不填时将一直阻塞直到获得锁定， 否则超时后将返回False
        # threadLock.acquire() #线程加锁
        self.result = self.func(*self.args)
        # threadLock.release()# 释放锁

    def get_result(self):
        '''
        获得线程返回结果
        :return:
        '''
        try:
            return self.result
        except Exception:
            return None




def fun_test(images_list):
    time.sleep(2)
    print(images_list)
    images=[]
    for filename in images_list:
        image = image_processing.read_image(filename, resize_height=224, resize_width=224, normalization=False)
        images.append(image)
    return images

def split_data_list(data_list, split_nums):
    '''
    :param data_list: 数据列表
    :param split_nums: 将列表分成多少块，注意split_nums块必须小于data_list的长度
    :return: 返回data_list分块后的索引
    '''
    data_size=len(data_list)
    if split_nums>data_size:
        print("illegal arguments,split_nums must be less than len(data_size)")
        exit(0)
    batch_index=[]
    for i in range(split_nums):
        start = int(data_size / split_nums * i)
        end = int(data_size / split_nums * (i + 1))
        if (i == split_nums - 1) :
            end = data_size
        batch_index.append((start,end))
    return batch_index

def thread_test(images_list, nums_thread=4):
    thread_collection = []#创建线程容器
    # 创建新线程
    batch_index=split_data_list(images_list, split_nums=nums_thread)
    print("batch_index:{}".format(batch_index))
    for i in range(nums_thread):
        start,end=batch_index[i]
        batch_image_list=images_list[start:end]
        thread = customThread(thread_id=i, func=fun_test, args=(batch_image_list,))
        thread.start()    # 开启新线程
        thread_collection.append(thread)# 添加线程到线程列表

    # 等待所有线程完成
    for thread in thread_collection:
        thread.join()
        batch_image=thread.get_result()
        image_processing.show_image(title="image_dict",image=batch_image[0])

    print("Exiting Main Thread")


if __name__=='__main__':
    image_dir="../../dataset/test_image"
    images_list = file_processing.get_images_list(image_dir, postfix=['*.png', '*.JPG'])
    print(images_list)
    thread_test(images_list, nums_thread=4)