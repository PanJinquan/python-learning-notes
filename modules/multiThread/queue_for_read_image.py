# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : queue_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 18:02:16
    @url:     https://www.cnblogs.com/xiangsikai/p/8185031.html
"""
import queue
from modules.multiThread.thread_operate import customThread
import os
import cv2
# 定义producer和consumer线程完成的标识
PRODUCER_EXIT_FLAG=False
CONSUMER_EXIT_FLAG=False
queue_container = queue.Queue(maxsize=5)

def producer(image_list, queue_container):
    for i,image_path in enumerate(image_list):
        queue_container.put(image_path)
        print("setp:{},将图片添加到队列中:{}".format(i,image_path))
        # time.sleep(3)

def consumer(batch_size, queue_container):
    '''
    队列消费线程consumer
    :param batch_size:一次从队列读取数据个数
    :param queue_container:队列
    :return:
    '''
    timeout_Flag=False
    dest_image=[]
    dest_image_list=[]
    while True:
        image_list=[]
        for i in range(batch_size):
            if len(image_list)>0 and queue_container.empty() :
                # 当队列为空时，并且image_list不为零，说明剩余的数据不足拼接成一个完整的batch,
                # 为避免堵塞后续处理，这时应该跳过堵塞
                continue
            try:
                image_path = queue_container.get(timeout=1)
                image_list.append(image_path)
            except Exception as e:
                print("get timeout...")
                timeout_Flag=True
                break

        if len(image_list)>0:
            qsize = queue_container.qsize()
            print("取出一个batch的数据：{},剩余：{}".format(image_list, qsize))
            image_batch,image_list_batch=read_image_batch(image_list)
            if len(image_batch)>0:
                dest_image+=image_batch
                dest_image_list+=image_list_batch

        # 如果获取队列超时(队列为空)并且生产线程已经完成，说明数据已经处理完毕了
        if timeout_Flag and PRODUCER_EXIT_FLAG:
            break
        # time.sleep(3)
    # 告知这个任务执行完了
    queue_container.task_done()  # 用于通知queue_container.join()可以继续干其他事啦
    return  dest_image,dest_image_list


def read_image_batch(image_list):
    image_batch=[]
    out_image_list=[]
    for image_path in image_list:
        image=cv2.imread(image_path)
        if image is None:
            print("no image:{}".format(image_path))
            continue
        image_batch.append(image)
        out_image_list.append(image_path)
    return image_batch,out_image_list


if __name__=="__main__":
    # 请求输入队列：最多存入10个
    queue_container = queue.Queue(maxsize=5)
    image_id_list=["000001.jpg","000002.jpg","000003.jpg","000004.jpg","000005.jpg","000006.jpg","000007.jpg","000008.jpg","000009.jpg","000010.jpg"]
    # image_id_list=["000001.jpg","000002.jpg","000003.jpg"]
    image_dir="../../dataset/VOC/JPEGImages"

    image_list=[os.path.join(image_dir,id) for id in image_id_list]
    batch_size=3
    # 创建队列产生线程producer
    producer_thread = customThread(thread_id="producer_thread",func=producer, args=(image_list,queue_container))
    # 创建队列消费线程consumer
    consumer_thread = customThread(thread_id="consumer_thread",func=consumer, args=(batch_size, queue_container))
    # consumer_thread2 = threading.Thread(target=consumer, args=(batch_size, queue_container))

    # 执行线程
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    PRODUCER_EXIT_FLAG=True
    print("producer_thread线程:完成")
    consumer_thread.join()

    CONSUMER_EXIT_FLAG=True
    dest_image,dest_image_list = consumer_thread.get_result()
    print(dest_image_list)
    cv2.imshow("image", dest_image[0])
    cv2.waitKey(1000)
    print("consumer_thread线程:完成")
    # consumer_thread2.start()
