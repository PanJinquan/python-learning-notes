# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : queue_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 18:02:16
    @url:     https://www.cnblogs.com/xiangsikai/p/8185031.html
"""
import threading, time
import queue

# 队列最多存入10个
queue_container = queue.Queue(maxsize=10)
# 最多产生的数据量
max_nums = 15


def producer(queue_container, max_nums):
    count = 1
    while True:
        # 　下载图片
        image_path = str(count) + ".jpg"
        queue_container.put(image_path)
        print("下载图片：{},放入队列中".format(image_path))
        count += 1
        time.sleep(1)
        if count == max_nums:
            break
    print("下载完成....................")


def consumer(queue_container):
    while True:
        image_path = queue_container.get()
        print("从队列取出图片，并处理图片：{}".format(image_path))
        time.sleep()
        print(queue_container.empty())
        # 用于通知queue_container.join()可以继续干其他事啦
    queue_container.task_done()


if __name__ == "__main__":
    # 生成线程
    p = threading.Thread(target=producer, args=(queue_container, max_nums))
    c = threading.Thread(target=consumer, args=(queue_container,))

    # 执行线程
    p.start()
    c.start()
