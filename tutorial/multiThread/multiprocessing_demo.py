# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-18 10:59:05
# --------------------------------------------------------
"""

# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : __init__.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2020-04-10 18:24:06
"""

import os
import cv2
import glob
# import threading
import multiprocessing
from multiprocessing.process import BaseProcess


class DisplayTask(object):
    """
    """

    def __init__(self, ):
        """
        :param width:
        :param height:
        """
        self.win_name = "frame"

    def show(self, image_path):
        print("block here 1")
        frame = cv2.imread(image_path)
        print("block here 2")
        cv2.imshow(self.win_name, frame)
        print("block here 3")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()




def display_task(th_id, image_path):
    print("start thread-{}".format(th_id))
    dt = DisplayTask()
    dt.show(image_path)


def demo(image_dir):
    image_list = glob.glob(os.path.join(image_dir, "*.jpg"))
    print("have image:{}".format(len(image_list)))
    for th_id, image_path in enumerate(image_list):
        # thread = threading.Thread(target=display_task, args=(th_id, image_path,))
        # thread = multiprocessing.Process(target=display_task, args=(th_id, image_path,))
        thread = CustomProcess(target=display_task, args=(th_id, image_path,))
        thread.start()
        thread.terminate()
        # 等待线程执行完毕
        thread.join()
        print("finish thread-{}".format(th_id))


if __name__ == "__main__":
    image_dir = "/media/dm/dm1/git/python-learning-notes/data"
    demo(image_dir)
