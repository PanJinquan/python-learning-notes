# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : download_image.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-06 15:51:05
"""

import time
from multiprocessing.pool import ThreadPool
import requests
import os
import PIL.Image as Image
from io import BytesIO
import cv2
import numpy as np
save_image_dir = None

def  download_image(url,):
    '''
    根据url下载图片
    :param url:
    :return: 返回保存的图片途径
    '''
    basename=os.path.basename(url)
    try:
        res = requests.get(url)
        if res.status_code == 200:
            print("download image successfully:{}".format(url))
            filename = os.path.join(save_image_dir,basename)
            with open(filename, "wb") as f:
                content=res.content
                # 使用Image解码为图片
                # image = Image.open(BytesIO(content))
                # image.show()
                # 使用opencv解码为图片
                content = np.asarray(bytearray(content), dtype="uint8")
                image = cv2.imdecode(content, cv2.IMREAD_COLOR)
                cv2.imshow("Image", image)
                cv2.waitKey(3000)
                f.write(content)
            return filename
    except Exception as e:
        print(e)
        return None
    print("download image failed:{}".format(url))
    return None


def download_image_thread(url_list,our_dir,num_processes,remove_bad=False):
    '''
    多线程下载图片
    :param url_list: image url list
    :param our_dir:  保存图片的路径
    :param num_processes: 开启线程个数
    :param remove_bad: 是否去除下载失败的数据
    :return:
    '''
    # 开启多线程
    global save_image_dir
    if not os.path.exists(our_dir):
        os.makedirs(our_dir)
    save_image_dir = our_dir
    pool = ThreadPool(processes=num_processes)# Sets the pool size to 4
    image_list = pool.map(download_image,url_list)
    pool.close()
    pool.join()
    if remove_bad:
        image_list = [i for i in image_list if i is not None]
    return image_list

if __name__=="__main__":
    our_dir="./image"
    # url_list=["https://farm3.staticflickr.com/2334/1828778894_271415878a_o.jpg",
    #       "https://farm4.staticflickr.com/3149/2355285447_290193393a_o.jpg",
    #       "https://farm3.staticflickr.com/2090/1792526652_8f37410561_o.jpg",
    #       "https://farm3.staticflickr.com/2099/1791684639_044827f860_o.jpg"]

    id_list= ["000000001.jpg", "000000002.jpg", "000000003.jpg", "000000004.jpg"]
    storage_url="http://192.168.4.50:8000/image/"
    url_list=[os.path.join(storage_url,id) for id in id_list]
    # url_list=[storage_url+id for id in id_list]
    startTime = time.time()
    # 不开启多线程
    # length = len(url_list);
    # for i in range(length):
    #     download_image(url_list[i])
    image_list=download_image_thread(url_list,our_dir=our_dir,num_processes=4,remove_bad=True)
    endTime = time.time()
    consumeTime = endTime - startTime
    print("程序运行时间："+str(consumeTime)+" 秒")
    print(image_list)

