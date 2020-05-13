# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : demo01.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-07 11:16:05
"""
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import numpy as np
import cv2

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)
depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height  # Default: 512, 424
# 添加点击事件，打印当前点的距离q
cv2.namedWindow("depth_image")
def callbackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(x,y)=({},{}),d={}".format(x,y,threeD[y][x]))
cv2.setMouseCallback("depth_image", callbackFunc, param=[1,2,3])

while True:
    # --- Getting frames and drawing
    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()
        threeD = np.reshape(frame, (depth_height, depth_width))
        image = threeD.astype(np.uint16)
        # image = cv2.convertScaleAbs(image, alpha=255 / 1500)
        image = np.uint8(image.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.imshow('depth_image', image)

    key = cv2.waitKey(1)
    if key == 27:
        break
