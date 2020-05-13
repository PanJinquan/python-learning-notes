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

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Infrared)
depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height # Default: 512, 424

while True:
    # --- Getting frames and drawing
    if kinect.has_new_infrared_frame():
        frame = kinect.get_last_infrared_frame()
        frame = frame.astype(np.uint16)              # infrared frame是uint16类型
        frame = np.uint8(frame.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
        frame = np.reshape(frame, (depth_height, depth_width))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow('KINECT Video Stream', frame)
        frame = None

    key = cv2.waitKey(1)
    if key == 27:
        break
