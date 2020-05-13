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

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height # Default: 1920, 1080

while True:
    # --- Getting frames and drawing
    if kinect.has_new_color_frame():
        frame = kinect.get_last_color_frame()
        frame = np.reshape(frame, (color_height, color_width,4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # Resize (1080, 1920, 4) into half (540, 960, 4)
        cv2.imshow('KINECT Video Stream', frame)
        frame = None

    key = cv2.waitKey(1)
    if key == 27:
        break
