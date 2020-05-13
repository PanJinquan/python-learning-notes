# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : kinect_for_rgbd.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 09:43:27
"""
import cv2
import numpy as np
import utils_PyKinectV2 as utils
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
from tools import image_processing

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body |
                                         PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Depth)

depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height  # Default: 512, 424
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height  # Default: 1920, 1080

depth_scale = 0.001  # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
# depth_scale                 = 1.0 # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
clipping_distance_in_meters = 4.080  # Set the maximum distance to display the point cloud data
clipping_distance = clipping_distance_in_meters / depth_scale  # Convert dist in mm to unit

while True:
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        body_frame = kinect.get_last_body_frame()
        color_frame = kinect.get_last_color_frame()
        depth_frame = kinect.get_last_depth_frame()
        # Reshape from 1D frame to 2D image ###
        color_img = color_frame.reshape(((color_height, color_width, 4))).astype(np.uint8)
        depth_img = depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16)
        # Useful functions in utils_PyKinectV2.py
        align_color_img = utils.get_align_color_image(kinect, color_img)
        rgbd_image = utils.get_rgbd_image(align_color_img, depth_img, depth_scale, clipping_distance_in_meters)

        # depth_img1 = np.uint8(depth_img.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
        depth_img1 = np.float32(depth_img * depth_scale).clip(0, clipping_distance_in_meters)  # 转换为uint8时，需要避免溢出255*16=4080
        depth_img2 = np.asarray(rgbd_image.depth)
        color_img2 = np.asarray(rgbd_image.color)
        image_processing.addMouseCallback("depth_img", depth_img1)
        image_processing.addMouseCallback("depth_img2", depth_img2)
        cv2.imshow('depth_img', depth_img1)
        cv2.imshow('align_color_img', align_color_img)
        cv2.imshow('depth_img2', depth_img2)
        cv2.imshow('color_img2', color_img2)
        cv2.waitKey(2)
