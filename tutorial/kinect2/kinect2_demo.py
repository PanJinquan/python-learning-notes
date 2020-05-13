# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : kinect_for_rgbd.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 09:43:27
"""
########################################################################
### Sample program to stream
### Coloured point cloud, joint and joint orientation in 3D using Open3D
########################################################################
import os
import cv2
import numpy as np
import tutorial.kinect2.demo.examples.utils_PyKinectV2 as utils
import open3d as open3d
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
from utils import image_processing, file_processing
from modules.utils_3d.core import geometry_3d_pose
from tutorial.kinect2.config import kinect_config


class Kinect2PointCloud():
    def __init__(self):
        # Kinect runtime object
        self.joint_count = PyKinectV2.JointType_Count  # 25
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body |
                                                      PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height
        self.g = geometry_3d_pose.Geometry3DPose(kinect_config)

    def start_capture(self):
        self.g.show_origin_pcd(True)
        self.g.show_bone_line_pcd(True)
        self.g.show_image_pcd(False)
        self.g.show_desktop_pcd(False)
        while True:
            # Get images from camera
            if self.kinect.has_new_body_frame() and self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():
                body_frame = self.kinect.get_last_body_frame()
                # ir_frame = self.kinect.get_last_infrared_frame()
                color_frame = self.kinect.get_last_color_frame()
                depth_frame = self.kinect.get_last_depth_frame()
                # Reshape from 1D frame to 2D image
                color_img = color_frame.reshape(((self.color_height, self.color_width, 4))).astype(np.uint8)
                depth_img = depth_frame.reshape(((self.depth_height, self.depth_width))).astype(np.uint16)
                align_color_img = utils.get_align_color_image(self.kinect, color_img)
                body_joint2D, body_orientation = utils.get_body_joint2D(body_frame,
                                                                        self.kinect,
                                                                        map_space="depth_space")
                # self.g.show(align_color_img, depth_img, body_joint2D,joint_count=PyKinectV2.JointType_Count)
                self.g.show(align_color_img, depth_img, body_joint2D)

    def close(self):
        self.kinect.close()
        self.g.close()


if __name__ == "__main__":
    kc = Kinect2PointCloud()
    kc.start_capture()
