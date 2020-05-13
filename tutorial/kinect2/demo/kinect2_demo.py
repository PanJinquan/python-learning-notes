# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : kinect2_demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 10:13:09
"""

import cv2
import numpy as np
import utils_PyKinectV2 as utils
from core import posture
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
from tools import image_processing


class Kinect2Capture():
    def __init__(self):
        # Kinect runtime object
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body |
                                                      PyKinectV2.FrameSourceTypes_BodyIndex |
                                                      PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth |
                                                      PyKinectV2.FrameSourceTypes_Infrared)
        # depth image Default: 512, 424
        self.depth_width, self.depth_height = self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height
        # color image Default: 1920, 1080
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height

    def start_capture(self, ):
        count = 0
        while True:
            # Get images from camera
            if self.kinect.has_new_body_frame() and \
                    self.kinect.has_new_body_index_frame() and \
                    self.kinect.has_new_color_frame() and \
                    self.kinect.has_new_depth_frame() and \
                    self.kinect.has_new_infrared_frame():
                body_frame = self.kinect.get_last_body_frame()
                body_index_frame = self.kinect.get_last_body_index_frame()
                color_frame = self.kinect.get_last_color_frame()
                depth_frame = self.kinect.get_last_depth_frame()
                infrared_frame = self.kinect.get_last_infrared_frame()

                # Reshape from 1D frame to 2D image
                body_index_img = body_index_frame.reshape(((self.depth_height, self.depth_width))).astype(np.uint8)
                color_img = color_frame.reshape(((self.color_height, self.color_width, 4))).astype(np.uint8)
                depth_img = depth_frame.reshape(((self.depth_height, self.depth_width))).astype(np.uint16)
                infrared_img = infrared_frame.reshape(((self.depth_height, self.depth_width))).astype(np.uint16)

                # Useful functions in utils_PyKinectV2.py
                align_color_img = utils.get_align_color_image(self.kinect, color_img)
                # align_color_img = cv2.resize(color_img, (depth_width, depth_height))
                # Overlay body joints on align_color_img
                # align_color_img = utils.draw_bodyframe(body_frame, self.kinect, align_color_img, "depth_space")
                body_joint2D = utils.get_body_joint2D(body_frame, self.kinect, map_space="depth_space")
                posture.save_depth("./data", depth_img, align_color_img, body_joint2D, count)
                align_color_img = utils.draw_joint2D_in_image(body_joint2D, align_color_img)
                posture.define_pose(depth_img, align_color_img, body_joint2D)
                # align_color_img = utils.draw_bodyframe(body_frame, kinect, color_img,map_space="color_space")
                body_index_img = utils.color_body_index(self.kinect, body_index_img)  # Add color to body_index_img
                self.display(color_img, depth_img, infrared_img, align_color_img, body_index_img)
                count += 1

    def display(self, color_img, depth_img, infrared_img, align_color_img, body_index_img):
        '''
        Display 2D images using OpenCV
        :param color_img:      camera_size:(1080,1920,4)->(540, 960, 4)
        :param depth_img:      camera_size:(424, 512)
        :param infrared_img:   camera_size:(424, 512)
        :param align_color_img:camera_size:(424, 512)
        :param body_index_img: camera_size:(424, 512)
        :return:
        '''
        # Resize (1080, 1920, 4) into half (540, 960, 4)
        color_img_resize = cv2.resize(color_img, (0, 0), fx=0.5, fy=0.5)
        # Scale to display from 0 mm to 1500 mm
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=255 / 1500), cv2.COLORMAP_JET)
        # Scale from uint16 to uint8
        infrared_img = cv2.convertScaleAbs(infrared_img, alpha=255 / 65535)
        # image_processing.addMouseCallback("body_index_img", body_index_img, callbackFunc=None)
        # image_processing.addMouseCallback("align_color_img", align_color_img, callbackFunc=None)
        # image_processing.addMouseCallback("depth", depth_img, callbackFunc=None)
        # image_processing.addMouseCallback("infrared_img", infrared_img, callbackFunc=None)
        cv2.imshow('body_index_img', body_index_img)  # (424, 512)
        cv2.imshow('color_img_resize', color_img_resize)  # (540, 960, 4)
        cv2.imshow('align_color_img', align_color_img)  # (424, 512)
        cv2.imshow('depth', depth_colormap)  # (424, 512)
        cv2.imshow('infrared_img', infrared_img)  # (424, 512)
        key = cv2.waitKey(30)
        if key == 27:  # Press esc to break the loop
            exit(0)

    def close(self):
        self.kinect.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    kc = Kinect2Capture()
    kc.start_capture()
