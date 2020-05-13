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
from libs.ultra_ligh_face.ultra_ligh_face import UltraLightFaceDetector


class Kinect2PointCloud():
    def __init__(self):
        # Kinect runtime object
        self.joint_count = PyKinectV2.JointType_Count  # 25
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body |
                                                      PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height

        self.detector = UltraLightFaceDetector(model_path=None, network=None, )

        self.g = geometry_3d_pose.Geometry3DPose(kinect_config)

    def start_capture(self, save_dir="", name=""):
        self.g.show_origin_pcd(False)
        self.g.show_bone_line_pcd(True)
        self.g.show_image_pcd(False)
        self.g.show_desktop_pcd(False)
        save_dir = os.path.join(save_dir, name)
        count = 0
        save_freq = 3
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
                # front = self.detect_face(color_img)
                body_joint2D, body_orientation = utils.get_body_joint2D(body_frame,
                                                                        self.kinect,
                                                                        map_space="depth_space")
                # if not front:
                #     body_joint2D = self.check_kpt_front(body_joint2D)
                self.g.show(align_color_img, depth_img, body_joint2D)
                body_joint2D = [d for d in body_joint2D if not d is None]
                if body_joint2D and count % save_freq == 0:
                    self.save(save_dir, flag=count)
                count += 1

    def detect_face(self, image, front_tf=0.1):
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        bboxes, scores, landms = self.detector.detect(image)
        front = False
        for s in scores:
            if s > front_tf:
                front = True
            else:
                front = False
                break
        image = image_processing.draw_image_bboxes_text(image, bboxes, scores, color=(0, 0, 255))
        image = image_processing.resize_image(image, resize_height=600)
        cv2.imshow("Det", image)
        cv2.waitKey(5)
        return front

    def check_kpt_front(self, body_joint2D):
        pair_kpt = [(4, 8), (5, 9), (6, 10), (7, 11), (22, 24), (21, 23), (12, 16), (13, 17), (14, 18), (15, 19)]
        for k in range(len(body_joint2D)):
            joint = body_joint2D[k]
            if joint is None:
                continue
            for i, j in pair_kpt:
                tmp_i = joint[i, :].copy()
                tmp_j = joint[j, :].copy()
                joint[i, :] = tmp_j
                joint[j, :] = tmp_i
            body_joint2D[k] = joint
        return body_joint2D

    def close(self):
        self.kinect.close()
        self.g.close()

    def save(self, save_dir, flag):
        if save_dir:
            self.g.save_data(save_dir, flag)


if __name__ == "__main__":
    kc = Kinect2PointCloud()
    save_dir = "F:/X2/Pose/dataset/kitnet_data"
    # name = "panjinquan"
    name = "test2"
    # name = "dengjianxiang"
    kc.start_capture(save_dir, name)
