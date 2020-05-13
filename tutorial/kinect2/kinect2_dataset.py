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
                                                      PyKinectV2.FrameSourceTypes_Infrared |
                                                      PyKinectV2.FrameSourceTypes_Depth)
        self.depth_width, self.depth_height = self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height
        self.ir_width, self.ir_height = self.kinect.infrared_frame_desc.Width, self.kinect.infrared_frame_desc.Height
        self.g = geometry_3d_pose.Geometry3DPose(kinect_config)

        self.detector = UltraLightFaceDetector(model_path=None, network=None, )
        class_name = "0"
        # real_part = True
        real_part = False
        dataset_root = "dataset/"
        self.scale = 2
        self.prefix = "v1"
        if real_part:
            self.snapshot_dir = os.path.join(dataset_root, "real_part", class_name)
        else:
            self.snapshot_dir = os.path.join(dataset_root, "fake_part", class_name)
        file_processing.create_dir(self.snapshot_dir, "color")
        file_processing.create_dir(self.snapshot_dir, "depth")
        file_processing.create_dir(self.snapshot_dir, "ir")
        file_processing.create_dir(self.snapshot_dir, "video")
        video_name = file_processing.get_time()
        self.save_video = os.path.join(self.snapshot_dir, "video", "{}_{}.avi".format(class_name, video_name))
        if self.save_video:
            self.video_writer = self.get_video_writer(self.save_video,
                                                      width=self.depth_width * self.scale,
                                                      height=self.depth_height * self.scale,
                                                      fps=20)  # (424, 512, 4)

    @staticmethod
    def get_video_writer(save_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frameSize = (width, height)
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
        print("video:width:{},height:{},fps:{}".format(width, height, fps))
        return video_writer

    def start_capture(self):
        self.g.show_origin_pcd()
        self.g.show_bone_line_pcd()
        self.g.show_image_pcd(False)
        self.g.show_desktop_pcd(True)
        self.count = 0
        while True:
            # Get images from camera
            if self.kinect.has_new_body_frame() and self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():
                body_frame = self.kinect.get_last_body_frame()
                ir_frame = self.kinect.get_last_infrared_frame()
                color_frame = self.kinect.get_last_color_frame()
                depth_frame = self.kinect.get_last_depth_frame()
                # Reshape from 1D frame to 2D image
                color_img = color_frame.reshape(((self.color_height, self.color_width, 4))).astype(np.uint8)
                ir_frame = self.get_infrared_frame(ir_frame)
                self.ir_frame1 = np.asarray(ir_frame/2,np.uint8)
                depth_img = depth_frame.reshape(((self.depth_height, self.depth_width))).astype(np.uint16)
                align_color_img = utils.get_align_color_image(self.kinect, color_img)
                bgr_image = cv2.cvtColor(align_color_img, cv2.COLOR_RGBA2BGR)
                bboxes, scores, landms = self.detector.detect(bgr_image, isshow=False)
                self.show_landmark_boxes("Det", align_color_img, bboxes, scores, landms)
                # body_joint2D, body_orientation = utils.get_body_joint2D(body_frame,
                #                                                         self.kinect,
                #                                                         map_space="depth_space")
                # self.g.show(align_color_img, depth_img, body_joint2D,joint_count=PyKinectV2.JointType_Count)
                # self.g.show(align_color_img, depth_img, body_joint2D)
                self.show(color_img, align_color_img, depth_img, ir_frame, bboxes)

    def get_infrared_frame(self, ir_frame):
        """
        :param ir_frame:
        :return:
        """
        ir_frame = ir_frame.reshape(((self.ir_height, self.ir_width)))
        ir_frame = np.uint8(ir_frame.clip(1, 4000) / 16.)
        # ir_frame = np.uint8(ir_frame.clip(1, 4000))
        ir_frame = np.dstack((ir_frame, ir_frame, ir_frame))

        # ir_frame = img_as_float(ir_frame)
        # ir_frame = exposure.adjust_gamma(image, 0.5)  # 调亮
        return ir_frame

    @staticmethod
    def show_landmark_boxes(win_name, image, bboxes, scores, landms):
        '''
        显示landmark和boxes
        :param win_name:
        :param image:
        :param landmarks_list: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        '''
        image = image_processing.draw_landmark(image, landms, vis_id=True)
        image = image_processing.draw_image_bboxes_text(image, bboxes, scores, color=(0, 0, 255))
        image_processing.cv_show_image(win_name, image, waitKey=1)

    def show(self, color_img, align_color_img, depth_img, ir_frame, bboxes):
        """
        :param color_img: <class 'tuple'>: (1080, 1920, 4),uint8
        :param align_color_img: <class 'tuple'>: (424, 512, 4),uint8
        :param depth_img: <class 'tuple'>: (424, 512),uint16
        :param ir_frame: <class 'tuple'>: (424, 512, 3),uint8
        :return:
        """
        # align_color_img = cv2.cvtColor(align_color_img, cv2.COLOR_RGBA2RGB)  # 将BGR转为RGB
        image_processing.addMouseCallback("depth_img", param=depth_img)
        image_processing.addMouseCallback("align_color_img", param=depth_img)

        # image_processing.cv_show_image("color_img", color_img, waitKey=1)
        image_processing.cv_show_image("align_color_img", align_color_img, waitKey=1)
        image_processing.cv_show_image("depth_img", depth_img, waitKey=1)
        image_processing.cv_show_image("ir_frame", ir_frame, waitKey=1)
        image_processing.cv_show_image("ir_frame1",  self.ir_frame1, waitKey=1)
        self.count += 1
        freq = 3
        # self.flag = cv2.waitKey(20) & 0xFF == ord('s')
        if self.snapshot_dir and len(bboxes) > 0 and self.count % freq == 0:
            print("save image:{}".format(self.count))
            # self.flag = True
            # pre = "{}.png".format(self.count)
            prefix = "{}_{}.png".format(self.prefix, self.count)
            cv2.imwrite(os.path.join(self.snapshot_dir, "depth/{}".format(prefix)), depth_img)
            cv2.imwrite(os.path.join(self.snapshot_dir, "color/{}".format(prefix)), align_color_img)
            cv2.imwrite(os.path.join(self.snapshot_dir, "ir/{}".format(prefix)), ir_frame)
        cv2.waitKey(20)
        self.save_videos(color_img, align_color_img, depth_img, ir_frame, bboxes)

    def save_videos(self, color_img, align_color_img, depth_img, ir_frame, bboxes):
        """
        (424, 512, 4)
        :param color_img:
        :param align_color_img:
        :param depth_img:
        :param ir_frame:
        :param bboxes:
        :return:
        """
        if self.save_video and len(bboxes) > 0:
            color_img = image_processing.resize_image(color_img, resize_height=self.depth_height * self.scale)
            # <class 'tuple'>: (424, 753, 4)
            # <class 'tuple'>: (848, 1024, 3)
            color_img = image_processing.center_crop(color_img,
                                                     crop_size=[self.depth_height * self.scale,
                                                                self.depth_width * self.scale])
            # image_processing.cv_show_image("color_img", color_img, waitKey=1)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2RGB)  # 将BGR转为RGB
            self.video_writer.write(color_img)

    def close(self):
        self.kinect.close()
        self.g.close()


if __name__ == "__main__":
    kc = Kinect2PointCloud()
    kc.start_capture()
