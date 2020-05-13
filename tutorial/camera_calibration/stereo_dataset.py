# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : __init__.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2020-04-10 18:24:06
"""

import sys
import os

sys.path.append("./")
import cv2
import argparse
import sys
import numpy as np
from utils import image_processing, file_processing
from core import camera_params
from modules.utils_3d import open3d_visual
from libs.ultra_ligh_face.ultra_ligh_face import UltraLightFaceDetector


class StereoDepth(object):
    """

    """

    def __init__(self, calibration_file, width=640, height=480):
        """
        :param calibration_file:
        :param width:
        :param height:
        """
        self.config = camera_params.get_stereo_coefficients(calibration_file, width, height)
        self.pcd = open3d_visual.Open3DVisual(camera_intrinsic=self.config["K1"],
                                              depth_width=width,
                                              depth_height=height)

        self.detector = UltraLightFaceDetector(model_path=None, network=None, )

        class_name = "ttt"
        real_part = True
        # real_part = False
        save_root = "dataset/"
        self.scale = 1
        self.prefix = "v"
        if real_part:
            self.save_dir = os.path.join(save_root, "real_part", class_name)
        else:
            self.save_dir = os.path.join(save_root, "fake_part", class_name)
        file_processing.create_dir(self.save_dir, "color")
        file_processing.create_dir(self.save_dir, "depth")
        file_processing.create_dir(self.save_dir, "ir")
        file_processing.create_dir(self.save_dir, "video")
        video_name = file_processing.get_time()
        self.save_l_video = os.path.join(self.save_dir, "video", "left_{}_{}.avi".format(class_name, video_name))
        self.save_r_video = os.path.join(self.save_dir, "video", "right_{}_{}.avi".format(class_name, video_name))
        if self.save_l_video:
            self.video_l_writer = self.get_video_writer(self.save_l_video,
                                                        width=width,
                                                        height=height,
                                                        fps=30)
        if self.save_r_video:
            self.video_r_writer = self.get_video_writer(self.save_r_video,
                                                        width=width,
                                                        height=height,
                                                        fps=30)

        self.pcd.show_image_pcd(True)
        self.pcd.show_origin_pcd(True)
        self.pcd.show_image_pcd(True)

    def capture(self, left_source, right_source):
        cap_left = cv2.VideoCapture(left_source)
        cap_right = cv2.VideoCapture(right_source)
        if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
            print("Can't opened the streams!")
            sys.exit(-9)
        # Change the resolution in need
        cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
        cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float
        cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
        cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float
        width, height, numFrames, fps = self.get_video_info(cap_left)
        self.count = 1
        while True:  # Loop until 'q' pressed or stream ends
            # Grab&retreive for sync images
            if not (cap_left.grab() and cap_right.grab()):
                print("No more frames")
                break
            self.count += 1
            self.task(cap_left, cap_right)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
                break

        # Release the sources.
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_video_info(video_cap):
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        numFrames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        print("video:width:{},height:{},fps:{}".format(width, height, fps))
        return width, height, numFrames, fps

    @staticmethod
    def get_video_writer(save_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frameSize = (width, height)
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
        print("video:width:{},height:{},fps:{}".format(width, height, fps))
        return video_writer

    def get_depth(self, disparity, scale=1.0, method=True):
        '''
        reprojectImageTo3D(disparity, Q),输入的Q,单位必须是毫米(mm)
        :param disparity:
        :param only_depth:
        :param scale:
        :return: returm scale=1.0,距离,单位为毫米
        '''
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        if method:
            depth = cv2.reprojectImageTo3D(disparity, self.config["Q"])  # 单位必须是毫米(mm)
            x, y, depth = cv2.split(depth)
        else:
            # baseline = 21.50635
            # fx = 419.29128272967574
            baseline = abs(self.config["T"][0])
            # fx = abs(self.config["K1"][0, 0])
            fx = abs(self.config["Q"][2, 3])
            depth = (fx * baseline) / disparity
        depth = depth * scale
        depth = np.asarray(depth, dtype=np.uint16)
        return depth

    @staticmethod
    def get_depth_colormap(depth, clipping_distance=1500):
        depth = np.clip(depth, 0, clipping_distance)
        depth_colormap = cv2.applyColorMap(
            # cv2.convertScaleAbs(depth, alpha=255 / clipping_distance),
            cv2.convertScaleAbs(depth, alpha=1),
            cv2.COLORMAP_JET)
        return depth_colormap

    def get_disparity(self, imgL, imgR):
        """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
        # SGBM Parameters -----------------
        window_size = 1  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        param = {'minDisparity': 0, 'numDisparities': 32, 'blockSize': 5, 'P1': 10, 'P2': 20, 'disp12MaxDiff': 1,
                 'preFilterCap': 65, 'uniquenessRatio': 10, 'speckleWindowSize': 150, 'speckleRange': 2, 'mode': 2}
        left_matcher = cv2.StereoSGBM_create(**param)
        # left_matcher = cv2.StereoSGBM_create(
        #     minDisparity=-1,
        #     numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        #     blockSize=window_size,
        #     P1=8 * 3 * window_size,
        #     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        #     P2=32 * 3 * window_size,
        #     disp12MaxDiff=12,
        #     uniquenessRatio=10,
        #     speckleWindowSize=50,
        #     speckleRange=32,
        #     preFilterCap=63,
        #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        # )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 8000
        sigma = 1.3
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        # 除以16得到真实视差（因为SGBM算法得到的视差是×16的）
        displ[displ < 0] = 0
        # disparity.astype(np.float32) / 16.
        displ = np.divide(displ.astype(np.float32), 16.)
        return filteredImg, displ

    def get_rectify_image(self, l_frame, r_frame):
        '''
        畸变校正和立体校正
        根据更正map对图片进行重构
        获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        :param l_frame:
        :param r_frame:
        :return:
        '''
        # left_rectified = cv2.remap(l_frame,
        #                            self.config["left_map_x"],
        #                            self.config["left_map_y"],
        #                            cv2.INTER_LINEAR)
        # right_rectified = cv2.remap(r_frame,
        #                             self.config["right_map_x"],
        #                             self.config["right_map_y"],
        #                             cv2.INTER_LINEAR)
        left_rectified = cv2.remap(l_frame,
                                   self.config["left_map_x"],
                                   self.config["left_map_y"],
                                   cv2.INTER_LINEAR,
                                   cv2.BORDER_CONSTANT)
        right_rectified = cv2.remap(r_frame,
                                    self.config["right_map_x"],
                                    self.config["right_map_y"],
                                    cv2.INTER_LINEAR,
                                    cv2.BORDER_CONSTANT)
        return left_rectified, right_rectified

    def task(self, cap_left, cap_right):
        """
        :param cap_left:
        :param cap_right:
        :return:
        """
        _, l_frame = cap_left.retrieve()
        _, r_frame = cap_right.retrieve()
        # 畸变校正和立体校正
        left_rectified, right_rectified = self.get_rectify_image(l_frame=l_frame, r_frame=r_frame)
        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        # Get the disparity map
        disparity_image, displ = self.get_disparity(gray_left, gray_right)
        depth = self.get_depth(disparity=disparity_image)
        bboxes, scores, landms = self.detector.detect(l_frame, isshow=False)
        self.show(l_frame, r_frame, disparity_image, depth, bboxes, scores)
        self.pcd.show(color_image=l_frame, depth_image=depth)

    def show(self, l_frame, r_frame, disparity_image, depth, bboxes, scores):
        """
        :param l_frame:
        :param r_frame:
        :param disparity_image:
        :param depth:
        :return:
        """
        depth_colormap = self.get_depth_colormap(depth, clipping_distance=4500)
        image_processing.addMouseCallback("depth_colormap", depth)
        image_processing.addMouseCallback("left", depth)
        image_processing.addMouseCallback("Disparity", disparity_image)
        det_img = image_processing.draw_image_bboxes_text(l_frame, bboxes, scores, color=(0, 0, 255))
        # Show the images
        cv2.imshow('det_img', det_img)
        cv2.imshow('left', l_frame)
        cv2.imshow('right', r_frame)
        cv2.imshow('Disparity', disparity_image)
        cv2.imshow('depth_colormap', depth_colormap)
        if self.count <= 2:
            cv2.moveWindow("det_img", 0, 0)
            cv2.moveWindow("left", 700, 0)
            cv2.moveWindow("right", 1400, 0)

            cv2.moveWindow("Disparity", 0, 600)
            cv2.moveWindow("depth_colormap", 700, 600)

        cv2.waitKey(10)
        self.save(l_img=l_frame,
                  r_img=r_frame,
                  depth_img=depth,
                  bboxes=bboxes)

    def save(self, l_img, r_img, depth_img, bboxes, freq=3):
        if self.save_dir and len(bboxes) > 0 and self.count % freq == 0:
            print("save image:{}".format(self.count))
            prefix = "{}_{}.png".format(self.prefix, self.count)
            cv2.imwrite(os.path.join(self.save_dir, "depth/{}".format(prefix)), depth_img)
            cv2.imwrite(os.path.join(self.save_dir, "color/{}".format(prefix)), l_img)
            cv2.imwrite(os.path.join(self.save_dir, "ir/{}".format(prefix)), r_img)
        if self.save_l_video and len(bboxes) > 0:
            # image_processing.cv_show_image("color_img", color_img, waitKey=1)
            # color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2RGB)  # 将BGR转为RGB
            self.video_l_writer.write(l_img)

        if self.save_r_video and len(bboxes) > 0:
            # image_processing.cv_show_image("color_img", color_img, waitKey=1)
            # color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2RGB)  # 将BGR转为RGB
            self.video_r_writer.write(r_img)


if __name__ == '__main__':
    # calibration_file = "config/config_wim/stereo_cam.yml"
    calibration_file = "config/config_linux/stereo_cam.yml"
    # calibration_file = "config/stereo_cam.yml"
    left_source = 1
    right_source = 0
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--calibration_file', type=str, default=calibration_file,
                        help='Path to the stereo calibration file')
    parser.add_argument('--left_source', type=str, default=left_source,
                        help='Left video or v4l2 device name')
    parser.add_argument('--right_source', type=str, default=right_source,
                        help='Right video or v4l2 device name')
    args = parser.parse_args()
    sd = StereoDepth(calibration_file)
    sd.capture(left_source=args.left_source, right_source=args.right_source)
