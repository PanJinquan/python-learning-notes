# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : __init__.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2020-04-10 18:24:06
"""

import sys

sys.path.append("./")
import cv2
import argparse
import sys
import numpy as np
from utils import image_processing
from core import camera_params
from modules.utils_3d import open3d_visual


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
        self.count = 0
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

    def get_rectify_image(self, image_left, image_right):
        '''
        畸变校正和立体校正
        根据更正map对图片进行重构
        获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        :param image_left:
        :param image_right:
        :return:
        '''
        # left_rectified = cv2.remap(image_left,
        #                            self.config["left_map_x"],
        #                            self.config["left_map_y"],
        #                            cv2.INTER_LINEAR)
        # right_rectified = cv2.remap(image_right,
        #                             self.config["right_map_x"],
        #                             self.config["right_map_y"],
        #                             cv2.INTER_LINEAR)
        left_rectified = cv2.remap(image_left,
                                   self.config["left_map_x"],
                                   self.config["left_map_y"],
                                   cv2.INTER_LINEAR,
                                   cv2.BORDER_CONSTANT)
        right_rectified = cv2.remap(image_right,
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
        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()
        # 畸变校正和立体校正
        left_rectified, right_rectified = self.get_rectify_image(image_left=leftFrame, image_right=rightFrame)
        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        # Get the disparity map
        disparity_image, displ = self.get_disparity(gray_left, gray_right)
        depth = self.get_depth(disparity=disparity_image)
        self.show(leftFrame, rightFrame, disparity_image, depth)
        self.pcd.show(color_image=leftFrame, depth_image=depth)

    def show(self, leftFrame, rightFrame, disparity_image, depth):
        """
        :param leftFrame:
        :param rightFrame:
        :param disparity_image:
        :param depth:
        :return:
        """
        depth_colormap = self.get_depth_colormap(depth, clipping_distance=4500)
        image_processing.addMouseCallback("depth_colormap", depth)
        image_processing.addMouseCallback("left", depth)
        image_processing.addMouseCallback("Disparity", disparity_image)
        # Show the images
        cv2.imshow('left', leftFrame)
        cv2.imshow('right', rightFrame)
        cv2.imshow('Disparity', disparity_image)
        cv2.imshow('depth_colormap', depth_colormap)
        cv2.waitKey(10)
        if self.count <= 2:
            cv2.moveWindow("left", 700, 0)
            cv2.moveWindow("right", 1400, 0)

            cv2.moveWindow("Disparity", 0, 600)
            cv2.moveWindow("depth_colormap", 700, 600)


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
