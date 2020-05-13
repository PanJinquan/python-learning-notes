# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : ov0575_config.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-16 19:58:47
"""
import open3d

import cv2
import numpy as np

'''
003-0575
'''
# 内参矩阵
left_camera_matrix = np.array([[423.4771, 0.3143, 325.5406],
                               [0, 423.0318, 218.7728],
                               [0, 0, 1.0000]])
# k1，k2，p1，p2，k3
left_distortion = np.array([[-0.0497, 0.3680, -0.0012, -0.0000, 0]])

right_camera_matrix = np.array([[417.5506, 0.0691, 326.9406],
                                [0, 417.0232, 232.0576],
                                [0., 0., 1.]])
right_distortion = np.array([[-0.0255, 0.1823, -0.0013, -0.0007, 0]])

R = np.array([[1.0000, -0.0004, 0.0024],
              [0.0004, 1.0000, 0.0004],
              [-0.0024, -0.0004, 1.0000]])
T = -np.array([ -21.4070 ,   1.4387,   -6.0864])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print("camera_configs.Q:{}".format(Q))

joint_lines = [[1, 2], [1, 5], [2, 3],
               [3, 4], [5, 6], [6, 7],
               [1, 8], [8, 9], [9, 10], [1, 11],
               [11, 12], [12, 13], [1, 0], [0, 14],
               [14, 16], [0, 15], [15, 17]]

#######################
joint_count = 25
depth_width, depth_height = 640, 480  # 512, 424
# color_width, color_height = 1920, 1080

# User defined variables
depth_scale = 0.001  # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
# depth_scale                 = 1.0 # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
clipping_distance_in_meters = 1.5  # Set the maximum distance to display the point cloud data
clipping_distance = clipping_distance_in_meters / depth_scale  # Convert dist in mm to unit
# Hardcode the camera intrinsic parameters for backprojection
# width=depth_width; height=depth_height; ppx=258.981; ppy=208.796; fx=367.033; fy=367.033 # Hardcode the camera intrinsic parameters for backprojection
# fx = 428.30114
# fy = 427.00564
# ppx = 316.41648
# ppy = 218.34591

fx = left_camera_matrix[0, 0]
fy = left_camera_matrix[1, 1]
ppx = left_camera_matrix[0, 2]
ppy = left_camera_matrix[1, 2]

# Open3D visualisation
intrinsic = open3d.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, ppx, ppy)
# To convert [x,y,z] -> [x.-y,-z]
flip_transform = [[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]]

print("intrinsic:{}".format(intrinsic.intrinsic_matrix))
###################
