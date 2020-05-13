# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : ov0575_config.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-16 19:58:47
"""
import open3d

'''
left_camera_matrix = np.array([[428.30114, 0., 316.41648],
                               [0., 427.00564, 218.34591],
                               [0., 0., 1.]])
left_distortion = np.array([[0.07256, -0.97506, 0.00419, -0.00546, 0.00000]])
'''

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
fx = 344.88116
fy = 345.29551
ppx = 237.33633
ppy = 317.37912

# Open3D visualisation
intrinsic = open3d.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, ppx, ppy)
# To convert [x,y,z] -> [x.-y,-z]
flip_transform = [[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]]

print("intrinsic:{}".format(intrinsic.intrinsic_matrix))
###################

import cv2
import numpy as np

'''
003-0575
'''
# 内参矩阵
left_camera_matrix = np.array([[344.88116, 0., 237.33633],
                               [0., 345.29551, 317.37912],
                               [0., 0., 1.]])
left_distortion = np.array([[ 0.00654,   -0.00752,   -0.00172,  0.00056,  0.00000]])

right_camera_matrix = np.array([[ 348.52150 , 0., 239.96762],
                                [0.,348.89231, 334.61484],
                                [0., 0., 1.]])
right_distortion = np.array([[0.00617 ,  -0.00256,   -0.00223,   0.00051,  0.00000]])

# right_camera_matrix = np.array([[428.30114, 0., 316.41648],
#                                [0., 427.00564, 218.34591],
#                                [0., 0., 1.]])
# right_distortion = np.array([[0.07256, -0.97506, 0.00419, -0.00546, 0.00000]])
#
# left_camera_matrix = np.array([[423.98223, 0., 323.12905],
#                                 [0., 422.07887, 230.53264],
#                                 [0., 0., 1.]])
# left_distortion = np.array([[-0.06160, 0.09230, -0.00474, -0.00398, 0.00000]])

om = np.array([ -0.00164 ,  0.00579,  0.00132])  # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([23.93128,   3.19508,  8.08959])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

print("camera_configs.Q:{}".format(Q))