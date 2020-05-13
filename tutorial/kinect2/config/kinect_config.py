# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : config.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-10 19:55:33
    @url:   : https://blog.csdn.net/aichipmunk/article/details/9264703
"""
import numpy as np
import open3d
joint_count = 25
depth_width, depth_height = 512, 424
color_width, color_height = 1920, 1080

# User defined variables
depth_scale = 0.001  # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
# depth_scale                 = 1.0 # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
clipping_distance_in_meters = 1.5  # Set the maximum distance to display the point cloud data
clipping_distance = clipping_distance_in_meters / depth_scale  # Convert dist in mm to unit
# Hardcode the camera intrinsic parameters for backprojection
# width=depth_width; height=depth_height; ppx=258.981; ppy=208.796; fx=367.033; fy=367.033 # Hardcode the camera intrinsic parameters for backprojection

# ppx = 260.166
# ppy = 205.197
# fx = 367.535
# fy = 367.535

# camera_intrinsic=intrinsic.intrinsic_matrix
camera_intrinsic = np.asarray([[367.535, 0., 260.166],
                               [0., 367.535, 205.197],
                               [0., 0., 1.]])
fx = camera_intrinsic[0, 0]
fy = camera_intrinsic[1, 1]
ppx = camera_intrinsic[0, 2]
ppy = camera_intrinsic[1, 2]
# Open3D visualisation
intrinsic = open3d.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, ppx, ppy).intrinsic_matrix


# To convert [x,y,z] -> [x.-y,-z]
flip_transform = [[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]]

joint_lines = [[0, 1], [1, 20], [20, 2], [2, 3],  # Spine
               [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],  # Left arm and hand
               [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],  # Right arm and hand
               [0, 12], [12, 13], [13, 14], [14, 15],  # Left leg
               [0, 16], [16, 17], [17, 18], [18, 19]]  # Right leg

print("camera_intrinsic:{}".format(camera_intrinsic))
print("intrinsic:{}".format(intrinsic))
