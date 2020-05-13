# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-23 18:16:49
# @url    : https://www.cnblogs.com/subic/p/8296794.html
# --------------------------------------------------------
"""

import cv2
import numpy as np
import math
object_3d_points = np.array(([0, 0, 0],
                            [0, 200, 0],
                            [150, 0, 0],
                            [150, 200, 0]), dtype=np.double)
object_2d_point = np.array(([2985, 1688],
                            [5081, 1690],
                            [2997, 2797],
                            [5544, 2757]), dtype=np.double)
camera_matrix = np.array(([6800.7, 0, 3065.8],
                         [0, 6798.1, 1667.6],
                         [0, 0, 1.0]), dtype=np.double)
dist_coefs = np.array([-0.189314, 0.444657, -0.00116176, 0.00164877, -2.57547], dtype=np.double)
# 求解相机位姿
found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)
rotM = cv2.Rodrigues(rvec)[0]
camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
print(camera_postion.T)
# 验证根据博客http://www.cnblogs.com/singlex/p/pose_estimation_1.html提供方法求解相机位姿
# 计算相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。旋转顺序z,y,x
thetaZ = math.atan2(rotM[1, 0], rotM[0, 0])*180.0/math.pi
thetaY = math.atan2(-1.0*rotM[2, 0], math.sqrt(rotM[2, 1]**2 + rotM[2, 2]**2))*180.0/math.pi
thetaX = math.atan2(rotM[2, 1], rotM[2, 2])*180.0/math.pi
# 相机坐标系下值
x = tvec[0]
y = tvec[1]
z = tvec[2]
# 进行三次旋转
def RotateByZ(Cx, Cy, thetaZ):
    rz = thetaZ*math.pi/180.0
    outX = math.cos(rz)*Cx - math.sin(rz)*Cy
    outY = math.sin(rz)*Cx + math.cos(rz)*Cy
    return outX, outY
def RotateByY(Cx, Cz, thetaY):
    ry = thetaY*math.pi/180.0
    outZ = math.cos(ry)*Cz - math.sin(ry)*Cx
    outX = math.sin(ry)*Cz + math.cos(ry)*Cx
    return outX, outZ
def RotateByX(Cy, Cz, thetaX):
    rx = thetaX*math.pi/180.0
    outY = math.cos(rx)*Cy - math.sin(rx)*Cz
    outZ = math.sin(rx)*Cy + math.cos(rx)*Cz
    return outY, outZ
(x, y) = RotateByZ(x, y, -1.0*thetaZ)
(x, z) = RotateByY(x, z, -1.0*thetaY)
(y, z) = RotateByX(y, z, -1.0*thetaX)
Cx = x*-1
Cy = y*-1
Cz = z*-1
# 输出相机位置
print(Cx, Cy, Cz)
# 输出相机旋转角
print(thetaX, thetaY, thetaZ)
# 对第五个点进行验证
Out_matrix = np.concatenate((rotM, tvec), axis=1)
pixel = np.dot(camera_matrix, Out_matrix)
pixel1 = np.dot(pixel, np.array([0, 100, 105, 1], dtype=np.double))
pixel2 = pixel1/pixel1[2]
print(pixel2)