# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-13 11:47:17
# --------------------------------------------------------
"""
import numpy as np
import cv2
from .calibration_store import load_stereo_coefficients


def get_stereo_coefficients(calibration_file, width, height):
    # Get cams params
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(calibration_file)
    config = {}
    config["K1"] = K1
    config["D1"] = D1
    config["K2"] = K2
    config["D2"] = D2
    config["R"] = R
    config["T"] = T
    config["E"] = E
    config["F"] = F
    config["R1"] = R1
    config["R2"] = R2
    config["P1"] = P1
    config["P2"] = P2
    config["Q"] = Q

    # Undistortion and Rectification part!
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    config["left_map_x"] = left_map_x
    config["left_map_y"] = left_map_y

    config["right_map_x"] = right_map_x
    config["right_map_y"] = right_map_y
    print("Q:{},focal_length:{}".format(Q, Q[2, 3]))
    return config


# 双目相机参数
class stereoCamera(object):
    def __init__(self, width=640, height=480):
        # 左相机内参
        # self.cam_matrix_left = np.array([[1499.64168081943, 0, 1097.61651199043],
        #                                  [0., 1497.98941910377, 772.371510027325],
        #                                  [0., 0., 1.]])
        self.cam_matrix_left = np.asarray([[4.1929128272967574e+02, 0., 3.2356123553538390e+02],
                                           [0., 4.1931862286777556e+02, 2.1942548262685406e+02],
                                           [0., 0., 1.]])
        # 右相机内参
        # self.cam_matrix_right = np.array([[1494.85561041115, 0, 1067.32184876563],
        #                                   [0., 1491.89013795616, 777.983913223449],
        #                                   [0., 0., 1.]])
        self.cam_matrix_right = np.asarray([[4.1680693687859372e+02, 0., 3.2769747052057716e+02],
                                            [0., 4.1688284886037280e+02, 2.3285709632482832e+02],
                                            [0., 0., 1.]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        # self.distortion_l = np.array([[-0.110331619900584, 0.0789239541458329, -0.000417147132750895,
        #                                0.00171210128855920, -0.00959533143245654]])
        self.distortion_l = np.asarray([[-2.9558582315073436e-02,
                                         1.5948145293240729e-01,
                                         -7.1046620767870137e-04,
                                         -6.5787270354389317e-04,
                                         -2.7169829618300961e-01]])
        # self.distortion_r = np.array([[-0.106539730103100, 0.0793246026401067, -0.000288067586478778,
        #                                -8.92638488356863e-06, -0.0161669384831612]])
        self.distortion_r = np.asarray([[-2.3391571805264716e-02,
                                         1.3648437316647929e-01,
                                         6.7233698457319337e-05,
                                         5.8610808515832777e-04,
                                         -2.3463198941301094e-01]])

        # 旋转矩阵
        # self.R = np.array([[0.993995723217419, 0.0165647819554691, 0.108157802419652],
        #                    [-0.0157381345263306, 0.999840084288358, -0.00849217121126161],
        #                    [-0.108281177252152, 0.00673897982027135, 0.994097466450785]])

        self.R = np.asarray([[9.9995518261153071e-01, 4.2888473189297411e-04, -9.4577389595457383e-03],
                             [-4.4122271031099070e-04, 9.9999905442083736e-01, -1.3024899043586808e-03],
                             [9.4571713984714298e-03, 1.3066044993798060e-03, 9.9995442630843034e-01]])

        # 平移矩阵
        # self.T = np.array([[-423.716923177417], [2.56178287450396], [21.9734621041330]])
        self.T = np.asarray([[-2.2987774547369614e-02], [3.0563972870288424e-05], [8.9781163185012418e-05]])

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.cam_matrix_left,
                                                          self.distortion_l,
                                                          self.cam_matrix_right,
                                                          self.distortion_r,
                                                          (width, height),
                                                          self.R,
                                                          self.T,
                                                          alpha=0)

        # 焦距
        # self.focal_length = 1602.46406  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]
        self.focal_length = Q[2, 3]  # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = self.T[0]  # 单位：mm， 为平移向量的第一个参数（取绝对值）
