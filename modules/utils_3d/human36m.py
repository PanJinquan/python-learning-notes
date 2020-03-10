# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: Integral-Human-Pose-Regression-for-3D-Human-Pose-Estimation
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-04 16:03:01
# @url    : https://www.jianshu.com/p/c5627ad019df
# --------------------------------------------------------
"""
import sys
import os
from tools import image_processing

sys.path.append(os.getcwd())
import numpy as np
from modules.utils_3d import vis

camera_intrinsic = {
    # R，旋转矩阵
    "R": [[-0.91536173, 0.40180837, 0.02574754],
          [0.05154812, 0.18037357, -0.98224649],
          [-0.39931903, -0.89778361, -0.18581953]],
    # t，平移向量
    "T": [1841.10702775, 4955.28462345, 1563.4453959],
    # 焦距，f/dx, f/dy
    "f": [1145.04940459, 1143.78109572],
    # principal point，主点，主轴与像平面的交点
    "c": [512.54150496, 515.45148698]

}


class Human36M(object):
    @staticmethod
    def convert_wc_to_cc(joint_world):
        """
        世界坐标系 -> 相机坐标系: R * (pt - T)
        :return:
        """
        joint_world = np.asarray(joint_world)
        R = np.asarray(camera_intrinsic["R"])
        T = np.asarray(camera_intrinsic["T"])
        joint_num = len(joint_world)
        # 世界坐标系 -> 相机坐标系
        # [R|t] world coords -> camera coords
        joint_cam = np.zeros((joint_num, 3))  # joint camera
        for i in range(joint_num):  # joint i
            joint_cam[i] = np.dot(R, joint_world[i] - T)  # R * (pt - T)
        return joint_cam

    @staticmethod
    def __cam2pixel(cam_coord, f, c):
        """
        相机坐标系 -> 像素坐标系: (f / dx) * (X / Z) = f * (X / Z) / dx
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
        将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
        u = X * fx / Z + cx
        v = Y * fy / Z + cy
        D(v,u) = Z / Alpha
        =====================================================
        camera_matrix = [[428.30114, 0.,   316.41648],
                        [   0.,    427.00564, 218.34591],
                        [   0.,      0.,    1.]])

        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        =====================================================
        :param cam_coord:
        :param f: [fx,fy]
        :param c: [cx,cy]
        :return:
        """
        # 等价于：(f / dx) * (X / Z) = f * (X / Z) / dx
        # 三角变换， / dx, + center_x
        u = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
        v = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
        d = cam_coord[..., 2]
        return u, v, d

    @staticmethod
    def convert_cc_to_ic(joint_cam):
        """
        相机坐标系 -> 像素坐标系
        :param joint_cam:
        :return:
        """
        # 相机坐标系 -> 像素坐标系，并 get relative depth
        # Subtract center depth
        # 选择 Pelvis骨盆 所在位置作为相机中心，后面用之求relative depth
        root_idx = 0
        center_cam = joint_cam[root_idx]  # (x,y,z) mm
        joint_num = len(joint_cam)
        f = camera_intrinsic["f"]
        c = camera_intrinsic["c"]
        # joint image_dict，像素坐标系，Depth 为相对深度 mm
        joint_img = np.zeros((joint_num, 3))
        joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = Human36M.__cam2pixel(joint_cam, f, c)  # x,y
        joint_img[:, 2] = joint_img[:, 2] - center_cam[2]  # z
        return joint_img


if __name__ == "__main__":
    joint_world = [[-91.679, 154.404, 907.261],
                   [-223.23566, 163.80551, 890.5342],
                   [-188.4703, 14.077106, 475.1688],
                   [-261.84055, 186.55286, 61.438915],
                   [39.877888, 145.00247, 923.98785],
                   [-11.675994, 160.89919, 484.39148],
                   [-51.550297, 220.14624, 35.834396],
                   [-132.34781, 215.73018, 1128.8396],
                   [-97.1674, 202.34435, 1383.1466],
                   [-112.97073, 127.96946, 1477.4457],
                   [-120.03289, 190.96477, 1573.4],
                   [25.895456, 192.35947, 1296.1571],
                   [107.10581, 116.050285, 1040.5062],
                   [129.8381, -48.024918, 850.94806],
                   [-230.36955, 203.17923, 1311.9639],
                   [-315.40536, 164.55284, 1049.1747],
                   [-350.77136, 43.442127, 831.3473],
                   [-102.237045, 197.76935, 1304.0605]]
    joint_world = np.asarray(joint_world)
    kps_lines = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                 (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    # show in 世界坐标系
    vis.vis_3d(joint_world, kps_lines, coordinate="WC", title="WC")

    human36m = Human36M()

    # show in 相机坐标系
    joint_cam = human36m.convert_wc_to_cc(joint_world)
    vis.vis_3d(joint_cam, kps_lines, coordinate="CC", title="CC")
    joint_img = human36m.convert_cc_to_ic(joint_cam)

    # show in 像素坐标系
    kpt_2d = joint_img[:, 0:2]
    image_path = "/media/dm/dm1/git/python-learning-notes/modules/utils_3d/s_01_act_02_subact_01_ca_02_000001.jpg"
    image = image_processing.read_image(image_path)
    image = image_processing.draw_key_point_in_image(image, key_points=[kpt_2d], pointline=kps_lines)
    image_processing.cv_show_image("image_dict", image)
