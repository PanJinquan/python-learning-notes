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
import cv2

sys.path.append(os.getcwd())
import numpy as np
from modules.utils_3d import vis_3d as vis
from utils import image_processing

# human36m_camera_intrinsic = {
#     # R，旋转矩阵
#     "R": [[-0.91536173, 0.40180837, 0.02574754],
#           [0.05154812, 0.18037357, -0.98224649],
#           [-0.39931903, -0.89778361, -0.18581953]],
#     # t，平移向量
#     "T": [1841.10702775, 4955.28462345, 1563.4453959],
#     # 焦距，f/dx, f/dy
#     "f": [1145.04940459, 1143.78109572],
#     # principal point，主点，主轴与像平面的交点
#     "c": [512.54150496, 515.45148698]
#
# }


human36m_camera_intrinsic = {
    # R，旋转矩阵
    "R": [[-0.91536173, 0.40180837, 0.02574754],
          [0.05154812, 0.18037357, -0.98224649],
          [-0.39931903, -0.89778361, -0.18581953]],
    # t，平移向量
    "T": [2097.3916015625, 4880.94482421875, 1605.732421875],
    # 焦距，f/dx, f/dy
    "f": [1145.0494384765625, 1143.7811279296875],
    # principal point，主点，主轴与像平面的交点
    "c": [512.54150390625, 515.4514770507812]

}

kinect2_camera_intrinsic = {

    # R，旋转矩阵
    "R": [[0.999853, -0.00340388, 0.0167495],
          [0.00300206, 0.999708, 0.0239986],
          [-0.0168257, -0.0239459, 0.999571]],
    # t，平移向量
    "T": [15.2562, 70.2212, -10.9926],
    # 焦距，f/dx, f/dy
    "f": [367.535, 367.535],
    # principal point，主点，主轴与像平面的交点
    "c": [260.166, 205.197]

}


def get_rotation(R):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    r = r.as_quat()
    print(r)


get_rotation(kinect2_camera_intrinsic["R"])


class KeyPointsVisual(object):
    t0 = np.asarray([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    t1 = np.asarray([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, -1]])

    @staticmethod
    def convert_wc_to_cc(joint_world):
        """
        世界坐标系 -> 相机坐标系: R * (pt - T):
        joint_cam = np.dot(R, (joint_world - T).T).T
        :return:
        """
        joint_world = np.asarray(joint_world)
        R = np.asarray(camera_intrinsic["R"])
        T = np.asarray(camera_intrinsic["T"])
        joint_num = len(joint_world)
        # 世界坐标系 -> 相机坐标系
        # [R|t] world coords -> camera coords
        # joint_cam = np.zeros((joint_num, 3))  # joint camera
        # for i in range(joint_num):  # joint i
        #     joint_cam[i] = np.dot(R, joint_world[i] - T)  # R * (pt - T)
        # .T is 转置, T is translation mat
        joint_cam = np.dot(R, (joint_world - T).T).T  # R * (pt - T)
        # joint_cam = np.dot((joint_world - T), R)
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
        joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = KeyPointsVisual.__cam2pixel(joint_cam, f, c)  # x,y
        joint_img[:, 2] = joint_img[:, 2] - center_cam[2]  # z
        return joint_img


import cv2

kinect2hum36m_id = [0, 12, 13, 14, 16, 17, 18, 1, 20, 2, 3, 8, 9, 10, 4, 5, 6]


def convert_kinect2h36m(kpt):
    h36m_joint = kpt[kinect2hum36m_id, :]
    return h36m_joint


def load_data(data_dir, flag):
    align_color_img_path = os.path.join(data_dir, "image", "image_{}.png".format(str(flag)))
    joint_path = os.path.join(data_dir, "joint", "joint_{}.npy".format(str(flag)))
    align_color_img = cv2.imread(align_color_img_path)
    joint3D = np.load(joint_path)
    return align_color_img, joint3D


def demo_for_human36m():
    from modules.utils_3d.data import human36m_data
    # x,y,z
    # joint_world = human36m_data.data0
    joint_world = human36m_data.data2 * 1000
    # joint_world = human36m_data.data1*1000
    joint_world = np.asarray(joint_world)
    kps_lines = human36m_data.kps_lines
    # show in 世界坐标系
    vis.vis_3d(joint_world, kps_lines, coordinate="WC", title="WC", set_lim=True)

    kp_vis = KeyPointsVisual()

    # show in 相机坐标系
    joint_cam = kp_vis.convert_wc_to_cc(joint_world)
    vis.vis_3d(joint_cam, kps_lines, coordinate="CC", title="CC", set_lim=True)
    joint_img = kp_vis.convert_cc_to_ic(joint_cam)

    # show in 像素坐标系
    kpt_2d = joint_img[:, 0:2]
    image_path = "/media/dm/dm1/git/python-learning-notes/modules/utils_3d/data/s_01_act_02_subact_01_ca_02_000001.jpg"
    image = image_processing.read_image(image_path)
    image = image_processing.draw_key_point_in_image(image, key_points=[kpt_2d], pointline=kps_lines)
    image_processing.cv_show_image("image_dict", image)


def demo_for_kinect():
    flip_transform = np.asarray([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]])

    # flip_transform = np.linalg.inv(flip_transform)
    # data_dir = "E:/git/python-learning-notes/tutorial/kinect2/dataset/kitnect3d"
    data_dir = "/media/dm/dm/X2/Pose/dataset/kitnet_data/panjinquan"  # flag= 5
    # data_dir = "/media/dm/dm/X2/Pose/dataset/kitnet_data/dengjianxiang"  # 241,245,348
    image, joint_world = load_data(data_dir, flag=503)
    h, w, d = image.shape
    joint_world = convert_kinect2h36m(joint_world)
    joint_world = joint_world * 1000
    joint_world = np.dot(-flip_transform, joint_world.T).T  # R * (pt - T)

    # kps_lines = [[0, 1], [1, 20], [20, 2], [2, 3],  # Spine
    #              [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],  # Left arm and hand
    #              [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],  # Right arm and hand
    #              [0, 12], [12, 13], [13, 14], [14, 15],  # Left leg
    #              [0, 16], [16, 17], [17, 18], [18, 19]]  # Right leg
    kps_lines = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                 (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

    # show in 世界坐标系
    vis.vis_3d(joint_world, kps_lines, coordinate="WC", title="WC", set_lim=True)

    kp_vis = KeyPointsVisual()

    # show in 相机坐标系
    joint_cam = kp_vis.convert_wc_to_cc(joint_world)
    vis.vis_3d(joint_cam, kps_lines, coordinate="CC", title="CC", set_lim=True)
    joint_cam = np.dot(flip_transform, joint_cam.T).T  # R * (pt - T)
    joint_img = kp_vis.convert_cc_to_ic(joint_cam)
    # show in 像素坐标系
    kpt_2d = joint_img[:, 0:2]
    image = image_processing.draw_key_point_in_image(image, key_points=[kpt_2d], pointline=kps_lines)
    image_processing.cv_show_image("image_dict", image)


if __name__ == "__main__":
    # camera_intrinsic = human36m_camera_intrinsic
    # demo_for_human36m()
    camera_intrinsic = kinect2_camera_intrinsic
    demo_for_kinect()
