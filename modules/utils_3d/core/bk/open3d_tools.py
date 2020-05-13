# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : open3d_tools.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-11 09:23:21
"""
# -*- coding: utf-8 -*-

import numpy as np
import open3d as open3d
import copy
import cv2
from tools import image_processing


def create_line_set_bones(joints, joint_line):
    # Draw the 24 bones (lines) connecting 25 joints
    # The lines below is the kinematic tree that defines the connection between parent and child joints

    colors = [[0, 0, 1] for i in range(24)]  # Default blue
    line_set = open3d.LineSet()
    line_set.lines = open3d.Vector2iVector(joint_line)
    line_set.colors = open3d.Vector3dVector(colors)
    line_set.points = open3d.Vector3dVector(joints)

    return line_set


def get_valid_joints(joints, joint_line):
    ff = np.zeros((24, 3))


def create_color_point_cloud(align_color_img, depth_img,
                             depth_scale, clipping_distance_in_meters, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    ppx = intrinsic[0, 2]
    ppy = intrinsic[1, 2]
    depth_height, depth_width = depth_img.shape
    intrinsic = open3d.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, ppx, ppy)
    rgbd_image = get_rgbd_image(align_color_img, depth_img, depth_scale, clipping_distance_in_meters)
    pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
    # pcd = open3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, open3d.camera.PinholeCameraIntrinsic(
    #     open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Point cloud only without color
    # pcd = create_point_cloud_from_depth_image(
    #     Image(depth_img),
    #     intrinsic,
    #     depth_scale=1.0/depth_scale,
    #     depth_trunc=clipping_distance_in_meters)

    return pcd.points, pcd.colors


def get_rgbd_image(align_color_img, depth_img, depth_scale, clipping_distance_in_meters):
    align_color_img = align_color_img[:, :, 0:3]  # Only get the first three channel
    align_color_img = align_color_img[..., ::-1]  # Convert opencv BGR to RGB
    rgbd_image = open3d.create_rgbd_image_from_color_and_depth(
        open3d.Image(align_color_img.copy()),
        open3d.Image(depth_img),
        depth_scale=1.0 / depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity=False)
    # rgbd_image = open3d.geometry.create_rgbd_image_from_color_and_depth(open3d.Image(align_color_img.copy()),
    #                                                                     open3d.Image(depth_img),)
    return rgbd_image


def get_single_joint3D_orientation(body_joint3D, body_orientation, joint_count):
    '''
    Currently only return single set of joint3D and orientations
    :param body_joint3D:
    :param body_orientation: if None,will
    :return:
    '''
    joint3D = np.zeros((joint_count, 3), dtype=np.float32)
    orientation = np.zeros((joint_count, 4), dtype=np.float32)
    if body_orientation is None:
        body_orientation = [orientation] * len(body_joint3D)
    for j, o in zip(body_joint3D, body_orientation):
        if j is None:
            continue
        joint3D = j
        orientation = o
    return joint3D, orientation


# Define the BGR color for 6 different bodies
colors_order = [(0, 0, 255),  # Red
                (0, 255, 0),  # Green
                (255, 0, 0),  # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (255, 255, 0)]  # Cyan


def draw_joint2D_in_image(body_joint2D, image, joint_lines):
    '''

    :param body_joint2D: list(ndarray(19,2)) or ndarray(n_person,19,3)
    :param image:
    :param joint_lines:
    :return:
    '''
    img = copy.deepcopy(image)
    for i, joint2D in enumerate(body_joint2D):
        if joint2D is None:
            continue
        img = draw_joint2D(img, joint2D, colors_order[i])  # 先用“o”画关节点
        if joint_lines is not None:
            img = draw_bone2D(img, joint2D, joint_lines, colors_order[i])  # 再用“--”连接关节点
    return img


def draw_joint2D(img, j2D, color=(0, 0, 255)):  # Default red circles
    j2D = np.asarray(j2D, dtype=np.int32)
    for i, point in enumerate(j2D):  # Should loop 25 times
        if point is None:
            continue
        cv2.circle(img, tuple(point), 5, color, -1)
        cv2.putText(img, '%d' % (i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    return img


def draw_bone2D_v2(img, j2D, color=(0, 0, 255)):  # Default red lines
    # Define the kinematic tree where each of the 25 joints is associated to om parent joint
    k = [0, 0, 1, 2,  # Spine
         20, 4, 5, 6,  # Left arm
         20, 8, 9, 10,  # Right arm
         0, 12, 13, 14,  # Left leg
         0, 16, 17, 18,  # Right leg
         1,  # Spine
         7, 7,  # Left hand
         11, 11]  # Right hand
    for i in range(j2D.shape[0]):  # Should loop 25 times
        if j2D[k[i], 0] > 0 and j2D[k[i], 1] > 0 and j2D[i, 0] > 0 and j2D[i, 1] > 0:
            cv2.line(img, (j2D[k[i], 0], j2D[k[i], 1]), (j2D[i, 0], j2D[i, 1]), color)
    return img


def draw_bone2D(img, j2D, joint_line, color=(0, 0, 255)):  # Default red lines
    j2D = np.asarray(j2D, dtype=np.int32)
    for line_point in joint_line:  # Should loop 25 times
        point1 = j2D[line_point[0]]
        point2 = j2D[line_point[1]]
        if point1 is None or point2 is None:
            continue
        if sum(point1) > 0 and sum(point2) > 0:
            cv2.line(img, tuple(point1), tuple(point2), color)
    return img


def convert_point3D_2D(point_3d, intrinsic, depth_scale):
    '''
    cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
    将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
    u = X * fx / Z + cx
    v = Y * fy / Z + cy
    D(v,u) = Z / Alpha
    :param point_2d:
    :param depth_img:
    :param intrinsic:
    :param depth_scale:
    :return:
    '''
    if len(point_3d.shape) == 1:
        point_3d = point_3d.reshape(1, 3)
    # fx = intrinsic.intrinsic_matrix[0, 0]
    # fy = intrinsic.intrinsic_matrix[1, 1]
    # cx = intrinsic.intrinsic_matrix[0, 2]
    # cy = intrinsic.intrinsic_matrix[1, 2]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    # Back project the 2D points to 3D coor
    point_num = len(point_3d)
    point_2d = np.zeros((point_num, 2), dtype=np.float32)  # [25, 3] Note: Total 25 joints
    point_depth = np.zeros((point_num, 1), dtype=np.float32)  # [25, 3] Note: Total 25 joints
    for i in range(point_num):
        X, Y, Z = point_3d[i, 0], point_3d[i, 1], point_3d[i, 2]
        # point_2d[i, 0] = X * fx / Z + cx  # u
        # point_2d[i, 1] = Y * fy / Z + cy  # v
        point_2d[i, 0] = np.where(Z == 0, 0, X * fx / Z + cx)
        point_2d[i, 1] = np.where(Z == 0, 0, Y * fy / Z + cy)
        point_depth[i, 0] = Z / depth_scale
    return point_2d, point_depth


def convert_point2D_3D(point_2d, depth_img, intrinsic, depth_scale):
    '''
    cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
    设像素坐标点P(u,v)，深度图D，单位m/米,将像素坐标P(u,v)从2D映射到3D坐标计算公式为：
    Z = Alpha*D(v,u) # 其中Alpha是缩放因子
    X =(u - cx) * Z / fx
    Y =(v - cy) * Z / fy
    :param point_2d:
    :param depth_img:
    :param intrinsic:
    :param depth_scale:
    :return:
    '''
    # fx = intrinsic.intrinsic_matrix[0, 0]
    # fy = intrinsic.intrinsic_matrix[1, 1]
    # cx = intrinsic.intrinsic_matrix[0, 2]
    # cy = intrinsic.intrinsic_matrix[1, 2]
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    # Back project the 2D points to 3D coor
    point_num = len(point_2d)
    point_3d = np.zeros((point_num, 3), dtype=np.float32)  # [25, 3] Note: Total 25 joints
    for i in range(point_num):
        u, v = point_2d[i, 0], point_2d[i, 1]
        point_3d[i, 2] = depth_img[v, u] * depth_scale  # Z coor
        point_3d[i, 0] = (u - cx) * point_3d[i, 2] / fx  # X coor
        point_3d[i, 1] = (v - cy) * point_3d[i, 2] / fy  # Y coor
    return point_3d


def convert_point2D_3D_list(point_2d_list, depth_img, intrinsics, depth_scale):
    point_3d_list = []
    for point_2d in point_2d_list:
        if point_2d is None:
            point_3d_list.append(None)
        else:
            point_3d = convert_point2D_3D(point_2d, depth_img, intrinsics, depth_scale)
            point_3d_list.append(point_3d)
    return point_3d_list


def compute_joint3D_distance(joint3D, index, source=None):
    '''
    :param joint3D: numpy
    :param index: index
    :param source: numpy,默认为原点
    :return:
    '''
    source_pcd = open3d.geometry.PointCloud()  # 定义点云
    if source is None:
        source = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    source_pcd.points = open3d.Vector3dVector(source)

    target = joint3D[index]
    target = target.reshape(1, 3)
    # d2=compute_distance(source, target)
    target_pcd = open3d.geometry.PointCloud()  # 定义点云
    target_pcd.points = open3d.Vector3dVector(target)
    d = open3d.geometry.compute_point_cloud_to_point_cloud_distance(source_pcd, target_pcd)
    return d


def compute_distance(vector1, vector2):
    d = np.sqrt(np.sum(np.square(vector1 - vector2)))
    # d = np.linalg.norm(vector1 - vector2)
    return d


def compute_point2area_distance(area_point, target_point):
    point1 = area_point[0, :]
    point2 = area_point[1, :]
    point3 = area_point[2, :]
    point4 = target_point
    d = point2area_distance(point1, point2, point3, point4)
    return d


def compute_point2point_distance(area_point, target_point):
    # point1 = area_point[0, :]
    # point2 = area_point[1, :]
    # point3 = area_point[2, :]
    mean_point = np.mean(area_point, axis=0)
    d = np.sqrt(np.sum(np.square(mean_point - target_point)))
    # d = np.linalg.norm(point1 - target_point)
    return d


def define_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    空间上某点：p={x0,y0,z0}
    点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
    https://wenku.baidu.com/view/12b44129af45b307e87197e1.html
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D


def point2area_distance(point1, point2, point3, point4):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param point4:
    :return:点到面的距离
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


def gen_vector(point1, point2):
    '''
    P12 = point2-point1
    :param point1:
    :param point2:
    :return:
    '''
    if not isinstance(point1, np.ndarray):
        point1 = np.asarray(point1, dtype=np.float32)
    if not isinstance(point2, np.ndarray):
        point2 = np.asarray(point2, dtype=np.float32)
    return point2 - point1


def gen_2vector(P1, P2, Q1, Q2):
    '''
    P12 = P2-P1
    Q21 = Q2-Q1
    :param P1:
    :param P2:
    :param Q1:
    :param Q2:
    :return:
    '''
    v1 = gen_vector(P1, P2)
    v2 = gen_vector(Q1, Q2)
    return v1, v2


def radian2angle(radian):
    '''弧度->角度'''
    angle = radian * (180 / np.pi)
    return angle


def angle2radian(angle):
    '''角度 ->弧度'''
    radian = angle * np.pi / 180.0
    return radian


def compute_point_angle(P1, P2, Q1, Q2):
    x, y = gen_2vector(P1, P2, Q1, Q2)
    angle = compute_vector_angle(x, y, minangle=True)
    return angle


def compute_vector_angle(a, b, minangle=True):
    '''
    cosφ = u·v/|u||v|
    https://wenku.baidu.com/view/301a6ba1250c844769eae009581b6bd97f19bca3.html?from=search
    :param a:
    :param b:
    :return:
    '''
    # 两个向量
    x = np.array(a)
    y = np.array(b)
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    value = x.dot(y) / ((Lx * Ly) + 1e-6)  # cosφ = u·v/|u||v|
    radian = np.arccos(value)
    angle = radian2angle(radian)
    if minangle:
        angle = np.where(angle > 90, 180 - angle, angle)
    return angle


def line_test():
    '''
    angle: 56.789092174788685
    radian: 0.9911566376686096
    cosφ = u·v/|u||v|
    :return:
    '''
    # 两个向量
    point1 = np.array([1, 1, 0.5], dtype=np.float32)
    point2 = np.array([0.5, 0, 1], dtype=np.float32)
    point3 = np.array([1, 0, 0], dtype=np.float32)
    point4 = np.array([0.5, 0, 1], dtype=np.float32)
    angle = compute_point_angle(point1, point2, point3, point4)
    radian = angle2radian(angle)
    print("angle:", angle)
    print("radian:", radian)


if __name__ == '__main__':
    # 初始化数据
    # point1 = [2, 3, 1]
    # point2 = [4, 1, 2]
    # point3 = [6, 3, 7]
    # point4 = [-5, -4, 8]
    # # 计算点到面的距离
    # d1 = point2area_distance(point1, point2, point3, point4)  # s=8.647058823529413
    # print("点到面的距离s: " + str(d1))
    line_test()
