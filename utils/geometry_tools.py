# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : geometry_tools.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-11 09:23:21
"""
# -*- coding: utf-8 -*-

import numpy as np
import copy
import cv2


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


def define_line(point1, point2):
    '''
    y-y1=k(x-x1),k=(y2-y1)/(x2-x1)=>
    kx-y+(y1-kx1)=0 <=> Ax+By+C=0
    => A=K=(y2-y1)/(x2-x1)
    => B=-1
    => C=(y1-kx1)
    :param point1:
    :param point2:
    :return:
    '''
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    A = (y2 - y1) / (x2 - x1)  # K
    B = -1
    C = y1 - A * x1
    return A, B, C


def point2line_distance(point1, point2, target_point):
    '''
    :param point1: line point1
    :param point2: line point2
    :param target_point: target_point
    :return:
    '''
    A, B, C = define_line(point1, point2)
    mod_d = A * target_point[0] + B * target_point[1] + C
    mod_sqrt = np.sqrt(np.sum(np.square([A, B])))
    d = abs(mod_d) / mod_sqrt
    return d


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
        # angle = np.where(angle > 90, 180 - angle, angle)
        angle = angle if angle < 90 else 180 - angle
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
    point1 = [5, 0]
    point2 = [0, 6]
    # r = define_line(point1, point2)
