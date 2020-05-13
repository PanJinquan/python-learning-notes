# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : posture.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 11:55:11
"""
from modules.utils_3d import open3d_tools
from tutorial.kinect2.config.joint_config import joint_config


def get_current_status(joint3D, joint2D):
    '''

    :param joint3D:
    :param joint2D:
    :return:
    '''
    origin = [0, 0, 0]
    X = open3d_tools.gen_vector(origin, point2=[1, 0, 0])
    Y = open3d_tools.gen_vector(origin, point2=[0, 1, 0])
    Z = open3d_tools.gen_vector(origin, point2=[0, 0, 1])
    joint_head = joint3D[joint_config.joint_head]
    joint_nect = joint3D[joint_config.joint_nect]
    joint_body = (joint3D[joint_config.joint_shoulder_left]+joint3D[joint_config.joint_shoulder_right])/2
    head_vector = open3d_tools.gen_vector(point1=joint_body, point2=joint_head)
    angle = define_head_pose(head_vector, base_axis=Z)
    return angle


def define_head_pose(head_vector, base_axis):
    angle = open3d_tools.compute_vector_angle(head_vector, base_axis, minangle=True)
    return angle


if __name__ == "__main__":
    save_dir = "../data/data01"
