# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : rgbd_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-10 19:48:07
"""
import os
import cv2
import numpy as np


def save_body_joint_data(save_dir, depth_img, align_color_img, body_joint2D, count=0):
    '''
    :param save_dir:
    :param depth_img:
    :param align_color_img:
    :param body_joint2D:
    :param count:
    :return:
    '''
    for i, joint2D in enumerate(body_joint2D):
        if joint2D is None:
            continue
        np.save(os.path.join(save_dir, "depth_img_{}.npy".format(count)), depth_img)
        np.save(os.path.join(save_dir, "align_color_img_{}.npy".format(count)), align_color_img)
        np.save(os.path.join(save_dir, "body_joint2D_{}.npy".format(count)), np.asarray(body_joint2D))
        # cv2.imwrite("depth_img_{}.png".format(count), depth_img)
        # cv2.imwrite("align_color_img_{}.png".format(count), align_color_img)
        break


def load_body_joint_data(save_dir, count=0):
    depth_img = np.load(os.path.join(save_dir, "depth_img_{}.npy".format(count)))
    align_color_img = np.load(os.path.join(save_dir, "align_color_img_{}.npy".format(count)))
    body_joint2D = np.load(os.path.join(save_dir, "body_joint2D_{}.npy".format(count)), allow_pickle=True)
    body_joint2D = body_joint2D.tolist()
    # depth_img = cv2.imread(os.path.join(save_dir, "depth_img.png"), cv2.IMREAD_UNCHANGED)
    # align_color_img = cv2.imread(os.path.join(save_dir, "align_color_img.png"), cv2.IMREAD_UNCHANGED)
    align_color_img = cv2.cvtColor(align_color_img, cv2.COLOR_RGBA2RGB)
    return align_color_img, depth_img, body_joint2D


def load_color_depth(data_dir, count=0):
    depth_img = cv2.imread(os.path.join(data_dir, "depth_{}.png".format(count)), cv2.IMREAD_UNCHANGED)
    align_color_img = cv2.imread(os.path.join(data_dir, "color_{}.png".format(count)), cv2.IMREAD_UNCHANGED)
    return align_color_img, depth_img


def read_color_depth(color_path, depth_path):
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    align_color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
    return align_color_img, depth_img
