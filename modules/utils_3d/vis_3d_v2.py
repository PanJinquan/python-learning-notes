# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-03 12:10:33
# --------------------------------------------------------
"""

import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import image_processing


# plt.switch_backend('TkAgg')


def vis_2d_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_keypoints(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    x_r = np.array([0, input_shape[1]], dtype=np.float32)
    y_r = np.array([0, input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    # ax.set_xlim([0,cfg.input_shape[1]])
    # ax.set_ylim([0,1])
    # ax.set_zlim([-cfg.input_shape[0],0])
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def vis_3d_skeleton_image(image, coords_3d, skeleton):
    # Matplotlib interprets the Z axis as vertical, but our pose
    # has Y as the vertical axis.
    # Therefore we do a 90 degree rotation around the horizontal (X) axis
    # draw 2D keypoints
    nums_joints = len(coords_3d)
    coords_2d = np.zeros((3, nums_joints))
    coords_2d[:2, :] = coords_3d[:, :2].transpose(1, 0) / output_shape[0] * input_shape[0]
    coords_2d[2, :] = 1
    image = vis_2d_keypoints(image, coords_2d, skeleton)

    coords2 = coords_3d.copy()
    coords_3d[:, 1], coords_3d[:, 2] = coords2[:, 2], -coords2[:, 1]

    fig = plt.figure(figsize=(10, 5))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.set_title('Input')
    image_ax.imshow(image)

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.set_title('Prediction')

    # range_ = 50
    # pose_ax.set_xlim3d(-range_, range_)
    # pose_ax.set_ylim3d(-range_, range_)
    # pose_ax.set_zlim3d(-range_, range_)
    x = coords_3d[:, 0]
    y = coords_3d[:, 1]
    z = coords_3d[:, 2]
    range_max = max(x)
    range_min = min(x)
    range_max = range_max + (range_max - range_min) * 0.2
    range_min = range_min - (range_max - range_min) * 0.2
    y = y + (range_min - min(y))
    z = z + (range_min - min(z))
    coords_3d[:, 1] = y
    coords_3d[:, 2] = z
    pose_ax.set_xlim3d(range_min, range_max)
    pose_ax.set_ylim3d(range_min, range_max)
    pose_ax.set_zlim3d(range_min, range_max)
    pose_ax.set_xlabel('X Label')
    pose_ax.set_ylabel('Z Label')
    pose_ax.set_zlabel('Y Label')

    for i_start, i_end in skeleton:
        pose_ax.plot(*zip(coords_3d[i_start], coords_3d[i_end]), marker='o', markersize=2)
    pose_ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], s=2)
    fig.tight_layout()
    image = image_processing.fig2data(fig)
    # plt.show()
    return image


if __name__ == "__main__":
    image_path = "/media/dm/dm/X2/Pose/3DPose/3DMPPE_POSENET_RELEASE/test.png"
    tmpimg = cv2.imread(image_path)
    input_shape = (256, 256)
    output_shape = (input_shape[0] // 4, input_shape[1] // 4)
    coords_3d = np.asarray([[32.89645, 29.245146, 32.00004],
                            [36.22007, 28.8877, 33.602837],
                            [36.28984, 40.997894, 36.545135],
                            [37.125984, 52.0086, 39.707615],
                            [29.498604, 29.585405, 30.443153],
                            [29.117733, 41.588974, 32.77747],
                            [28.673431, 52.98233, 33.7291],
                            [33.07794, 22.855503, 29.107742],
                            [33.014565, 15.737941, 26.741709],
                            [32.537285, 13.542124, 28.76703],
                            [33.372738, 10.819509, 26.570528],
                            [29.10359, 16.715046, 24.979181],
                            [21.166656, 18.585464, 23.381084],
                            [15.262758, 18.606749, 27.925941],
                            [37.048244, 16.550882, 27.829075],
                            [43.673508, 18.658102, 31.728264],
                            [48.55796, 19.544056, 36.622536],
                            [33.07514, 16.656559, 26.200886]], dtype=np.float32)

    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    nums_joints = len(coords_3d)
    joint_vis = np.zeros(shape=(nums_joints, 1)) + 1
    vis_3d_keypoints(coords_3d, joint_vis, skeleton, filename=None)

    # tmpkps = np.zeros((3, nums_joints))
    # tmpkps[:2, :] = kpt_3d[:, :2].transpose(1, 0) / output_shape[0] * input_shape[0]
    # tmpkps[2, :] = 1
    # tmpimg1 = vis_2d_keypoints(tmpimg, tmpkps, skeleton)
    tmpimg2 = vis_3d_skeleton_image(image=tmpimg, coords_3d=coords_3d, skeleton=skeleton)
    # cv2.imwrite(filename + '_output.jpg', tmpimg)
    cv2.imshow("Det2", tmpimg2)
    cv2.waitKey(0)
