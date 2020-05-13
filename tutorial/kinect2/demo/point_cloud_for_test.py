# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : test_for_point_cloud.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-10 20:03:28
"""

# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : posture.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 11:55:11
"""
import os
import cv2
import numpy as np
from tools import image_processing
import utils_PyKinectV2 as utils
from core import rgbd_data, posture
from config import kinect_config as config
import open3d as open3d
from pykinect2 import PyKinectV2

'''
i=0,(x,y)=(234,344),data=841
i=1,(x,y)=(232,251),data=943
i=2,(x,y)=(231,170),data=1033
i=3,(x,y)=(224,115),data=963
i=4,(x,y)=(168,211),data=1017
i=5,(x,y)=(141,268),data=919
i=6,(x,y)=(113,289),data=629
i=7,(x,y)=(126,296),data=622
i=8,(x,y)=(289,211),data=995
i=9,(x,y)=(322,283),data=985
i=10,(x,y)=(349,308),data=747
i=11,(x,y)=(365,322),data=631
i=12,(x,y)=(202,347),data=864
i=13,(x,y)=(0,0),data=0
i=14,(x,y)=(0,0),data=0
i=15,(x,y)=(0,0),data=0
i=16,(x,y)=(268,340),data=865
i=17,(x,y)=(0,0),data=0
i=18,(x,y)=(0,0),data=0
i=19,(x,y)=(0,0),data=0
i=20,(x,y)=(231,189),data=1036
i=21,(x,y)=(138,294),data=630
i=22,(x,y)=(144,279),data=676
i=23,(x,y)=(368,317),data=625
i=24,(x,y)=(388,311),data=720
'''


def show_depth(depth_img, align_color_img, body_joint2D):
    align_color_img2 = utils.draw_joint2D_in_image(body_joint2D, align_color_img)
    depth_img2 = utils.draw_joint2D_in_image(body_joint2D, depth_img)
    # Resize (1080, 1920, 4) into half (540, 960, 4)
    # Scale to display from 0 mm to 1500 mm
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img2, alpha=255 / 1500), cv2.COLORMAP_JET)
    # Scale from uint16 to uint8
    image_processing.addMouseCallback("align_color_img", align_color_img, callbackFunc=None)
    image_processing.addMouseCallback("depth", depth_img, callbackFunc=None)
    image_processing.cv_show_image('align_color_img', align_color_img2, type="bgr", waitKey=30)  # (424, 512)
    image_processing.cv_show_image('depth', depth_colormap, type="bgr")  # (424, 512)


def rgbd_test(depth_img, align_color_img, body_joint2D):
    show_origin = True  # 原点必须显示
    show_image = True  # 显示图像点云
    show_bone_line = True  # 显示关节点连接线

    flip_transform = [[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]]
    # 定义图像点云
    image_pcd = open3d.PointCloud()
    # self.joint_pcd = open3d.PointCloud()
    # 定义原点
    origin_point = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # 定义关节点点云连接线：24关节点连接线
    bone_line_pcd = utils.create_line_set_bones(np.zeros((24, 3), dtype=np.float32))

    # Create Open3D Visualizer
    vis = open3d.Visualizer()
    vis.create_window('Open3D_1', width=config.depth_width, height=config.depth_height, left=10, top=10)
    vis.get_render_option().point_size = 3
    if show_origin:
        vis.add_geometry(origin_point)  # 添加原始点到Visualizer
    if show_image:
        vis.add_geometry(image_pcd)  # 添加图像点云到Visualizer
    if show_bone_line:
        vis.add_geometry(bone_line_pcd)  # 添加24个关节点连接线到Visualizer
    while True:
        image_pcd.points, image_pcd.colors = utils.create_color_point_cloud(align_color_img,
                                                                            depth_img,
                                                                            config.depth_scale,
                                                                            config.clipping_distance_in_meters,
                                                                            config.intrinsic)
        image_pcd.transform(flip_transform)
        body_joint3D = utils.map_body_joint2D_3D(body_joint2D, depth_img, config.intrinsic, config.depth_scale)
        joint3D, orientation = utils.get_single_joint3D_orientation(body_joint3D, body_orientation=None)
        bone_line_pcd.points = open3d.Vector3dVector(joint3D)
        bone_line_pcd.transform(config.flip_transform)

        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        # self.draw_geometries()
        # Display 2D images using OpenCV
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=255 / config.clipping_distance),
                                           cv2.COLORMAP_JET)
        align_color_img = utils.draw_joint2D_in_image(body_joint2D, align_color_img)
        # depth_colormap = utils.draw_key_point_in_image(body_joint, depth_colormap)
        d = posture.compute_joint3D_distance(joint3D, index=PyKinectV2.JointType_Head)
        # print("d={}".format(d))
        image_processing.addMouseCallback("depth_colormap", param=depth_img)
        image_processing.addMouseCallback("align_color_img", param=depth_img)

        cv2.imshow('depth_img', depth_img)  # (424, 512)
        cv2.imshow('depth_colormap', depth_colormap)  # (424, 512)
        cv2.imshow("align_color_img", align_color_img)
        cv2.waitKey(30)


if __name__ == "__main__":
    save_dir = "../data/data01"
    depth_img, align_color_img, body_joint2D = rgbd_data.load_body_joint_data(save_dir, count=30)
    rgbd_test(depth_img, align_color_img, body_joint2D)
