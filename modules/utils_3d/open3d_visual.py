# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : geometry.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-11 10:04:34
"""
import cv2
import numpy as np
from utils import image_processing
from modules.utils_3d import open3d_tools
from modules.utils_3d.core import camera_tools
import open3d as open3d


class Open3DVisual(object):
    def __init__(self, camera_intrinsic, depth_width, depth_height, depth_scale=0.001, clipping_distance_in_meters=1.5):
        self.flip_transform = [[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]]
        self.depth_scale = depth_scale
        self.clipping_distance_in_meters = clipping_distance_in_meters
        self.camera_intrinsic = camera_intrinsic

        # Create Open3D Visualizer
        self.vis = open3d.Visualizer()
        self.vis.create_window('Open3D_1', width=depth_width, height=depth_height,
                               left=10, top=10)
        self.vis.get_render_option().point_size = 3

        # 定义图像点云
        self.image_pcd = open3d.PointCloud()
        # 定义原点
        self.origin_point = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
        self.cam = camera_tools.Camera(self.camera_intrinsic)

    def show_image_pcd(self, isshow=True):
        if isshow:
            self.vis.add_geometry(self.image_pcd)  # 添加图像点云到Visualizer

    def show_origin_pcd(self, isshow=True):
        self.origin_point.transform(self.flip_transform)
        if isshow:
            self.vis.add_geometry(self.origin_point)  # 添加原始点到Visualizer

    def show(self, color_image, depth_image):
        """
        :param color_image:
        :param depth_image:
        :return:
        """
        self.image_pcd.points, self.image_pcd.colors = open3d_tools.create_color_point_cloud(color_image,
                                                                                             depth_image,
                                                                                             self.depth_scale,
                                                                                             self.clipping_distance_in_meters,
                                                                                             self.camera_intrinsic)
        self.image_pcd.transform(self.flip_transform)

        self.__update(color_image, depth_image)

    def __update(self, color_image, depth_image):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()
