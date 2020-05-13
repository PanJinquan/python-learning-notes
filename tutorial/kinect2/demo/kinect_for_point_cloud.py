# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : kinect_for_rgbd.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 09:43:27
"""
########################################################################
### Sample program to stream
### Coloured point cloud, joint and joint orientation in 3D using Open3D
########################################################################
import cv2
import numpy as np
import utils_PyKinectV2 as utils
import open3d as open3d
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
from tools import image_processing


class Kinect2PointCloud():
    def __init__(self):
        # Kinect runtime object
        self.joint_count = PyKinectV2.JointType_Count  # 25
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body |
                                                      PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth)

        self.depth_width, self.depth_height = self.kinect.depth_frame_desc.Width, self.kinect.depth_frame_desc.Height  # Default: 512, 424
        self.color_width, self.color_height = self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height  # Default: 1920, 1080

        # User defined variables
        self.depth_scale = 0.001  # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
        # depth_scale                 = 1.0 # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
        self.clipping_distance_in_meters = 1.5  # Set the maximum distance to display the point cloud data
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale  # Convert dist in mm to unit
        # Hardcode the camera intrinsic parameters for backprojection
        # width=depth_width; height=depth_height; ppx=258.981; ppy=208.796; fx=367.033; fy=367.033 # Hardcode the camera intrinsic parameters for backprojection
        ppx = 260.166
        ppy = 205.197
        fx = 367.535
        fy = 367.535
        # Open3D visualisation
        self.intrinsic = open3d.PinholeCameraIntrinsic(self.depth_width, self.depth_height, fx, fy, ppx, ppy)
        # To convert [x,y,z] -> [x.-y,-z]
        self.flip_transform = [[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]]
        # 定义图像点云
        self.image_pcd = open3d.PointCloud()
        # self.joint_pcd = open3d.PointCloud()
        # 定义原点
        self.origin_point = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # 定义关节点坐标点云
        self.axis_pcd = []
        for i in range(self.joint_count):  # 25 axes for 25 joints
            # XYZ axis length of 0.1 m
            pre_axis = open3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.axis_pcd.append(pre_axis)
        # 定义关节点点云连接线：24关节点连接线
        self.bone_line_pcd = utils.create_line_set_bones(np.zeros((24, 3), dtype=np.float32))

    def start_capture(self):
        show_origin = True  # 原点必须显示
        show_axis = True  # 显示关节朝向
        show_image = False  # 显示图像点云
        show_bone_line = True  # 显示关节点连接线
        ##
        # Create Open3D Visualizer
        self.vis = open3d.Visualizer()
        self.vis.create_window('Open3D_1', width=self.depth_width, height=self.depth_height, left=10, top=10)
        self.vis.get_render_option().point_size = 3
        if show_origin:
            self.vis.add_geometry(self.origin_point)  # 添加原始点到Visualizer
        if show_image:
            self.vis.add_geometry(self.image_pcd)  # 添加图像点云到Visualizer
        if show_bone_line:
            self.vis.add_geometry(self.bone_line_pcd)  # 添加24个关节点连接线到Visualizer
        if show_axis:
            for i in range(self.joint_count):  # 添加25个关节点的朝向到Visualizer
                self.vis.add_geometry(self.axis_pcd[i])
        # self.vis.add_geometry(self.joint_pcd)    # 添加25个关节点到Visualizer

        while True:
            # Get images from camera
            if self.kinect.has_new_body_frame() and self.kinect.has_new_color_frame() and self.kinect.has_new_depth_frame():
                body_frame = self.kinect.get_last_body_frame()
                color_frame = self.kinect.get_last_color_frame()
                depth_frame = self.kinect.get_last_depth_frame()
                # Reshape from 1D frame to 2D image
                color_img = color_frame.reshape(((self.color_height, self.color_width, 4))).astype(np.uint8)
                depth_img = depth_frame.reshape(((self.depth_height, self.depth_width))).astype(np.uint16)
                align_color_img = utils.get_align_color_image(self.kinect, color_img)

                # Useful functions in utils_PyKinectV2.py
                self.image_pcd.points, self.image_pcd.colors = utils.create_color_point_cloud(align_color_img,
                                                                                              depth_img,
                                                                                              self.depth_scale,
                                                                                              self.clipping_distance_in_meters,
                                                                                              self.intrinsic)
                self.image_pcd.transform(self.flip_transform)
                # joint3D, orientation = utils.get_single_joint3D_and_orientation(self.kinect, body_frame, depth_img,
                #                                                                 self.intrinsic,
                #                                                                 self.depth_scale)
                body_joint2D, body_orientation = utils.get_body_joint2D(body_frame, self.kinect,
                                                                        map_space="depth_space")
                body_joint3D = utils.map_body_joint2D_3D(body_joint2D, depth_img, self.intrinsic, self.depth_scale)
                joint3D, orientation = utils.get_single_joint3D_orientation(body_joint3D, body_orientation)
                self.bone_line_pcd.points = open3d.Vector3dVector(joint3D)
                self.bone_line_pcd.transform(self.flip_transform)
                if show_axis:
                    self.add_orientation(joint3D, orientation)
                self.vis.update_geometry()
                self.vis.poll_events()
                self.vis.update_renderer()
                # self.draw_geometries()
                if show_axis:
                    self.inverse_transformation(joint3D, orientation)
                # Display 2D images using OpenCV
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=255 / self.clipping_distance),
                                                   cv2.COLORMAP_JET)
                image_processing.addMouseCallback("depth_img", param=depth_img)
                image_processing.addMouseCallback("align_color_img", param=align_color_img)
                cv2.imshow('depth_img', depth_colormap)  # (424, 512)
                cv2.imshow("align_color_img", align_color_img)

            key = cv2.waitKey(30)
            if key == 27:  # Press esc to break the loop
                break

    def add_orientation(self, joint3D, orientation):
        # Draw the orientation axes at each joint
        for i in range(self.joint_count):
            self.axis_pcd[i].transform(utils.transform_geometry_quaternion(joint3D[i, :], orientation[i, :]))
            self.axis_pcd[i].transform(self.flip_transform)

    def inverse_transformation(self, joint3D, orientation):
        # Need to inverse the transformation else it will be additive in every loop
        for i in range(self.joint_count):
            self.axis_pcd[i].transform(self.flip_transform)
            self.axis_pcd[i].transform(
                np.linalg.inv(np.array(utils.transform_geometry_quaternion(joint3D[i, :], orientation[i, :]))))

    def draw_geometries(self):
        show_pcd = []
        # show_pcd.append(self.image_pcd)
        # show_pcd += self.axis_pcd
        show_pcd.append(self.bone_line_pcd)
        open3d.visualization.draw_geometries(show_pcd, window_name='Open3D_2', width=1920, height=1080, left=50, top=50)

    def close(self):
        self.kinect.close()
        self.vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    kc = Kinect2PointCloud()
    kc.start_capture()
