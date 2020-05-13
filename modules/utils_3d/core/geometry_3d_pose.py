# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : geometry.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-11 10:04:34
"""
import cv2
import os
import numpy as np
from utils import image_processing
from modules.utils_3d import open3d_tools
from modules.utils_3d.core import rgbd_data
from modules.utils_3d.core import camera_tools, posture
import open3d as open3d
from tutorial.kinect2.config.joint_config import joint_config


class Geometry3DPose():
    def __init__(self, camera_conf):
        self.camera_conf = camera_conf
        self.flip_transform = [[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]]
        # Create Open3D Visualizer
        self.vis = open3d.Visualizer()
        self.vis.create_window('Open3D_1', width=self.camera_conf.depth_width, height=self.camera_conf.depth_height,
                               left=10, top=10)
        self.vis.get_render_option().point_size = 3

        # 定义图像点云
        self.image_pcd = open3d.PointCloud()
        # 定义原点
        self.origin_point = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # 定义关节点点云连接线：24关节点连接线
        self.bone_line_pcd = open3d_tools.create_line_set_bones(
            np.zeros((joint_config.joint_count, 3), dtype=np.float32),
            joint_line=joint_config.joint_lines)
        self.cam = camera_tools.Camera(self.camera_conf.camera_intrinsic)

    def show_image_pcd(self, isshow=True):
        if isshow:
            self.vis.add_geometry(self.image_pcd)  # 添加图像点云到Visualizer

    def show_origin_pcd(self, isshow=True):
        self.origin_point.transform(self.camera_conf.flip_transform)
        if isshow:
            self.vis.add_geometry(self.origin_point)  # 添加原始点到Visualizer

    def show_bone_line_pcd(self, isshow=True):
        if isshow:
            self.vis.add_geometry(self.bone_line_pcd)  # 添加24个关节点连接线到Visualizer

    def show_desktop_pcd(self, isshow=True):
        self.desktop_3d_point = self.define_triangle()
        # self.triangle_points_2d, triangle_depth = open3d_tools.convert_point3D_2D(self.desktop_3d_point,
        #                                                                           self.camera_conf.camera_intrinsic,
        #                                                                           self.camera_conf.depth_scale)
        self.triangle_points_2d, triangle_depth2 = self.cam.convert_point3d_2d(self.desktop_3d_point,
                                                                               self.camera_conf.depth_scale)

        lines = [[0, 1], [1, 2], [2, 0]]  # Right leg
        colors = [[0, 0, 1] for i in range(len(lines))]  # Default blue
        # 定义三角形的三个角点
        point_pcd = open3d.geometry.PointCloud()  # 定义点云
        point_pcd.points = open3d.Vector3dVector(self.desktop_3d_point)
        point_pcd.transform(self.camera_conf.flip_transform)

        # 定义三角形三条连接线
        desktop_line_pcd = open3d.LineSet()
        desktop_line_pcd.lines = open3d.Vector2iVector(lines)
        desktop_line_pcd.colors = open3d.Vector3dVector(colors)
        desktop_line_pcd.points = open3d.Vector3dVector(self.desktop_3d_point)
        desktop_line_pcd.transform(self.camera_conf.flip_transform)

        if isshow:
            self.vis.add_geometry(desktop_line_pcd)
            self.vis.add_geometry(point_pcd)

    def define_triangle(self, ):
        '''
        定义三角形的点云
        (x,y)=(269,420),data=602,point3d:[[0.01446956 0.35183427 0.602     ]]
        (x,y)=(210,398),data=658,point3d:[[-0.08981247  0.3451763   0.658     ]]
        (x,y)=(175,416),data=615,point3d:[[-0.14250912  0.35273877  0.615     ]]
        :return:
        '''
        # triangle_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        # triangle_points_3d = np.array([[0.01446956, 0.35183427, 0.602],
        #                                [-0.08981247, 0.3451763, 0.658],
        #                                [-0.14250912, 0.35273877, 0.615]], dtype=np.float32)
        #
        # triangle_points_3d = np.array([[-0.0123947, 0.14445746, 0.465],
        #                                [0.0120081, 0.15561016, 0.444],
        #                                [0.04850774, 0.14804208, 0.466]], dtype=np.float32)

        triangle_points_3d = np.array([[-0.00185865, 0.215392, 0.56],
                                       [0.01281946, 0.17611486, 0.474],
                                       [0.02140762, 0.16130832, 0.406]], dtype=np.float32)

        return triangle_points_3d

    def show(self, align_color_img, depth_img, body_joint=None, joint_type="2D", body_rects=None):
        '''

        :param align_color_img: <class 'tuple'>: (424, 512, 3),uint8
        :param depth_img: <class 'tuple'>: (424, 512),uint16
        :param body_joint2D:
        :return:
        '''
        self.image_pcd.points, self.image_pcd.colors = open3d_tools.create_color_point_cloud(align_color_img,
                                                                                             depth_img,
                                                                                             self.camera_conf.depth_scale,
                                                                                             self.camera_conf.clipping_distance_in_meters,
                                                                                             self.camera_conf.camera_intrinsic)
        self.image_pcd.transform(self.flip_transform)
        # pointcloud = pcl_tools.open3d2DepthColor2Cloud(np.asarray(self.image_pcd.points), np.asarray(self.image_pcd.colors))
        # pcl_tools.view_cloud(pointcloud)  # 显示点云
        if joint_type == "2D" and body_joint:
            body_joint2D = body_joint
            # body_joint3D = utils.map_body_joint2D_3D(body_joint, depth_img, config.intrinsic, config.depth_scale)
            # body_joint3D = open3d_tools.convert_point2D_3D_list(body_joint,depth_img,self.camera_conf.camera_intrinsic, self.camera_conf.depth_scale)
            body_joint3D = self.cam.convert_point2D_3D_list(body_joint2D, depth_img, self.camera_conf.depth_scale)
            # body_joint2D, _ = self.cam.convert_point3d_2d_list(body_joint3D, self.camera_conf.depth_scale)
            joint3D, orientation = open3d_tools.get_single_joint3D_orientation(body_joint3D,
                                                                               body_orientation=None,
                                                                               joint_count=joint_config.joint_count)
        elif joint_type == "3D" and body_joint:
            body_joint3D = body_joint
            body_joint2D, _ = self.cam.convert_point3d_2d_list(body_joint3D, self.camera_conf.depth_scale)
            joint3D, orientation = open3d_tools.get_single_joint3D_orientation(body_joint3D,
                                                                               body_orientation=None,
                                                                               joint_count=joint_config.joint_count)
        if body_joint:
            self.set_data(align_color_img, depth_img, joint3D)

        if body_joint:
            joint2D, _ = self.cam.convert_point3d_2d(joint3D)
            self.bone_line_pcd.points = open3d.Vector3dVector(joint3D)
            self.bone_line_pcd.transform(self.camera_conf.flip_transform)
            # self.draw_geometries()
            # Display 2D images using OpenCV
            # align_color_img = open3d_tools.draw_joint2D_in_image(body_joint2D, align_color_img,self.camera_conf.joint_lines)
            align_color_img = image_processing.draw_key_point_in_image(align_color_img, body_joint2D,
                                                                       joint_config.joint_lines)

            # depth_colormap = utils.draw_key_point_in_image(body_joint, depth_colormap)
            # d = geometry_tools.compute_joint3D_distance(joint3D, index=JointType_Head)
            # d = open3d_tools.compute_point2area_distance(self.desktop_3d_point, joint3D[joint_head])
            d = open3d_tools.compute_point2point_distance(self.desktop_3d_point, joint3D[joint_config.joint_head])
            angle = posture.get_current_status(joint3D, joint2D)
            # points_2d, _ = open3d_tools.convert_point3D_2D(joint3D[joint_head], self.camera_conf.camera_intrinsic,self.camera_conf.depth_scale)
            points_2d, _ = self.cam.convert_point3d_2d(joint3D[joint_config.joint_head], self.camera_conf.depth_scale)
            # print("points:{},distance:{}".format(points_2d, d))
            info = "{:.5f}".format(d)
            align_color_img = image_processing.draw_points_text(align_color_img, points=points_2d, texts=[info],
                                                                color=None)
            info = "head_angle:{:.5f}".format(angle)
            align_color_img = image_processing.draw_points_text(align_color_img, points=[[10, 20]], texts=[info],
                                                                color=None)
        if body_rects:
            align_color_img = image_processing.show_image_rects('align_color_img', align_color_img, body_rects,
                                                                waitKey=1)
        self.__update(depth_img, align_color_img)

    def __update(self, depth_img, align_color_img):
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

        image_processing.addMouseCallback("depth_colormap", param=depth_img, callbackFunc=self.default_callbackFunc)
        image_processing.addMouseCallback("align_color_img", param=depth_img, callbackFunc=self.default_callbackFunc)
        align_color_img = image_processing.draw_point_line(align_color_img, self.triangle_points_2d.tolist(),
                                                           color=(0, 0, 255), pointline="auto")
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_img, alpha=255 / self.camera_conf.clipping_distance),
            cv2.COLORMAP_JET)
        cv2.imshow('depth_img', depth_img)  # (424, 512)
        cv2.imshow('depth_colormap', depth_colormap)  # (424, 512)
        cv2.imshow("align_color_img", align_color_img)
        cv2.waitKey(30)

    def default_callbackFunc(self, event, x, y, flags, depth_img):
        '''
        (x,y)=(210,276),data=926,point3d:[[-0.1643461   0.41402617  0.837     ]]

        :param event:
        :param x:
        :param y:
        :param flags:
        :param depth_img:
        :return:
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            point2d = np.asarray([[x, y]])
            # point3d = open3d_tools.convert_point2D_3D(point2d,
            #                                           depth_img,
            #                                           self.camera_conf.intrinsic,
            #                                           self.camera_conf.depth_scale)
            point3d = self.cam.backproject_2d_3d_depth(point2d, depth_img, self.camera_conf.depth_scale)
            print("(x,y)=({},{}),data={},point3d:{}".format(x, y, depth_img[y][x], point3d))

    def close(self):
        self.vis.destroy_window()
        cv2.destroyAllWindows()

    def save_data(self, save_dir, flag):
        from utils import file_processing
        align_color_img, depth_img, joint3D = self.data
        align_color_img_path = os.path.join(save_dir, "image", "image_{}.png".format(str(flag)))
        joint_path = os.path.join(save_dir, "joint", "joint_{}.npy".format(str(flag)))
        file_processing.create_file_path(align_color_img_path)
        file_processing.create_file_path(joint_path)
        cv2.imwrite(align_color_img_path, align_color_img)
        np.save(joint_path, joint3D)

    def load_data(self, data_dir, flag):
        align_color_img_path = os.path.join(data_dir, "image", "image_{}.png".format(str(flag)))
        joint_path = os.path.join(data_dir, "joint", "joint_{}.npy".format(str(flag)))
        align_color_img = cv2.imread(align_color_img_path)
        joint3D = np.load(joint_path)
        return align_color_img, joint3D

    def set_data(self, align_color_img, depth_img, joint3D):
        self.data = align_color_img, depth_img, joint3D


if __name__ == "__main__":
    from tutorial.kinect2.config import kinect_config as camera_configs

    # from config import tum_config as camera_configs
    # from config import rgbd_pose_config as camera_configs
    # from config import ov0575_config as camera_configs

    save_dir = "E:/git/python-learning-notes/tutorial/kinect2/dataset/data01"
    align_color_img, depth_img, body_joint = rgbd_data.load_body_joint_data(save_dir, count=40)
    # save_dir = "../data/depth"
    # align_color_img, depth_img = rgbd_data.load_color_depth(save_dir, count=1)
    # color_path = "../data/snapshot3/left_2.png"
    # color_path = "../data/snapshot3/left_rectified_2.png"
    # depth_path = "../data/snapshot3/depth_2.png"
    # color_path = "../data/freiburg/1305031453.359684.png"
    # depth_path = "../data/freiburg/1305031453.374112.png"
    # color_path = "../data/color.png"
    # depth_path = "../data/depth.png"
    # color_path = "../data/ov0575/left_rectified_15.png"
    # depth_path = "../data/ov0575/depth_15.png"
    # align_color_img, depth_img = rgbd_data.read_color_depth(color_path, depth_path)
    # align_color_img=image_processing.resize_image(align_color_img,resize_width=640,resize_height=480)
    # depth_img=image_processing.resize_image(depth_img,resize_width=640,resize_height=480)
    print("align_color_img:{}, depth_img:{}".format(align_color_img.shape, depth_img.shape))
    g = Geometry3DPose(camera_configs)
    g.show_origin_pcd(True)
    g.show_bone_line_pcd(True)
    g.show_image_pcd(False)
    g.show_desktop_pcd(False)
    # depth_img = cv2.cvtColor(depth_img, cv2.COLOR_RGBA2GRAY)
    # image_processing.cv_show_image("depth_img",depth_img)
    # image_processing.cv_show_image("image",align_color_img)
    save_dir = "F:/X2/Pose/dataset/kitnet_data"
    while True:
        g.show(align_color_img, depth_img, body_joint)
        g.save_data(save_dir, flag=1)
        # g.show(align_color_img, depth_img, body_joint=None)
