# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : open3d_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-10 09:49:27
"""
import open3d
import numpy as np
import cv2

# 绘制open3d坐标系
axis_pcd = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
# 在3D坐标上绘制点：坐标点[x,y,z]对应R，G，B颜色
points = np.array([[1, 0, 0]], dtype=np.float64)
colors = [[1, 0, 0]]

test_pcd = open3d.geometry.PointCloud()  # 定义点云
test_pcd2 = open3d.geometry.PointCloud()  # 定义点云

# 方法1（非阻塞显示）
vis = open3d.Visualizer()
vis.create_window(window_name='Open3D_1', width=600, height=600, left=10, top=10, visible=True)
vis.get_render_option().point_size = 10  # 设置点的大小
# 先把点云对象添加给Visualizer
vis.add_geometry(axis_pcd)
vis.add_geometry(test_pcd)
source = open3d.geometry.PointCloud()  # 定义点云
target = open3d.geometry.PointCloud()  # 定义点云
source.points = open3d.Vector3dVector(np.asarray([[0.0, 0.0, 0.0]]))
while True:
    # 给点云添加显示的数据
    points -= [0.001, 0, 0]
    test_pcd.points = open3d.utility.Vector3dVector(points)  # 定义点云坐标位置
    test_pcd.colors = open3d.Vector3dVector(colors)  # 定义点云的颜色
    # update_renderer显示当前的数据
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    # d = open3d.geometry.compute_point_cloud_to_point_cloud_distance(source, target)
    d = open3d.geometry.compute_point_cloud_to_point_cloud_distance(source, test_pcd)
    print("d={}".format(d))
    cv2.waitKey(100)

# 方法2（阻塞显示）：调用draw_geometries直接把需要显示点云数据
test_pcd.points = open3d.utility.Vector3dVector(points)  # 定义点云坐标位置
test_pcd.colors = open3d.Vector3dVector(colors)  # 定义点云的颜色
open3d.visualization.draw_geometries([test_pcd] + [axis_pcd], window_name="Open3D_2")
