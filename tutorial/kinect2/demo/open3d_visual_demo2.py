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
from tools import open3d_tools


def triangle_pcd():
    '''
    定义三角形的点云
    :return:
    '''
    triangle_points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    lines = [[0, 1], [1, 2], [2, 0]]  # Right leg
    colors = [[0, 0, 1] for i in range(len(lines))]  # Default blue
    # 定义三角形的三个角点
    point_pcd = open3d.geometry.PointCloud()  # 定义点云
    point_pcd.points = open3d.Vector3dVector(triangle_points)

    # 定义三角形三条连接线
    line_pcd = open3d.LineSet()
    line_pcd.lines = open3d.Vector2iVector(lines)
    line_pcd.colors = open3d.Vector3dVector(colors)
    line_pcd.points = open3d.Vector3dVector(triangle_points)

    return line_pcd, point_pcd


if __name__ == "__main__":
    source_points = np.asarray([[0.0, 0.0, 0.0]])
    # 绘制open3d坐标系
    axis_pcd = open3d.geometry.create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # 在3D坐标上绘制点：坐标点[x,y,z]对应R，G，B颜色
    points = np.array([[1, 0, 0]], dtype=np.float64)
    colors = [[1, 0, 0]]

    # 方法1（非阻塞显示）
    vis = open3d.Visualizer()
    vis.create_window(window_name='Open3D_1', width=600, height=600, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 10  # 设置点的大小
    # 先把点云对象添加给Visualizer
    vis.add_geometry(axis_pcd)

    line_pcd, point_pcd = triangle_pcd()
    vis.add_geometry(line_pcd)
    vis.add_geometry(point_pcd)

    while True:
        # 主机减少点云的大小
        points = np.asarray(line_pcd.points) - [0.001, 0.001, 0.001]
        line_pcd.points = open3d.utility.Vector3dVector(points)
        point_pcd.points = open3d.utility.Vector3dVector(points)
        d = open3d_tools.compute_point2area_distance(np.asarray(point_pcd.points), source_points)
        print("d:{}".format(d))
        # update_renderer显示当前的数据
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        cv2.waitKey(100)
