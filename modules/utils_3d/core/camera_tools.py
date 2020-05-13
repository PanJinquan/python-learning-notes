# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : camera_tools.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-22 16:15:56
"""
import cv2
import numpy as np


def rotation_vector2mat(om):
    '''
    https://blog.csdn.net/qq_40475529/article/details/89409303
    将Rotation vector旋转向量转换为旋转矩阵
    :param om:
    :return:
    '''
    R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
    return R


def rotation_mat2vector(R):
    '''
    将旋转矩阵转换为Rotation vector旋转向量
    :param R:
    :return:
    '''
    om = cv2.Rodrigues(R)[0]
    om = om.reshape(-1)
    return om


class Camera(object):
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)

    def convert_point3d_2d_list(self, xyz_coords_list, depth_scale=1.0):
        uv_coords_list = []
        z_coords_list = []
        for xyz_coords in xyz_coords_list:
            if xyz_coords is None:
                uv_coords_list.append(None)
            else:
                uv_coords, z_coords = self.convert_point3d_2d(xyz_coords, depth_scale)
                uv_coords_list.append(uv_coords)
                z_coords_list.append(z_coords)
        return uv_coords_list, z_coords_list

    def convert_point3d_2d(self, xyz_coords, depth_scale=1.0):
        '''
        see @convert_point3d_2d_v2
        :param xyz_coords:
        :param depth_scale:
        :return:
        '''
        """ Projects a (x, y, z) tuple of world coords into the image frame. """
        xyz_coords = np.reshape(xyz_coords, [-1, 3])
        z_coords = xyz_coords[:, 2] / depth_scale
        uv_coords = np.matmul(xyz_coords, np.transpose(self.camera_intrinsic, [1, 0]))
        return self._from_hom(uv_coords), z_coords

    def convert_point3d_2d_v2(self, xyz_coords, depth_scale):
        '''
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
        将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
        u = X * fx / Z + cx
        v = Y * fy / Z + cy
        D(v,u) = Z / Alpha
        :param point_2d:
        :param depth_img:
        :param intrinsic:
        :param depth_scale:
        :return:
        '''
        if len(xyz_coords.shape) == 1:
            xyz_coords = xyz_coords.reshape(1, 3)
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]
        # Back project the 2D points to 3D coor
        point_num = len(xyz_coords)
        point_2d = np.zeros((point_num, 2), dtype=np.float32)  # [25, 3] Note: Total 25 joints
        z_coords = np.zeros((point_num, 1), dtype=np.float32)  # [25, 3] Note: Total 25 joints
        for i in range(point_num):
            X, Y, Z = xyz_coords[i, 0], xyz_coords[i, 1], xyz_coords[i, 2]
            # point_2d[i, 0] = X * fx / Z + cx  # u
            # point_2d[i, 1] = Y * fy / Z + cy  # v
            point_2d[i, 0] = np.where(Z == 0, 0, X * fx / Z + cx)
            point_2d[i, 1] = np.where(Z == 0, 0, Y * fy / Z + cy)
            z_coords[i, 0] = Z / depth_scale
        return point_2d, z_coords

    def backproject_2d_3d(self, uv_coords, z_coords):
        '''
        see @backproject_2d_3d_v2 or backproject_2d_3d_depth
        Projects a (x, y, z) tuple of world coords into the world frame.
        :param uv_coords:
        :param z_coords:
        :return:
        '''
        uv_coords = np.reshape(uv_coords, [-1, 2])
        z_coords = np.reshape(z_coords, [-1, 1])

        uv_coords_h = self._to_hom(uv_coords)
        z_coords = np.reshape(z_coords, [-1, 1])
        xyz_coords = z_coords * np.matmul(uv_coords_h, np.transpose(self.camera_intrinsic_inv, [1, 0]))
        return xyz_coords

    def backproject_2d_3d_v2(self, uv_coords, z_coords):
        '''
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
        设像素坐标点P(u,v)，深度图D，单位m/米,将像素坐标P(u,v)从2D映射到3D坐标计算公式为：
        Z = Alpha*D(v,u) # 其中Alpha是缩放因子
        X =(u - cx) * Z / fx
        Y =(v - cy) * Z / fy
        :param uv_coords:
        :param depth_img:
        :param intrinsic:
        :param depth_scale:
        :return:
        '''
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]
        # Back project the 2D points to 3D coor
        point_num = len(uv_coords)
        point_3d = np.zeros((point_num, 3), dtype=np.float32)  # [25, 3] Note: Total 25 joints
        for i in range(point_num):
            u, v = uv_coords[i, 0], uv_coords[i, 1]
            point_3d[i, 2] = z_coords[i]  # Z coor
            point_3d[i, 0] = (u - cx) * point_3d[i, 2] / fx  # X coor
            point_3d[i, 1] = (v - cy) * point_3d[i, 2] / fy  # Y coor
        return point_3d

    def backproject_2d_3d_depth(self, point_2d, depth_img, depth_scale):
        '''
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
        设像素坐标点P(u,v)，深度图D，单位m/米,将像素坐标P(u,v)从2D映射到3D坐标计算公式为：
        Z = Alpha*D(v,u) # 其中Alpha是缩放因子
        X =(u - cx) * Z / fx
        Y =(v - cy) * Z / fy
        :param point_2d:
        :param depth_img:
        :param intrinsic:
        :param depth_scale:
        :return:
        '''
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]
        # Back project the 2D points to 3D coor
        point_num = len(point_2d)
        point_3d = np.zeros((point_num, 3), dtype=np.float32)  # [25, 3] Note: Total 25 joints
        for i in range(point_num):
            u, v = point_2d[i, 0], point_2d[i, 1]
            point_3d[i, 2] = depth_img[v, u] * depth_scale  # Z coor
            point_3d[i, 0] = (u - cx) * point_3d[i, 2] / fx  # X coor
            point_3d[i, 1] = (v - cy) * point_3d[i, 2] / fy  # Y coor
        return point_3d

    def backproject_from_depth(self, uv_coords, depth_img, depth_scale=1.0):
        u = uv_coords[:, 0].astype(np.int32)
        v = uv_coords[:, 1].astype(np.int32)
        z_coords = depth_img[v, u] * depth_scale
        xyz_coords = self.backproject_2d_3d(uv_coords, z_coords)
        # xyz_coords1 = self.backproject_2d_3d_v2(uv_coords, z_coords)
        # xyz_coords2 = self.backproject_2d_3d_depth(uv_coords, depth_img,depth_scale)
        return xyz_coords

    def convert_point2D_3D_list(self, point_2d_list, depth_img, depth_scale):
        point_3d_list = []
        for point_2d in point_2d_list:
            if point_2d is None:
                point_3d_list.append(None)
            else:
                point_3d = self.backproject_from_depth(point_2d, depth_img, depth_scale)
                point_3d_list.append(point_3d)
        return point_3d_list

    @staticmethod
    def get_meshgrid_vector(shape_hw, mask=None):
        """ Given an imageshape it outputs all coordinates as [N, dim] matrix. """
        if mask is None:
            H, W = np.meshgrid(range(0, shape_hw[0]), range(0, shape_hw[1]), indexing='ij')
            h_vec = np.reshape(H, [-1])
            w_vec = np.reshape(W, [-1])
            coords = np.stack([w_vec, h_vec], 1)
        else:
            H, W = np.meshgrid(range(0, shape_hw[0]), range(0, shape_hw[1]), indexing='ij')
            h_vec = np.reshape(H[mask], [-1])
            w_vec = np.reshape(W[mask], [-1])
            coords = np.stack([w_vec, h_vec], 1)
        return coords

    @staticmethod
    def _to_hom(coords):
        """ Turns the [N, D] coord matrix into homogeneous coordinates [N, D+1]. """
        coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)
        return coords_h

    @staticmethod
    def _from_hom(coords_h):
        """ Turns the homogeneous coordinates [N, D+1] into [N, D]. """
        coords = coords_h[:, :-1] / (coords_h[:, -1:] + 1e-10)
        return coords


if __name__ == "__main__":
    R = [[1.0000, 0.0003, -0.0076],
         [-0.0003, 1.0000, -0.0027],
         [0.0076, 0.0027, 1.0000]]

    R = np.asarray(R)
    print("Rotation vector", rotation_mat2vector(R))
