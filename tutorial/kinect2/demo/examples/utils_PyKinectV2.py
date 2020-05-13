##############################################################
### Set of useful utilities function related to PyKinectV2 ###
##############################################################
import cv2
import ctypes
import copy

import numpy as np
import open3d
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


##########################
### Map color to depth ###
##########################
def get_align_color_image(kinect, color_img, color_height=1080, color_width=1920, depth_height=424, depth_width=512):
    CSP_Count = kinect._depth_frame_data_capacity  # 512*424
    CSP_type = _ColorSpacePoint * CSP_Count.value
    CSP = ctypes.cast(CSP_type(), ctypes.POINTER(_ColorSpacePoint))

    kinect._mapper.MapDepthFrameToColorSpace(kinect._depth_frame_data_capacity, kinect._depth_frame_data, CSP_Count,
                                             CSP)

    colorXYs = np.copy(
        np.ctypeslib.as_array(CSP, shape=(depth_height * depth_width,)))  # Convert ctype pointer to array
    colorXYs = colorXYs.view(np.float32).reshape(colorXYs.shape + (
        -1,))  # Convert struct array to regular numpy array https://stackoverflow.com/questions/5957380/convert-structured-array-to-regular-numpy-array
    colorXYs += 0.5
    colorXYs = colorXYs.reshape(depth_height, depth_width, 2).astype(np.int)
    colorXs = np.clip(colorXYs[:, :, 0], 0, color_width - 1)
    colorYs = np.clip(colorXYs[:, :, 1], 0, color_height - 1)

    align_color_img = np.zeros((depth_height, depth_width, 4), dtype=np.uint8)
    align_color_img[:, :] = color_img[colorYs, colorXs, :]

    return align_color_img


##################################
### Get the joints information ###
##################################
def get_single_joint(joints, jointPoints, jointType):
    jointState = joints[jointType].TrackingState;

    # Joint not tracked or not 'really' tracked
    if (jointState == PyKinectV2.TrackingState_NotTracked) or (jointState == PyKinectV2.TrackingState_Inferred):
        return np.zeros((1, 2), dtype=np.int32)  # Return zeros
    else:
        return np.array([jointPoints[jointType].x, jointPoints[jointType].y], dtype=np.int32)


def get_joint2D(joints, jointPoints):
    joint2D = np.zeros((PyKinectV2.JointType_Count, 2), dtype=np.int32)  # [25, 2] Note: Total 25 joints
    for i in range(PyKinectV2.JointType_Count):
        joint2D[i, :] = get_single_joint(joints, jointPoints, i)

    return joint2D


def get_joint3D(joints, jointPoints, depth_img, intrinsics, depth_scale):
    joint3D = np.zeros((PyKinectV2.JointType_Count, 3), dtype=np.float32)  # [25, 3] Note: Total 25 joints
    joint2D = get_joint2D(joints, jointPoints)

    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    # Back project the 2D points to 3D coor
    for i in range(PyKinectV2.JointType_Count):
        u, v = joint2D[i, 0], joint2D[i, 1]
        joint3D[i, 2] = depth_img[v, u] * depth_scale  # Z coor
        joint3D[i, 0] = (u - cx) * joint3D[i, 2] / fx  # X coor
        joint3D[i, 1] = (v - cy) * joint3D[i, 2] / fy  # Y coor

    return joint3D


def map_body_joint2D_3D(body_joint2D, depth_img, intrinsics, depth_scale):
    '''
    cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :return:
    '''
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    # Back project the 2D points to 3D coor
    body_joint3D = []
    joint3D = np.zeros((PyKinectV2.JointType_Count, 3), dtype=np.float32)  # [25, 3] Note: Total 25 joints
    for joint2D in body_joint2D:
        if joint2D is None:
            body_joint3D.append(None)
        else:
            for i in range(PyKinectV2.JointType_Count):
                u, v = joint2D[i, 0], joint2D[i, 1]
                joint3D[i, 2] = depth_img[v, u] * depth_scale  # Z coor
                joint3D[i, 0] = (u - cx) * joint3D[i, 2] / fx  # X coor
                joint3D[i, 1] = (v - cy) * joint3D[i, 2] / fy  # Y coor
            body_joint3D.append(joint3D)
    return body_joint3D


def get_joint_quaternions(orientations):
    quat = np.zeros((PyKinectV2.JointType_Count, 4), dtype=np.float32)  # [25, 4] Note: Total 25 joints
    for i in range(PyKinectV2.JointType_Count):
        quat[i, 0] = orientations[i].Orientation.w
        quat[i, 1] = orientations[i].Orientation.x
        quat[i, 2] = orientations[i].Orientation.y
        quat[i, 3] = orientations[i].Orientation.z

    return quat


######################
### Draw on OpenCV ###
######################
# Define the BGR color for 6 different bodies
colors_order = [(0, 0, 255),  # Red
                (0, 255, 0),  # Green
                (255, 0, 0),  # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (255, 255, 0)]  # Cyan


def draw_joint2D(img, j2D, color=(0, 0, 255)):  # Default red circles
    for i in range(j2D.shape[0]):  # Should loop 25 times
        cv2.circle(img, (j2D[i, 0], j2D[i, 1]), 5, color, -1)

    return img


def draw_bone2D(img, j2D, color=(0, 0, 255)):  # Default red lines
    # Define the kinematic tree where each of the 25 joints is associated to om parent joint
    k = [0, 0, 1, 2,  # Spine
         20, 4, 5, 6,  # Left arm
         20, 8, 9, 10,  # Right arm
         0, 12, 13, 14,  # Left leg
         0, 16, 17, 18,  # Right leg
         1,  # Spine
         7, 7,  # Left hand
         11, 11]  # Right hand

    for i in range(j2D.shape[0]):  # Should loop 25 times
        if j2D[k[i], 0] > 0 and j2D[k[i], 1] > 0 and j2D[i, 0] > 0 and j2D[i, 1] > 0:
            cv2.line(img, (j2D[k[i], 0], j2D[k[i], 1]), (j2D[i, 0], j2D[i, 1]), color)

    return img


def color_body_index(kinect, img):
    height, width = img.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(kinect.max_body_count):
        color_img[np.where(img == i)] = colors_order[i]

    return color_img


def draw_bodyframe2(body_frame, kinect, img, map_space="depth_space"):
    if body_frame is not None:
        for i in range(0, kinect.max_body_count):
            body = body_frame.bodies[i]
            if body.is_tracked:
                joints = body.joints
                if map_space == "depth_space":
                    joint_points = kinect.body_joints_to_depth_space(
                        joints)  # Convert joint coordinates to depth space
                elif map_space == "color_space":
                    joint_points = kinect.body_joints_to_color_space(joints)  # Convert joint coordinates to depth space
                joint2D = get_joint2D(joints, joint_points)  # Convert to numpy array format
                img = draw_joint2D(img, joint2D, colors_order[i])  # 先用“o”画关节点
                img = draw_bone2D(img, joint2D, colors_order[i])  # 再用“--”连接关节点

    return img


def draw_bodyframe(body_frame, kinect, img, map_space="depth_space"):
    body_joint2D = get_body_joint2D(body_frame, kinect, map_space)
    img = draw_joint2D_in_image(body_joint2D, img)
    return img


def draw_joint2D_in_image(body_joint2D, image):
    img = copy.deepcopy(image)
    for i, joint2D in enumerate(body_joint2D):
        if joint2D is None:
            continue
        img = draw_joint2D(img, joint2D, colors_order[i])  # 先用“o”画关节点
        img = draw_bone2D(img, joint2D, colors_order[i])  # 再用“--”连接关节点
    return img


def get_body_joint2D(body_frame, kinect, map_space="depth_space"):
    body_joint2D = []
    body_orientation = []
    if body_frame is not None:
        for i in range(0, kinect.max_body_count):
            body = body_frame.bodies[i]
            joint2D, orientation = get_joint2D_points(body, kinect, map_space)
            body_joint2D.append(joint2D)
            body_orientation.append(orientation)
    return body_joint2D, body_orientation


def get_joint2D_points(body, kinect, map_space="depth_space"):
    joint2D = None
    orientation = None
    if body.is_tracked:
        joints = body.joints
        if map_space == "depth_space":
            joint_points = kinect.body_joints_to_depth_space(joints)  # Convert joint coordinates to depth space
        elif map_space == "color_space":
            joint_points = kinect.body_joints_to_color_space(joints)  # Convert joint coordinates to depth space
        orientation = get_joint_quaternions(body.joint_orientations)
        joint2D = get_joint2D(joints, joint_points)  # Convert to numpy array format
    return joint2D, orientation


def get_joint_points(body, kinect, map_space="depth_space"):
    joint2D = None
    if body.is_tracked:
        joints = body.joints
        if map_space == "depth_space":
            joint_points = kinect.body_joints_to_depth_space(
                joints)  # Convert joint coordinates to depth space
        elif map_space == "color_space":
            joint_points = kinect.body_joints_to_color_space(joints)  # Convert joint coordinates to depth space
        joint2D = get_joint2D(joints, joint_points)  # Convert to numpy array format
    return joint2D


################################
### For Open3D visualisation ###
################################
def create_line_set_bones(joints):
    # Draw the 24 bones (lines) connecting 25 joints
    # The lines below is the kinematic tree that defines the connection between parent and child joints
    lines = [[0, 1], [1, 20], [20, 2], [2, 3],  # Spine
             [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],  # Left arm and hand
             [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],  # Right arm and hand
             [0, 12], [12, 13], [13, 14], [14, 15],  # Left leg
             [0, 16], [16, 17], [17, 18], [18, 19]]  # Right leg
    colors = [[0, 0, 1] for i in range(24)]  # Default blue
    line_set = open3d.LineSet()
    line_set.lines = open3d.Vector2iVector(lines)
    line_set.colors = open3d.Vector3dVector(colors)
    line_set.points = open3d.Vector3dVector(joints)

    return line_set


def create_color_point_cloud(align_color_img, depth_img,
                             depth_scale, clipping_distance_in_meters, intrinsic):
    rgbd_image = get_rgbd_image(align_color_img, depth_img, depth_scale, clipping_distance_in_meters)
    pcd = open3d.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic)
    # pcd = open3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, open3d.camera.PinholeCameraIntrinsic(
    #     open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Point cloud only without color
    # pcd = create_point_cloud_from_depth_image(
    #     Image(depth_img),
    #     intrinsic,
    #     depth_scale=1.0/depth_scale,
    #     depth_trunc=clipping_distance_in_meters)

    return pcd.points, pcd.colors


def get_rgbd_image(align_color_img, depth_img, depth_scale, clipping_distance_in_meters):
    align_color_img = align_color_img[:, :, 0:3]  # Only get the first three channel
    align_color_img = align_color_img[..., ::-1]  # Convert opencv BGR to RGB
    rgbd_image = open3d.create_rgbd_image_from_color_and_depth(
        open3d.Image(align_color_img.copy()),
        open3d.Image(depth_img),
        depth_scale=1.0 / depth_scale,
        depth_trunc=clipping_distance_in_meters,
        convert_rgb_to_intensity=False)
    # rgbd_image = open3d.geometry.create_rgbd_image_from_color_and_depth(open3d.Image(align_color_img.copy()),
    #                                                                     open3d.Image(depth_img),)
    return rgbd_image



def get_single_joint3D_orientation(body_joint3D, body_orientation=None):
    '''
    Currently only return single set of joint3D and orientations
    :param body_joint3D:
    :param body_orientation:
    :return:
    '''
    joint3D = np.zeros((PyKinectV2.JointType_Count, 3), dtype=np.float32)
    orientation = np.zeros((PyKinectV2.JointType_Count, 4), dtype=np.float32)
    if body_orientation is None:
        body_orientation = [orientation] * len(body_joint3D)
    for j, o in zip(body_joint3D, body_orientation):
        if j is None:
            continue
        joint3D = j
        orientation = o
    return joint3D, orientation

def get_single_joint3D_and_orientation(kinect, body_frame, depth_img, intrinsic, depth_scale):
    '''
    body_joint, body_orientation = utils.get_body_joint2D(body_frame, self.kinect,
                                                            map_space="depth_space")
    body_joint3D = utils.map_body_joint2D_3D(body_joint, depth_img, self.intrinsic, self.depth_scale)
    joint3D, orientation = utils.get_single_joint3D_orientation(body_joint3D, body_orientation)
    :param kinect:
    :param body_frame:
    :param depth_img:
    :param intrinsic:
    :param depth_scale:
    :return:
    '''
    joint3D = np.zeros((PyKinectV2.JointType_Count, 3), dtype=np.float32)
    orientation = np.zeros((PyKinectV2.JointType_Count, 4), dtype=np.float32)

    if body_frame is not None:
        for i in range(0, kinect.max_body_count):
            body = body_frame.bodies[i]
            if body.is_tracked:
                joints = body.joints
                joint_points = kinect.body_joints_to_depth_space(joints)  # Convert joint coordinates to depth space
                joint3D = get_joint3D(joints, joint_points, depth_img, intrinsic,
                                      depth_scale)  # Convert to numpy array format
                orientation = get_joint_quaternions(body.joint_orientations)

    # Note: Currently only return single set of joint3D and orientations
    return joint3D, orientation


def get_single_joint3D(kinect, body_frame, depth_img, intrinsic, depth_scale):
    joint3D = np.zeros((PyKinectV2.JointType_Count, 3), dtype=np.float32)
    if body_frame is not None:
        for i in range(0, kinect.max_body_count):
            body = body_frame.bodies[i]
            if body.is_tracked:
                joints = body.joints
                joint_points = kinect.body_joints_to_depth_space(joints)  # Convert joint coordinates to depth space
                joint3D = get_joint3D(joints, joint_points, depth_img, intrinsic,
                                      depth_scale)  # Convert to numpy array format

    # Note: Currently only return single set of joint3D and orientations
    return joint3D


def transform_geometry_quaternion(joint3D, orientation):
    qw, qx, qy, qz = orientation[0], orientation[1], orientation[2], orientation[3]
    tx, ty, tz = joint3D[0], joint3D[1], joint3D[2]

    # Convert quaternion to rotation matrix
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    transform_matrix = [[1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw, tx],
                        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw, ty],
                        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy, tz],
                        [0, 0, 0, 1]]

    return transform_matrix
