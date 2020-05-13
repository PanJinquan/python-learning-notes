########################################################################
### Sample program to stream
### Coloured point cloud, joint and joint orientation in 3D using Open3D
########################################################################
import cv2
import numpy as np
import open3d as open3d
import utils_PyKinectV2 as utils
import open3d as open3d
from numpy.linalg import inv
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

#############################
### Kinect runtime object ###
#############################
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body |
                                         PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Depth)

depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height  # Default: 512, 424
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height  # Default: 1920, 1080

##############################
### User defined variables ###
##############################
depth_scale = 0.001  # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
# depth_scale                 = 1.0 # Default kinect depth scale where 1 unit = 0.001 m = 1 mm
clipping_distance_in_meters = 1.5  # Set the maximum distance to display the point cloud data
clipping_distance = clipping_distance_in_meters / depth_scale  # Convert dist in mm to unit
width = depth_width;
height = depth_height;
ppx = 260.166;
ppy = 205.197;
fx = 367.535;
fy = 367.535  # Hardcode the camera intrinsic parameters for backprojection
# width=depth_width; height=depth_height; ppx=258.981; ppy=208.796; fx=367.033; fy=367.033 # Hardcode the camera intrinsic parameters for backprojection

############################
### Open3D visualisation ###
############################
intrinsic = open3d.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]  # To convert [x,y,z] -> [x.-y,-z]
# Define the objects to be drawn
image_pcd = open3d.PointCloud()
bone_line_pcd = utils.create_line_set_bones(np.zeros((24, 3), dtype=np.float32))  # 24 bones connecting 25 joints
axis_pcd = []
for i in range(PyKinectV2.JointType_Count):  # 25 axes for 25 joints
    axis_pcd.append(open3d.create_mesh_coordinate_frame(size=0.1, origin=[0, 0, 0]))  # XYZ axis length of 0.1 m
# Create Open3D Visualizer
vis = open3d.Visualizer()
vis.create_window(width=width, height=height)
vis.get_render_option().point_size = 3
vis.add_geometry(bone_line_pcd)
for i in range(PyKinectV2.JointType_Count):  # 25 axes for 25 joints
    vis.add_geometry(axis_pcd[i])

first_loop = True
while True:
    ##############################
    ### Get images from camera ###
    ##############################
    if kinect.has_new_body_frame() and kinect.has_new_color_frame() and kinect.has_new_depth_frame():

        body_frame = kinect.get_last_body_frame()
        color_frame = kinect.get_last_color_frame()
        depth_frame = kinect.get_last_depth_frame()

        #########################################
        ### Reshape from 1D frame to 2D image ###
        #########################################
        color_img = color_frame.reshape(((color_height, color_width, 4))).astype(np.uint8)
        depth_img = depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16)

        ###############################################
        ### Useful functions in utils_PyKinectV2.py ###
        ###############################################
        align_color_img = utils.get_align_color_image(kinect, color_img)
        image_pcd.points, image_pcd.colors = utils.create_color_point_cloud(align_color_img, depth_img, depth_scale,
                                                                            clipping_distance_in_meters, intrinsic)
        image_pcd.transform(flip_transform)
        joint3D, orientation = utils.get_single_joint3D_and_orientation(kinect, body_frame, depth_img, intrinsic,
                                                                        depth_scale)
        bone_line_pcd.points = open3d.Vector3dVector(joint3D)
        bone_line_pcd.transform(flip_transform)

        ###############################################
        ### Draw the orientation axes at each joint ###
        ###############################################
        for i in range(PyKinectV2.JointType_Count):
            axis_pcd[i].transform(utils.transform_geometry_quaternion(joint3D[i, :], orientation[i, :]))
            axis_pcd[i].transform(flip_transform)

        if first_loop:
            vis.add_geometry(image_pcd)
            first_loop = False
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

        # Need to inverse the transformation else it will be additive in every loop
        for i in range(PyKinectV2.JointType_Count):
            axis_pcd[i].transform(flip_transform)
            axis_pcd[i].transform(inv(np.array(utils.transform_geometry_quaternion(joint3D[i, :], orientation[i, :]))))

        # show_pcd = []
        # show_pcd.append(image_pcd)
        # show_pcd.append(axis_pcd)
        # show_pcd.append(bone_line_pcd)
        # open3d.visualization.draw_geometries(show_pcd)
        ######################################
        ### Display 2D images using OpenCV ###
        ######################################
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=255 / clipping_distance),
                                           cv2.COLORMAP_JET)
        cv2.imshow('depth', depth_colormap)  # (424, 512)

    key = cv2.waitKey(30)
    if key == 27:  # Press esc to break the loop
        break

kinect.close()
vis.destroy_window()
cv2.destroyAllWindows()
