# -*-coding: utf-8 -*-
"""
    @Project: DepthDemo
    @File   : stereocamera_demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-12 14:41:34
"""
import os

# os.path.join("modules/humanpose")
import cv2
import numpy as np
# import config.config as camera_configs
from tutorial.stereo_camera.config import ov0575_config as camera_configs
# from config import ov0575_config2 as camera_configs
# from config import ov0575_config3 as camera_configs
# from config import smkdt01_config as camera_configs
from utils import image_processing, file_processing
from modules.utils_3d import open3d_visual
from libs.ultra_ligh_face.ultra_ligh_face import UltraLightFaceDetector


class StereoCamera():
    def __init__(self, invert_camera=False):
        self.win_depth = "depth"
        self.win_left = "left_rectified"
        self.win_right = "right_rectified"
        self.win_disparity = "disparity"
        self.win_colormap = "colormap"
        cv2.namedWindow(self.win_depth)
        cv2.createTrackbar("num", self.win_depth, 2, 10, lambda x: None)
        cv2.createTrackbar("blockSize", self.win_depth, 5, 255, lambda x: None)
        self.ecv = image_processing.EventCv()
        self.ecv.add_mouse_event(self.win_depth)
        self.ecv.add_mouse_event(self.win_disparity)
        self.ecv.add_mouse_event(self.win_colormap)

        camera_height = camera_configs.camera_height
        camera_width = camera_configs.camera_width
        input_size = [camera_height, camera_width]
        # self.pose = openpose.OpenPose(model_path, input_size=input_size)
        if invert_camera:
            self.camera1 = cv2.VideoCapture(1)  # left camera1
            self.camera2 = cv2.VideoCapture(0)  # right camera2
        else:
            self.camera1 = cv2.VideoCapture(0)  # left camera1
            self.camera2 = cv2.VideoCapture(1)  # right camera2

        self.detector = UltraLightFaceDetector(model_path=None, network=None, )
        self.g = open3d_visual.Open3DVisual(camera_intrinsic=camera_configs.camera_intrinsic,
                                            depth_width=camera_configs.depth_width,
                                            depth_height=camera_configs.camera_height,
                                            depth_scale=camera_configs.depth_scale,
                                            clipping_distance_in_meters=camera_configs.clipping_distance_in_meters)
        self.g.show_origin_pcd()
        self.g.show_bone_line_pcd()
        self.g.show_image_pcd(True)

        class_name = "ttt"
        real_part = True
        # real_part = False
        dataset_root = "dataset/"
        self.scale = 1
        self.prefix = "v"
        if real_part:
            self.snapshot_dir = os.path.join(dataset_root, "real_part", class_name)
        else:
            self.snapshot_dir = os.path.join(dataset_root, "fake_part", class_name)
        file_processing.create_dir(self.snapshot_dir, "color")
        file_processing.create_dir(self.snapshot_dir, "depth")
        # file_processing.create_dir(self.snapshot_dir, "ir")
        file_processing.create_dir(self.snapshot_dir, "video")
        video_name = file_processing.get_time()
        self.save_video = os.path.join(self.snapshot_dir, "video", "{}_{}.avi".format(class_name, video_name))
        if self.save_video:
            self.video_writer = self.get_video_writer(self.save_video,
                                                      width=640,
                                                      height=480,
                                                      fps=20)  # (424, 512, 4)

    @staticmethod
    def get_video_writer(save_path, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frameSize = (width, height)
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
        print("video:width:{},height:{},fps:{}".format(width, height, fps))
        return video_writer

    def get_rectify_image(self, image_left, image_right):
        '''
        畸变校正和立体校正
        根据更正map对图片进行重构
        获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
        :param image_left:
        :param image_right:
        :return:
        '''
        left_rectified = cv2.remap(image_left, camera_configs.left_map1, camera_configs.left_map2,
                                   cv2.INTER_LINEAR)
        right_rectified = cv2.remap(image_right, camera_configs.right_map1, camera_configs.right_map2,
                                    cv2.INTER_LINEAR)
        return left_rectified, right_rectified

    def get_disparity_map(self, left_rectified, right_rectified, num=1, blockSize=1, stereo_type="SGBM"):
        '''
        获得视差图
        :param imgL:
        :param imgR:
        :param num:
        :param blockSize:
        :param stereo_type
        :return:
        '''
        # 将图片置为灰度图，为StereoBM作准备
        imgL = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
        if stereo_type == "BM":
            stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
            # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        elif stereo_type == "SGBM":
            # SGBM匹配参数设置
            if imgL.ndim == 2:
                img_channels = 1
            else:
                img_channels = 3
            # blockSize = 3
            num = np.where(num > 0, num, 1)
            param = {'minDisparity': 0,
                     'numDisparities': 16 * num,
                     'blockSize': blockSize,
                     'P1': 2 * img_channels * blockSize,
                     'P2': 4 * img_channels * blockSize,
                     'disp12MaxDiff': 1,
                     'preFilterCap': 65,
                     'uniquenessRatio': 10,
                     'speckleWindowSize': 150,
                     'speckleRange': 2,
                     'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                     }
            # 构建SGBM对象
            # print(param)
            stereo = cv2.StereoSGBM_create(**param)
        # disparity_left = stereo.compute(imgL, imgR)   # 计算左视差图
        # disparity_right = stereo.compute(imgR, imgL)  # 计算右视差图
        disparity = stereo.compute(imgL, imgR)

        # 除以16得到真实视差（因为SGBM算法得到的视差是×16的）
        disparity[disparity < 0] = 0
        # disparity.astype(np.float32) / 16.
        disparity = np.divide(disparity.astype(np.float32), 16.)
        return disparity

    # 顺时针旋转90度
    def RotateClockWise90(self, img):
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 1)
        return new_img

    def get_depth(self, disparity, scale=1.0, method=False):
        '''
        reprojectImageTo3D(disparity, Q),输入的Q,单位必须是毫米(mm)
        :param disparity:
        :param only_depth:
        :param scale:
        :return: returm scale=1.0,距离,单位为毫米
        '''
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        if method:
            depth = cv2.reprojectImageTo3D(disparity, camera_configs.Q)
            x, y, depth = cv2.split(depth)
        else:
            # baseline = 21.50635
            # fx = 419.29128272967574
            baseline = abs(camera_configs.T[0])
            # fx = abs(camera_configs.left_camera_matrix[0][0])
            fx = abs(camera_configs.Q[2, 3])
            depth = (fx * baseline) / disparity
        depth = depth * scale
        depth = np.asarray(depth, dtype=np.uint16)

        return depth

    def start_capture(self):
        snapshot_dir = "data/ov0575"
        file_processing.create_dir(snapshot_dir)
        self.count = 0
        while True:
            # 两个trackbar用来调节不同的参数查看效果
            num = cv2.getTrackbarPos("num", "depth")
            blockSize = cv2.getTrackbarPos("blockSize", "depth")
            if blockSize % 2 == 0:
                blockSize += 1
            if blockSize < 5:
                blockSize = 5
            # 获取左右摄像头的数据
            ret1, frameR = self.camera1.read()  # <class 'tuple'>: (480, 640, 3)
            ret2, frameL = self.camera2.read()  # <class 'tuple'>: (480, 640, 3)
            # frameL=self.RotateClockWise90(frameL)
            # frameR=self.RotateClockWise90(frameR)
            # 畸变校正和立体校正
            self.left_rectified, self.right_rectified = self.get_rectify_image(image_left=frameL, image_right=frameR)
            # 获得视差图
            self.disparity = self.get_disparity_map(self.left_rectified, self.right_rectified, num, blockSize)
            # 计算像素点的3D坐标（左相机坐标系下）
            self.depth = self.get_depth(self.disparity)
            self.depth = np.asarray(self.depth, dtype=np.uint16)
            # save_dir = "./data/data01"
            # self.left_rectified, self.depth, _ = rgbd_data.load_body_joint_data(save_dir, count=40)
            # self.g.show(self.left_rectified, self.depth, body_joint2D, body_rects=body_rects)
            # self.g.show(frameL, self.depth, body_joint=None)
            bboxes, scores, landms = self.detector.detect(self.left_rectified, isshow=False)
            self.show_landmark_boxes("Det", self.left_rectified, bboxes, scores, landms)
            self.display(bboxes)

    @staticmethod
    def show_landmark_boxes(win_name, image, bboxes, scores, landms):
        '''
        显示landmark和boxes
        :param win_name:
        :param image:
        :param landmarks_list: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        '''
        image = image_processing.draw_landmark(image, landms, vis_id=True)
        image = image_processing.draw_image_bboxes_text(image, bboxes, scores, color=(0, 0, 255))
        cv2.imshow(win_name, image)

    def display(self, bboxes):
        self.g.show(color_image=self.left_rectified, depth_image=self.depth)
        self.ecv.update_image(self.depth)
        # disparity_image = cv2.normalize(self.disparity, self.disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        #                                 dtype=cv2.CV_8U)
        disparity_image = np.uint8(self.disparity.clip(1, 4080) / 16.)  # 转换为uint8时，需要避免溢出255*16=4080
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(self.depth, alpha=255 / camera_configs.clipping_distance),
            cv2.COLORMAP_JET)
        cv2.imshow(self.win_left, self.left_rectified)
        cv2.imshow(self.win_right, self.right_rectified)
        cv2.imshow(self.win_disparity, disparity_image)
        cv2.imshow(self.win_depth, self.depth)
        cv2.imshow(self.win_colormap, depth_colormap)
        cv2.waitKey(20)
        self.count += 1
        # self.flag = cv2.waitKey(20) & 0xFF == ord('s')
        freq = 3
        self.save_videos(self.left_rectified, bboxes)
        if self.snapshot_dir and len(bboxes) > 0 and self.count % freq == 0:
            print("save image:{}".format(self.count))
            # self.flag = True
            # pre = "{}.png".format(self.count)
            prefix = "{}_{}.png".format(self.prefix, self.count)
            cv2.imwrite(os.path.join(self.snapshot_dir, "depth/{}".format(prefix)), self.depth)
            cv2.imwrite(os.path.join(self.snapshot_dir, "color/{}".format(prefix)), self.left_rectified)
            # cv2.imwrite(os.path.join(self.snapshot_dir, "ir/{}".format(prefix)), ir_frame)

    def save_videos(self, color_img, bboxes):
        """
        <class 'tuple'>: (480, 640, 3)
        :param color_img:
        :param bboxes:
        :return:
        """
        if self.save_video and len(bboxes) > 0:
            # image_processing.cv_show_image("color_img", color_img, waitKey=1)
            # color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2RGB)  # 将BGR转为RGB
            self.video_writer.write(color_img)

    def close(self):
        self.camera1.release()
        self.camera2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sc = StereoCamera()
    sc.start_capture()
