# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-01 14:15:41
# --------------------------------------------------------
"""
from utils import image_processing, file_processing
import numpy as np
import cv2
from libs.ultra_ligh_face.ultra_ligh_face import UltraLightFaceDetector


class CVVideo():
    def __init__(self):
        pass

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        self.scale = 0.5
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        if save_video:
            # (760, 540, 3)
            self.video_writer = image_processing.get_video_writer(save_video,
                                                                  width=540,
                                                                  height=810,
                                                                  fps=fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out_frame = self.do_something(frame)
            if save_video:
                self.write_video(out_frame)
            count += 1
        video_cap.release()

    def write_video(self, img):
        self.video_writer.write(img)

    def do_something(self, frame):
        frame = np.rot90(frame, -1)
        h, w, d = frame.shape
        resize_height = int(h / 2)
        frame = image_processing.resize_image(frame, resize_height=resize_height)
        bboxes_list = [[0, 150, w, h]]
        frame = image_processing.get_bboxes_image(frame, bboxes_list)[0]
        out_frame = np.asarray(frame)
        cv2.imshow("image", out_frame)
        # cv2.imwrite("image.png",out_frame)
        cv2.waitKey(3)
        return out_frame


if __name__ == "__main__":
    time = file_processing.get_time()
    cvv = CVVideo()
    video_path = "/media/dm/dm2/X2/Pose/3DPose/VideoPose3D/data/video/video3.mp4"
    # video_path = 1
    save_video = "/media/dm/dm2/X2/Pose/3DPose/VideoPose3D/data/video/video4.avi".format(time)
    cvv.start_capture(video_path, save_video)
