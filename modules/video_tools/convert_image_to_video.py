# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-01 14:15:41
# --------------------------------------------------------
"""
import os
import numpy as np
from tqdm import tqdm
from utils import image_processing, file_processing
import cv2


class CVVideo(object):
    def convert_images2video(self, image_dir, save_video, freq=1, fps=30):
        """
        :param image_dir:
        :param save_video:
        :param freq:
        :return:
        """
        image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg", "*.png"])
        image_path = image_list[0]
        frame = cv2.imread(image_path)
        h, w, d = frame.shape
        video_writer = image_processing.get_video_writer(save_video, width=w, height=h, fps=fps)
        # freq = int(fps / detect_freq)
        count = 0
        for image_path in tqdm(image_list):
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            if count % freq == 0:
                out_frame = self.do_something(frame)
                video_writer.write(out_frame)

            count += 1
        video_writer.release()

    def convert_video2images(self, video_path, save_dir, freq=1):
        """
        :param video_path:
        :param save_dir:
        :param freq:
        :return:
        """
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % freq == 0:
                out_frame = self.do_something(frame)
                path = os.path.join(save_dir, "{:0=6d}.jpg".format(count))
                cv2.imwrite(path, out_frame)
            count += 1
        video_cap.release()

    def do_something(self, frame):
        pass
        return frame


if __name__ == "__main__":
    time = file_processing.get_time()
    cvv = CVVideo()
    image_dir = "/media/dm/dm/FaceRecognition/face_cpp/outputs"
    save_video = "/media/dm/dm/FaceRecognition/face_cpp/demo.avi"
    cvv.convert_images2video(image_dir, save_video)
    # cvv.convert_video2images(save_video, image_dir)
