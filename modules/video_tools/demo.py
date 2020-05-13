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
        model_path = "/media/dm/dm1/git/python-learning-notes/libs/ultra_ligh_face/face_detection_rbf.pth"
        network = "RFB"
        confidence_threshold = 0.85
        nms_threshold = 0.3
        top_k = 500
        keep_top_k = 750
        device = "cuda:0"
        self.detector = UltraLightFaceDetector(model_path=model_path,
                                               network=network,
                                               confidence_threshold=confidence_threshold,
                                               nms_threshold=nms_threshold,
                                               top_k=top_k,
                                               keep_top_k=keep_top_k,
                                               device=device)

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        video_cap = image_processing.get_video_capture(video_path,width=640, height=480)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_processing.get_video_writer(save_video, width, height, fps)
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

    # def do_something(self, frame):
    #     frame = np.rot90(frame, -1)
    #     h, w, d = frame.shape
    #     resize_height = int(h / 2)
    #     frame = image_processing.resize_image(frame, resize_height=resize_height)
    #     bboxes_list = [[0, 200, w, h]]
    #     frame = image_processing.get_bboxes_image(frame, bboxes_list)[0]
    #     out_frame = np.asarray(frame)
    #     cv2.imshow("image", out_frame)
    #     cv2.imwrite("image.png",out_frame)
    #     cv2.waitKey(0)
    #     return out_frame

    def do_something(self, frame):
        bgr_image = frame.copy()
        bboxes, scores, landms = self.detector.detect(bgr_image, isshow=False)
        bgr_image = image_processing.draw_landmark(bgr_image, landms, vis_id=True)
        bgr_image = image_processing.draw_image_bboxes_text(bgr_image, bboxes, scores, color=(0, 0, 255))
        cv2.imshow("image", bgr_image)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(10)
        return frame


if __name__ == "__main__":
    time = file_processing.get_time()
    cvv = CVVideo()
    # video_path = "/media/dm/dm/FaceRecognition/dataset/乐学_带底库视频/X3-1-S.mp4"
    video_path = 0
    save_video = "/media/dm/dm/FaceRecognition/dataset/Facebank/video/{}_test.avi".format(time)
    cvv.start_capture(video_path, save_video)
