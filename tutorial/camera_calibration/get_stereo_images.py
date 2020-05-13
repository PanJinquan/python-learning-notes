# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : __init__.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2020-04-10 18:24:06
"""

import cv2
import os


class VideoCapture():
    def __init__(self):
        pass

    @staticmethod
    def get_video_info(video_cap):
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        numFrames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        return width, height, numFrames, fps

    @staticmethod
    def capture(save_dir):
        """
        :param save_dir:
        :return:
        """
        # Set the values for your cameras
        capL = cv2.VideoCapture(0)
        capR = cv2.VideoCapture(1)
        widthL, heightL, numFramesL, fpsL = VideoCapture.get_video_info(capL)
        widthR, heightR, numFramesR, fpsR = VideoCapture.get_video_info(capR)
        print("capL:\n", widthL, heightL, numFramesL, fpsL)
        print("capR:\n", widthR, heightR, numFramesR, fpsR)
        # Use these if you need high resolution.
        # capL.set(3, 1024) # width
        # capL.set(4, 768) # height

        # capR.set(3, 1024) # width
        # capR.set(4, 768) # height

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        i = 0
        while True:
            # Grab and retreive for sync
            if not (capL.grab() and capR.grab()):
                print("No more frames")
                break
            _, leftFrame = capL.retrieve()
            _, rightFrame = capR.retrieve()

            # Use if you need high resolution. If you set the camera for high res, you can pass these.
            # cv2.namedWindow('capL', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('capL', 1024, 768)
            # cv2.namedWindow('capR', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('capR', 1024, 768)

            cv2.imshow('left', leftFrame)
            cv2.imshow('right', rightFrame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('c') or key == ord('s'):
                print("save image:{}".format(i))
                cv2.imwrite(os.path.join(save_dir, "left" + str(i) + ".png"), leftFrame)
                cv2.imwrite(os.path.join(save_dir, "right" + str(i) + ".png"), rightFrame)
                i += 1

        capL.release()
        capR.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    save_dir = "data/dataset"
    vc = VideoCapture()
    vc.capture(save_dir)
