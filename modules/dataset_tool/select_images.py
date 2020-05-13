# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : select_images.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-09-23 16:04:20
"""

import os
import cv2
import numpy as np
import shutil
import os.path
from utils import file_processing, image_processing
from libs.ultra_ligh_face.ultra_ligh_face import UltraLightFaceDetector


def ramdom_select_image_dir(image_dir, dest_dir):
    select_nums = 100
    image_id = file_processing.get_sub_directory_list(image_dir)
    for id in image_id:
        image_list = file_processing.get_files_list(os.path.join(image_dir, id),
                                                    postfix=['*.jpg', "*.jpeg", '*.png', "*.JPG"])
        image_list = np.random.permutation(image_list)[:select_nums]
        for src_path in image_list:
            basename = os.path.basename(src_path)
            dest_path = file_processing.create_dir(dest_dir, id, basename)
            shutil.copy(src_path, dest_path)


def select_image_dir(image_dir, dest_dir):
    image_id = file_processing.get_sub_directory_list(image_dir)
    for id in image_id:
        image_list = file_processing.get_files_list(os.path.join(image_dir, id),
                                                    postfix=['*.jpg', "*.jpeg", '*.png', "*.bmp"])
        for src_path in image_list:
            basename = os.path.basename(src_path)
            index = basename.split(".")[0].split("_")[1]
            if index == "0":
                dest_path = file_processing.create_dir(dest_dir, id, basename)
                # shutil.copy(src_path, dest_path)
                file_processing.move_file(src_path, dest_path)


def select_facebank_detect(image_dir, dest_dir, id_nums=1, detect_face=True):
    per_nums = 1
    if detect_face:
        # model_path = "../../face_detection/face_detection_rbf.pth"
        model_path = "/media/dm/dm1/git/python-learning-notes/libs/ultra_ligh_face/face_detection_rbf.pth"
        network = "RFB"
        confidence_threshold = 0.85
        nms_threshold = 0.3
        top_k = 500
        keep_top_k = 750
        device = "cuda:0"
        detector = UltraLightFaceDetector(model_path=model_path,
                                          network=network,
                                          confidence_threshold=confidence_threshold,
                                          nms_threshold=nms_threshold,
                                          top_k=top_k,
                                          keep_top_k=keep_top_k,
                                          device=device)
    image_id = file_processing.get_sub_directory_list(image_dir)
    print("have ID:{}".format(len(image_id)))
    image_id = image_id[:id_nums]
    for id in image_id:
        image_list = file_processing.get_files_list(os.path.join(image_dir, id),
                                                    postfix=['*.jpg', "*.jpeg", '*.png', "*.JPG"])
        count = 0
        for src_path in image_list:
            basename = os.path.basename(src_path)
            if detect_face:
                bgr_image = cv2.imread(src_path)
                bboxes, scores, landms = detector.detect(bgr_image, isshow=True)
            if count >= per_nums:
                break
            count += 1
            dest_path = file_processing.create_dir(dest_dir, id, basename)
            file_processing.copy_file(src_path, dest_path)


def select_facebank(image_dir, dest_dir, id_nums=10):
    per_nums = 1

    image_id = file_processing.get_sub_directory_list(image_dir)
    nums_images = len(image_id)
    print("have ID:{}".format(nums_images))
    if id_nums:
        id_nums = min(id_nums, nums_images)
        image_id = image_id[:id_nums]
    print("select ID:{}".format(nums_images))
    for id in image_id:
        image_list = file_processing.get_files_list(os.path.join(image_dir, id),
                                                    postfix=['*.jpg', "*.jpeg", '*.png', "*.JPG"])
        count = 0
        for src_path in image_list:
            basename = os.path.basename(src_path)
            if count >= per_nums:
                break
            count += 1
            dest_path = file_processing.create_dir(dest_dir, id, basename)
            file_processing.copy_file(src_path, dest_path)


def image_to_facebank(image_dir, dest_dir):
    from xpinyin import Pinyin
    p = Pinyin()
    image_list = file_processing.get_files_list(image_dir,
                                                postfix=['*.jpg', "*.jpeg", '*.png', "*.JPG"])
    nums_images = len(image_list)
    print("have ID:{}".format(nums_images))
    for image_path in image_list:
        basename = os.path.basename(image_path)
        id_name = basename.split(".")[0]
        id_name = p.get_pinyin(id_name, '')
        dest_path = file_processing.create_dir(dest_dir, id_name, basename)
        file_processing.copy_file(image_path, dest_path)


if __name__ == "__main__":
    # image_dir = '/media/dm/dm1/FaceDataset/lexue/lexue1/val-src'
    # dest_dir = '/media/dm/dm1/FaceDataset/lexue/lexue1/val'
    # ramdom_select_image_dir(image_dir, dest_dir)

    # image_dir = '/media/dm/dm1/FaceDataset/X4/CASIA-FaceV5/trainval'
    # dest_dir = '/media/dm/dm1/FaceDataset/X4/CASIA-FaceV5/facebank'
    # select_image_dir(image_dir, dest_dir)

    # image_dir = "/data0/panjinquan/FaceData/DMFR_V1"
    # dest_dir = "/data0/panjinquan/FaceData/facebank_DMFR_V1"

    image_dir = "/media/dm/dm/FaceRecognition/dataset/Facebank/工牌照片"
    dest_dir = "/media/dm/dm/FaceRecognition/dataset/Facebank/badge"

    image_to_facebank(image_dir, dest_dir)
