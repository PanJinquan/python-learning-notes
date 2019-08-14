# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : convert_image_format.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-27 11:31:07
"""
from utils import image_processing, file_processing
import os


def convert_images_format(image_dir, out_dir):
    image_list = file_processing.get_files_list(image_dir, postfix=["*.bmp"])
    for image_path in image_list:
        image = image_processing.read_image(image_path)
        basename = os.path.basename(image_path).replace("bmp", "jpg")
        # dest_path=os.path.join(out_dir,basename)
        dest_path = file_processing.create_dir(out_dir, dir1=None, filename=basename)
        image_processing.save_image(dest_path, image, toUINT8=False)


def convert_images_dirs_format(src_image_dir, out_dir):
    image_list, image_label = file_processing.get_files_labels(src_image_dir, postfix=["*.bmp"])
    for i, (image_path, label) in enumerate(zip(image_list, image_label)):
        image = image_processing.read_image(image_path)
        basename = os.path.basename(image_path).replace("bmp", "jpg")
        # dest_path=os.path.join(out_dir,basename)
        dest_path = file_processing.create_dir(out_dir, dir1=label, filename=basename)
        image_processing.save_image(dest_path, image, toUINT8=False)
        if i % 100 == 0 or i == len(image_list) - 1:
            print("processing:{}/{}".format(i, len(image_list)))


if __name__ == "__main__":
    dataset = "/media/dm/dm2/project/dataset/face_recognition/CASIA-FaceV5/"
    image_dir = dataset + "CASIA-FaceV5"
    out_dir = dataset + "CASIA-FaceV5_JPG"
    convert_images_dirs_format(image_dir, out_dir)
