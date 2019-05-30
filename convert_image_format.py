# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : convert_image_format.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-27 11:31:07
"""
from utils import image_processing,file_processing
import os
def convert_image_format(image_dir,out_dir):
    image_list=file_processing.get_files_list(image_dir,postfix=["*.bmp"])
    for image_path in image_list:
        image= image_processing.read_image(image_path)
        basename=os.path.basename(image_path).replace("bmp","jpg")
        dest_path=os.path.join(out_dir,basename)
        image_processing.save_image(dest_path,image,toUINT8=False)

if __name__=="__main__":
    image_dir="/media/dm/dm/project/dataset/VOC_wall/src"
    out_dir="/media/dm/dm/project/dataset/VOC_wall/images"
    convert_image_format(image_dir, out_dir)

