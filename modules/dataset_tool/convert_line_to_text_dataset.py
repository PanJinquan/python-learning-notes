# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : convert_linedata.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-08-28 14:37:43
"""
import os
from utils import file_processing, image_processing


def convert_line_to_text_dataset(anno_filename, label_out_dir, show=True):
    file_processing.create_dir(label_out_dir)
    boxes_label_lists = file_processing.read_lines_image_labels(anno_filename)
    for image_id, box, label in boxes_label_lists:
        filename = image_id[:-len('jpg')] + "txt"
        content_list = [[c] + r for c, r in zip(label, box)]
        path = os.path.join(label_out_dir, filename)
        file_processing.write_data(path, content_list, mode='w')


if __name__ == "__main__":
    image_dir = '/media/dm/dm2/XMC/FaceDataset/NVR/NVR-Teacher2/image'
    anno_filename = "/media/dm/dm2/XMC/FaceDataset/NVR/NVR-Teacher2/teacher_data_anno.txt"
    label_out_dir = "/media/dm/dm2/XMC/FaceDataset/NVR/NVR-Teacher2/label"
    convert_line_to_text_dataset(anno_filename, label_out_dir)
