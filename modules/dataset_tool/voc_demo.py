# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-03-16 09:20:14
# --------------------------------------------------------
"""

from modules.dataset_tool.voc_tools.custom_voc import CustomVoc

if __name__ == '__main__':
    # image_dir = "/media/dm/dm2/FaceRecognition/dataset/facebody_align_data/face_id/lx2_sitting"
    # anno_dir = "/media/dm/dm2/FaceRecognition/dataset/facebody_align_data/face_id/lx2_sitting_anno"


    image_dir = "/media/dm/dm2/FaceRecognition/dataset/facebody_align_data/face_id/hh1_handing"
    anno_dir = "/media/dm/dm2/FaceRecognition/dataset/facebody_align_data/face_id/hh1_handing_anno"

    VOC2coco = CustomVoc(anno_dir, image_dir=image_dir, seg_dir=None)
    VOC2coco.decode_voc()
