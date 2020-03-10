# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-03-06 19:16:29
# --------------------------------------------------------
"""

from modules.dataset_tool.voc_tools.custom_voc import CustomVoc

if __name__ == '__main__':
    anno_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/xmc2_anno_data/teacher_2D_pose_estimator/annotations/val/xml"
    image_dir = None
    json_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/xmc2_anno_data/teacher_2D_pose_estimator/annotations/val/json"
    VOC2coco = CustomVoc(anno_dir, image_dir=image_dir, seg_dir=None)
    VOC2coco.write_to_json(json_dir)
