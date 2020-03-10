# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-03-05 11:32:09
# --------------------------------------------------------
"""
import os
import cv2
import glob
import xmltodict
import numpy as np
from modules.dataset_tool.voc_tools.segmentation import SegmentationObject
from utils import image_processing, file_processing


class CustomVoc():
    # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
    #             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    # skeleton = np.asarray(skeleton) - 1

    # skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    #             (8, 10), (11, 13), (12, 14), (13, 15), (14, 16), (11, 17), (12, 17)]

    skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
                (8, 10), (11, 13), (12, 14), (13, 15), (14, 16), (11, 18), (12, 18), (18, 17)]

    def __init__(self, anno_dir, image_dir=None, seg_dir=None):
        """
        Custom VOC dataset
        :param anno_dir:
        :param image_dir:
        :param seg_dir:
        """
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.xml_list = self.get_xml_files(self.anno_dir)
        self.seg = SegmentationObject()

    @staticmethod
    def get_xml_files(xml_dir):
        """
        :param xml_dir:
        :return:
        """
        xml_path = os.path.join(xml_dir, "*.xml")
        xml_list = glob.glob(xml_path)
        return xml_list

    @staticmethod
    def read_xml2json(xml_file):
        """
        :param xml_file:
        :return:
        """
        with open(xml_file) as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def check_image(self, filename, shape: tuple):
        """
        check image size
        :param filename:
        :param shape:
        :return:
        """
        if self.image_dir:
            image_path = os.path.join(self.image_dir, filename)
            assert os.path.exists(image_path), "not path:{}".format(image_path)
            image = cv2.imread(image_path)
            _shape = image.shape
            assert _shape == shape, "Error:{}".format(image_path)

    def get_segmentation_area(self, filename, bbox):
        """
        :param filename:
        :param bbox:[xmin, ymin, xmax, ymax]
        :return:
        """
        seg = []
        area = 0
        if self.seg_dir:
            # if exist VOC SegmentationObject
            seg_path = os.path.join(self.seg_dir, filename.split('.')[0] + '.png')
            seg, area = self.seg.get_segmentation_area(seg_path, bbox)
        if not seg:
            # cal seg and area by bbox
            xmin, ymin, xmax, ymax = bbox
            seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
            area = (xmax - xmin) * (ymax - ymin)
        return seg, area

    def get_annotation(self, xml_file):
        """
        keypoints = object["keypoints"]
        joint = np.asarray(keypoints).reshape(17, 3)
        joint = joint[:, 0:2]
        :param xml_file:
        :return:
        """
        content = self.read_xml2json(xml_file)
        annotation = content["annotation"]
        # get image shape
        width = int(annotation["size"]["width"])
        height = int(annotation["size"]["height"])
        depth = int(annotation["size"]["depth"])

        filename = annotation["filename"]
        self.check_image(filename, shape=(height, width, depth))

        objects_list = []
        objects = annotation["object"]
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            class_name = object["name"]
            xmin = float(object["bndbox"]["xmin"])
            xmax = float(object["bndbox"]["xmax"])
            ymin = float(object["bndbox"]["ymin"])
            ymax = float(object["bndbox"]["ymax"])
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = [xmin, ymin, xmax, ymax]
            # get person keypoints ,if exist
            if 'keypoints' in object:
                keypoints = object["keypoints"]
                keypoints = [float(i) for i in keypoints.split(",")]
            else:
                keypoints = [0] * 17 * 3
            kp_bbox = {}
            kp_bbox["keypoints"] = keypoints
            kp_bbox["bbox"] = bbox
            kp_bbox["class_name"] = class_name
            objects_list.append(kp_bbox)
            # seg, area = self.get_segmentation_area(filename, bbox=bbox)
        annotation_dict = {}
        annotation_dict["image"] = filename
        annotation_dict["object"] = objects_list
        return annotation_dict

    def decode_voc(self, vis=True):
        """
        :return:
        """
        for xml_file in self.xml_list:
            anns = self.get_annotation(xml_file)
            filename = anns["image"]
            object = anns["object"]
            keypoints = []
            bboxes = []
            for item in object:
                joint = item["keypoints"]
                joint = np.asarray(joint).reshape(17, 3)
                joint = joint[:, 0:2]
                keypoints.append(joint.tolist())
                bboxes.append(item["bbox"])
            image_path = os.path.join(self.image_dir, filename)
            image = image_processing.read_image(image_path)
            self.show(filename, image, keypoints, bboxes, vis=True)

    def write_to_json(self, json_dir):
        file_processing.create_dir(json_dir)
        for xml_file in self.xml_list:
            anns = self.get_annotation(xml_file)
            name = os.path.basename(xml_file)[:-len(".jpg")]
            json_path = os.path.join(json_dir, name + ".json")
            file_processing.write_json_path(json_path, anns)

    def show(self, filename, image, keypoints, bboxes, vis=True):
        save_image = True
        for i, joints in enumerate(keypoints):
            if np.sum(np.asarray(joints[5])) == 0 or np.sum(np.asarray(joints[6])) == 0 or \
                    np.sum(np.asarray(joints[11])) == 0 or np.sum(np.asarray(joints[12])) == 0:
                save_image = False
            else:
                save_image = True
            chest_joint = (np.asarray(joints[5]) + np.asarray(joints[6])) / 2
            hip_joint = (np.asarray(joints[11]) + np.asarray(joints[12])) / 2
            keypoints[i].append(chest_joint.tolist())
            keypoints[i].append(hip_joint.tolist())

        if vis:
            image_processing.show_image_boxes(None, image, bboxes)
            # image_processing.show_image_boxes(None, image, joints_bbox, color=(255, 0, 0))
            image = image_processing.draw_key_point_in_image(image, keypoints, pointline=self.skeleton)
            # image_processing.cv_show_image("Det", image, waitKey=0)
            if save_image:
                out_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/Posture/1"
                out_dir = file_processing.create_dir(out_dir)
                out_image_path = os.path.join(out_dir, filename)
                image_processing.save_image(out_image_path, image)
            else:
                out_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/Posture/unknown/1"
                out_dir = file_processing.create_dir(out_dir)
                out_image_path = os.path.join(out_dir, filename)
                image_processing.save_image(out_image_path, image)



if __name__ == '__main__':
    # anno_dir = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations'  # 这是xml文件所在的地址
    # seg_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/SegmentationObject"
    # image_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/JPEGImages"
    # json_file = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json'  # 这是你要生成的json文件

    image_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/images/1"
    anno_dir = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/annotations/src/1"
    label_file = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/annotations/src/1_bbox.txt"
    out_voc_ann = "/media/dm/dm/project/dataset/COCO/HumanPose/LeXue_teacher/annotations/voc/1"
    #
    VOC2coco = CustomVoc(out_voc_ann, image_dir=image_dir, seg_dir=None)
    VOC2coco.decode_voc()
