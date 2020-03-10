# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-12 18:28:16
# @url    :
# --------------------------------------------------------
"""

import argparse
import os
import glob
import numpy as np
import json
import xmltodict
import cv2
import PIL.Image
import time
import copy as copy
from modules.dataset_tool.voc_tools.custom_voc import CustomVoc


# from modules.dataset_tool.voc_tools.segmentation import SegmentationObject


def save_json(data_coco, json_file):
    """
    save COCO data in json file
    :param json_file:
    :return:
    """
    # json.dump(self.coco, open(json_file, 'w'))
    json.dump(data_coco, open(json_file, 'w'), indent=4)  # indent=4 更加美观显示
    # dirname = os.path.dirname(json_file)
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # with open(json_file, 'w') as f:
    #     json.dump(data_coco, f, indent=4)
    print("save file:{}".format(json_file))


def read_json(json_path):
    """
    读取数据
    :param json_path:
    :return:
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


class COCOTools(object):
    """COCO Tools"""

    @staticmethod
    def get_categories_id(categories):
        """
        get categories id dict
        :param categories:
        :return: dict:{name:id}
        """
        supercategorys = []
        categories_id = {}
        for item in categories:
            supercategory = item["supercategory"]
            name = item["name"]
            id = item["id"]
            categories_id[name] = id
        return categories_id

    @staticmethod
    def get_annotations_id(annotations):
        """
        get annotations id list
        :param annotations:
        :return: annotations id list
        """
        annotations_id = []
        for item in annotations:
            id = item["id"]
            annotations_id.append(id)
        return annotations_id

    @staticmethod
    def get_images_id(images):
        """
        get image id list
        :param images:
        :return: images id list
        """
        images_id = []
        for item in images:
            id = item["id"]
            images_id.append(id)
        return images_id

    @staticmethod
    def check_uniqueness(id_list: list, title="id"):
        """
        检测唯一性
        :return:
        """
        for i in id_list:
            n = id_list.count(i)
            assert n == 1, Exception("have same {}:{}".format(title, i))

    @staticmethod
    def check_coco(coco):
        """
        检测COCO合并后数据集的合法性
            检测1: 检测categories id唯一性
            检测2: 检测image id唯一性
            检测3: 检测annotations id唯一性
        :return:
        """
        categories_id = COCOTools.get_categories_id(coco["categories"])
        print("categories_id:{}".format(categories_id))
        categories_id = list(categories_id.values())
        COCOTools.check_uniqueness(categories_id, title="categories_id")

        image_id = COCOTools.get_images_id(coco["images"])
        COCOTools.check_uniqueness(image_id, title="image_id")

        annotations_id = COCOTools.get_annotations_id(coco["annotations"])
        COCOTools.check_uniqueness(annotations_id, title="annotations_id")
        print("have image:{}".format(len(image_id)))


class PascalVoc2Coco(CustomVoc):
    """Convert Pascal VOC Dataset to COCO dataset format"""

    def __init__(self, anno_dir, image_dir=None, seg_dir=None, init_id=None):
        """
        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`,if image_dir=None ,will ignore checking image shape
        :param seg_dir:   for voc `SegmentationObject`,if seg_dir=None,will ignore Segmentation Object
        :param image_id: 初始的image_id,if None,will reset to currrent time
        """
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.xml_list = self.get_xml_files(self.anno_dir)
        # self.seg = SegmentationObject()

        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        if not init_id:
            init_id = int(time.time()) * 2
        self.image_id = init_id
        # self.image_id = 20200207
        self.annotation_id = 0

    def addCatItem(self, name):
        """
        :param name:
        :return:
        """
        self.category_item_id += 1
        category_item = dict()
        category_item['supercategory'] = name
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        if name == "person":
            category_item['keypoints'] = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                                          'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                                          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
                                          'right_ankle']
            category_item['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                         [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4],
                                         [3, 5], [4, 6], [5, 7]]

        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, image_size):
        """
        :param file_name:
        :param image_size: [height, width]
        :return:
        """
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['height'] = image_size[0]
        image_item['width'] = image_size[1]
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def addAnnoItem(self, image_id, category_id, rect, seg, area, keypoints):
        """
        :param image_id:
        :param category_id:
        :param rect:[x,y,w,h]
        :param seg:
        :param area:
        :param keypoints:
        :return:
        """
        self.annotation_id += 1
        annotation_item = dict()
        annotation_item['segmentation'] = seg
        annotation_item['area'] = area
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id  #
        annotation_item['bbox'] = rect  # [x,y,w,h]
        annotation_item['category_id'] = category_id
        annotation_item['id'] = self.annotation_id
        annotation_item['num_keypoints'] = int(len(keypoints) / 3)
        annotation_item['keypoints'] = keypoints
        self.coco['annotations'].append(annotation_item)

    def generate_dataset(self):
        """
        :return:
        """
        for xml_file in self.xml_list:
            # convert XML to Json
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])

            filename = annotation["filename"]
            self.check_image(filename, shape=(height, width, depth))
            if filename in self.category_set:
                raise Exception('file_name duplicated')

            if filename not in self.image_set:
                image_size = [height, width]
                current_image_id = self.addImgItem(filename, image_size=image_size)
                print('add filename {}'.format(filename))
            else:
                raise Exception('duplicated image_dict: {}'.format(filename))

            objects = annotation["object"]
            if not isinstance(objects, list):
                objects = [objects]
            for object in objects:
                class_name = object["name"]
                if class_name not in self.category_set:
                    current_category_id = self.addCatItem(class_name)
                else:
                    current_category_id = self.category_set[class_name]
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
                # get segmentation info
                seg, area = self.get_segmentation_area(filename, bbox=bbox)
                self.addAnnoItem(current_image_id, current_category_id, rect, seg, area, keypoints)
        COCOTools.check_coco(self.coco)

    def get_coco(self):
        return self.coco

    def save_coco(self, json_file):
        save_json(self.get_coco(), json_file)


parser = argparse.ArgumentParser(description="COCO Dataset")
parser.add_argument("-i", "--image_dir", help="path/to/image", type=str)
parser.add_argument("-a", "--anno_dir", help="path/to/anno_dir", type=str)
parser.add_argument("-seg_dir", "--seg_dir", help="path/to/VOC/SegmentationObject", default=None, type=str)
parser.add_argument("-s", "--save_path", help="out/to/save_json-file", type=str)
parser.add_argument("-id", "--init_id", help="init id", type=int, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    image_dir = args.image_dir
    anno_dir = args.anno_dir
    seg_dir = args.seg_dir
    save_path = args.save_path
    init_id = args.init_id
    # anno_dir = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations'  # 这是xml文件所在的地址
    # seg_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/SegmentationObject"
    # image_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/JPEGImages"
    # json_file = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json'  # 这是你要生成的json文件

    seg_dir = None
    image_dir = "/media/dm/dm/project/dataset/coco/images/lexue_val"
    anno_dir = '/media/dm/dm/project/dataset/coco/annotations/lexue/annotations/lexue_val'  # 这是xml文所在的地址
    save_path = "/media/dm/dm/project/dataset/coco/annotations/lexue/lexue_val.json"
    #
    # image_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/JPEGImages"
    # anno_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations"
    # seg_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/SegmentationObject"
    # json_file = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json'  # 这是你要生成的json文件
    #
    VOC2coco = PascalVoc2Coco(anno_dir, image_dir=image_dir, seg_dir=seg_dir, init_id=init_id)
    VOC2coco.generate_dataset()
    VOC2coco.save_coco(save_path)
