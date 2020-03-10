# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-12 18:28:16
# --------------------------------------------------------
"""
"""
https://blog.csdn.net/weixin_41765699/article/details/100124689
"""
import glob
import numpy as np
import xml.etree.ElementTree as ET
import os
import json
import xmltodict


class PascalVOC2coco():
    def __init__(self):
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = -1
        self.image_id = 20180000000
        self.annotation_id = 0

    def addCatItem(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        self.category_item_id += 1
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id

    def addImgItem(self, file_name, size):
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        image_item['file_name'] = file_name
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        return self.image_id

    def get_segmentation_area(self, bbox):
        seg = []
        # bbox[] is x,y,w,h
        # left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        # left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        # right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        # right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
        area = bbox[2] * bbox[3]
        return seg, area

    def addAnnoItem(self, object_name, image_id, category_id, bbox):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg, area = self.get_segmentation_area(bbox)
        annotation_item['segmentation'].append(seg)
        annotation_item['area'] = area
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        ##bjq
        keypoints = [100, 200, 2, 510, 191, 2, 506, 191, 2, 512, 192, 2, 503, 192, 1, 515, 202, 2, 499, 202, 2, 524,
                     214, 2,
                     497, 215, 2, 516, 226, 2, 496, 224, 2, 511, 232, 2, 497, 230, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0]
        keypoints = np.asarray(keypoints)
        keypoints = keypoints.tolist()
        annotation_item['num_keypoints'] = int(len(keypoints) / 3)
        annotation_item['keypoints'] = keypoints
        self.coco['annotations'].append(annotation_item)

    def convert2coco(self, ann_dir):
        ann_path = os.path.join(ann_dir, "*.xml")
        xml_list = glob.glob(ann_path)
        self.parseXmlFiles(xml_list)

    def parseXmlFiles(self, xml_list):
        """
        ElementTree:

        :param xml_list:
        :return:
        """
        for xml_file in xml_list:
            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            print(xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            # elem is <folder>, <filename>, <size>, <object>
            for elem1 in root:
                elem1_tag = elem1.tag
                elem2_tag = None
                object_name = None

                if elem1.tag == 'folder':
                    continue

                if elem1.tag == 'filename':
                    file_name = elem1.text
                    if file_name in self.category_set:
                        raise Exception('file_name duplicated')

                # add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in self.image_set:
                        current_image_id = self.addImgItem(file_name, size)
                        print('add image_dict with {} and {}'.format(file_name, size))
                    else:
                        raise Exception('duplicated image_dict: {}'.format(file_name))
                        # subelem is <width>, <height>, <depth>, <name>, <bndbox>

                for elem2 in elem1:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None

                    elem2_tag = elem2.tag
                    if elem1_tag == 'object' and elem2_tag == 'name':
                        object_name = elem2.text
                        if object_name not in self.category_set:
                            current_category_id = self.addCatItem(object_name)
                        else:
                            current_category_id = self.category_set[object_name]

                    elif elem1_tag == 'size':
                        if size[elem2.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[elem2.tag] = int(elem2.text)

                    # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for elem3 in elem2:
                        if elem2_tag == 'bndbox':
                            if bndbox[elem3.tag] is not None:
                                raise Exception('xml structure corrupted at bndbox tag.')
                            bndbox[elem3.tag] = int(elem3.text)
                    if elem2_tag == 'keypoints':
                        keypoints = elem2.text

                    # only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        bbox = []
                        # x
                        bbox.append(bndbox['xmin'])
                        # y
                        bbox.append(bndbox['ymin'])
                        # w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        # h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        print('add annotation with {},{},{},{}'.format(object_name,
                                                                       current_image_id,
                                                                       current_category_id,
                                                                       bbox))
                        self.addAnnoItem(object_name, current_image_id, current_category_id, bbox)

    def save_json(self, json_file):
        json.dump(self.coco, open(json_file, 'w'))


if __name__ == '__main__':
    ann_dir = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations'  # 这是xml文件所在的地址
    json_file = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json'  # 这是你要生成的json文件
    VOC2coco = PascalVOC2coco()
    # VOC2coco.generate_dataset(ann_dir)  # 只需要改动这两个参数就行了
    # VOC2coco.save_json(json_file)
    xml_path = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations/000000.xml"
    with open(xml_path) as fd:  # 将XML文件装载到dict里面
        doc = xmltodict.parse(fd.read())
        print(doc)
