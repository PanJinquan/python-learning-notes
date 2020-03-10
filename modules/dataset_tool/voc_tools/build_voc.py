# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-12 18:54:28
# --------------------------------------------------------
"""
import cv2
import os
import codecs


def covert_voc_xml(image_shape, filename, xml_path, objects: list):
    """
    :param image_shape:image_dict.shape
    :param filename: file name
    :param xml_path: save Annotations(*.xml) file path
    :param objects: [object] ,object= {
                                    "name": name,
                                    "bndbox": bndbox,
                                    "keypoints": keypoint
                                     }
            - name: bbox label name
            - bndbox: bbox =[x_min, y_min, x_max, y_max]
            - keypoint: [x_1, y_1, v_1,...,x_k, y_k, v_k],
                    其中x,y为Keypoint的坐标，v为可见标志
                    v = 0 : 未标注点
                    v = 1 : 标注了但是图像中不可见（例如遮挡）
                    v = 2 : 标注了并图像可见
    :return:
    """
    height, width, depth = image_shape
    xml = codecs.open(xml_path, 'w', encoding='utf-8')
    xml.write('<annotation>\n')
    xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
    xml.write('\t<filename>' + filename + '</filename>\n')
    xml.write('\t<source>\n')
    xml.write('\t\t<database>The VOC2007 Database</database>\n')
    xml.write('\t\t<annotation>PASCAL VOC2007</annotation>\n')
    xml.write('\t\t<image_dict>flickr</image_dict>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t</source>\n')
    xml.write('\t<owner>\n')
    xml.write('\t\t<flickrid>NULL</flickrid>\n')
    xml.write('\t\t<name>pjq</name>\n')
    xml.write('\t</owner>\n')
    xml.write('\t<size>\n')
    xml.write('\t\t<width>' + str(width) + '</width>\n')
    xml.write('\t\t<height>' + str(height) + '</height>\n')
    xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
    xml.write('\t</size>\n')
    xml.write('\t\t<segmented>0</segmented>\n')
    for o in objects:
        name = o["name"]
        x_min, y_min, x_max, y_max = o["bndbox"]
        keypoints = o["keypoints"]
        xml.write('\t<object>\n')
        xml.write('\t\t<name>{}</name>\n'.format(name))
        xml.write('\t\t<pose>Unspecified</pose>\n')
        xml.write('\t\t<truncated>0</truncated>\n')
        xml.write('\t\t<difficult>0</difficult>\n')
        xml.write('\t\t<bndbox>\n')
        xml.write('\t\t\t<xmin>' + str(x_min) + '</xmin>\n')
        xml.write('\t\t\t<ymin>' + str(y_min) + '</ymin>\n')
        xml.write('\t\t\t<xmax>' + str(x_max) + '</xmax>\n')
        xml.write('\t\t\t<ymax>' + str(y_max) + '</ymax>\n')
        xml.write('\t\t</bndbox>\n')
        if keypoints:
            add_keypoints(xml, keypoints)
        xml.write('\t</object>\n')
    xml.write('</annotation>')


def add_keypoints(xml, keypoint: list):
    keypoint = [str(i) for i in keypoint]
    keypoint = ",".join(keypoint)
    xml.write('\t\t<keypoints>{}</keypoints>\n'.format(keypoint))


def create_object(name, bndbox, keypoint=[]):
    object = {
        "name": name,
        "bndbox": bndbox,
        "keypoints": keypoint}
    return object


def create_voc_demo(image_path, out_anno_dir):
    image = cv2.imread(image_path)
    image_shape = image.shape
    filename = os.path.basename(image_path)
    id = filename[:-len(".jpg")]
    xml_path = os.path.join(out_anno_dir, "{}.xml".format(id))
    keypoint = [100, 200, 2, 510, 191, 2, 506, 191, 2, 512, 192, 2, 503, 192, 1, 515, 202, 2,
                499, 202, 2, 524, 214, 2, 497, 215, 2, 516, 226, 2, 496, 224, 2, 511, 232, 2,
                497, 230, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    keypoint = []
    object1 = create_object(name="dog", bndbox=[48, 240, 195, 371], keypoint=keypoint)
    object2 = create_object(name="person", bndbox=[8, 12, 352, 498], keypoint=keypoint)
    objects = []
    objects.append(object1)
    objects.append(object2)
    covert_voc_xml(image_shape, filename, xml_path, objects)


if __name__ == "__main__":
    label = 1
    out_anno_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations"
    image_path = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/JPEGImages/000000.jpg"
    create_voc_demo(image_path, out_anno_dir)
