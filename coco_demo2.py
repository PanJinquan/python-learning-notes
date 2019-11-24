# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : cocoDemo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-07 16:33:01
"""
# %%
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from utils import image_processing, file_processing
import random
from modules.dataset_tool import pascal_voc, comment

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# 第1个label是背景
# COCO_NAME = ['background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
#                   'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#                   'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#                   'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
#                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                   'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
#                   'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
#                   'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#                   'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                   'teddy bear', 'hair drier', 'toothbrush']
# 如果不需要全部label,则可如下定义COCO_NAME，这时会自定获取满足该label的对象
# COCO_NAME = ['background', 'person', 'bicycle', 'car', 'motorcycle']
COCO_NAME = ['background', 'person']


class COCO_Instances(object):
    def __init__(self, annFile, image_dir, COCO_NAME=None):
        '''
        :param annFile: path/to/coco_dataset/instances_xxx.json -> instances文件路径
        :param image_dir: path/to/coco_dataset/image_dir        -> 图片目录
        '''
        # initialize COCO api for instance annotations
        self.coco = COCO(annFile)
        self.imge_dir = image_dir
        # display COCO categories and supercategories
        catIds = self.coco.getCatIds()  # 获得所有种类的id,
        cats = self.coco.loadCats(catIds)  # 获得所有超类
        # COCO数据集catIds有1-90，但coco_name只有80个，因此其下标不一致对应的
        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

        self.COCO_ID = [0]
        if COCO_NAME is None:
            self.COCO_ID += self.coco.getCatIds()
            self.COCO_NAME = ['background']
            self.COCO_NAME += [cat['name'] for cat in cats]  # 注意coco数据集中，label的id从1开始
        else:
            self.COCO_ID += self.coco.getCatIds(COCO_NAME)
            self.COCO_NAME = COCO_NAME
        print('COCO categories: \n{}\n'.format(' '.join(self.COCO_NAME)))

    def getImgIds(self, catNms=[]):
        '''
        根据label名称，查找同时满足catNms条件的图片id
        :param catNms: 过滤名称：catNms=['person', 'dog', 'skateboard']，当catNms=[]
        :return:
        '''
        #
        if catNms == []:
            # imgIds=[]
            catIds = []
        else:
            catIds = self.coco.getCatIds(catNms=catNms)
            # imgIds = self.coco.getImgIds(catIds=catIds)
        # 获得同时满足catNms的图片
        imgIds = self.coco.getImgIds(catIds=catIds)
        print("COCO image num:{}".format(len(imgIds)))
        return imgIds

    def get_object_rects(self, imgIds, minarea=None, show=False):
        '''
        根据图像id,返回对应的目标检测框和标签名称
        :param imgIds:
        :param show:
        :return:
        '''
        info_list = []
        img_anns = self.coco.loadImgs(imgIds)
        for img_ann in img_anns:
            annIds = self.coco.getAnnIds(imgIds=img_ann['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            # 根据anns获得rects, category_id
            rects, category_id = self.get_anns_bbox_list(anns)
            rects, category_id = self.filter_rects(rects, category_id, minAreaTH=200)
            if rects == [] and category_id == []:
                continue
            # 将id映射为对应的名称
            names = self.category_id2name(category_id)
            file_name = img_ann['file_name']
            info = {'file_name': file_name, "rects": rects, "label": names}
            info_list.append(info)
            if show:
                image_path = os.path.join(self.imge_dir, file_name)
                I = io.imread(image_path)
                image_processing.show_image_rects_text("image", I, rects, names)
        return info_list

    def get_anns_bbox_list(self, anns):
        bbox_list = []
        category_id_list = []
        for ann in anns:
            bbox = ann['bbox']  # x,y,w,h
            id = ann['category_id']
            if id not in self.COCO_ID:
                continue
            bbox_list.append(bbox)
            category_id_list.append(id)
        return bbox_list, category_id_list

    def get_anns_keypoints(self, anns):
        bbox_list = []
        category_id_list = []
        keypoints = []
        for ann in anns:
            bbox = ann['bbox']  # x,y,w,h
            id = ann['category_id']
            keypoint = ann['keypoints']
            keypoint = np.asarray(keypoint).reshape(17, 3)
            keypoint = keypoint[:, 0:2]
            if id not in self.COCO_ID:
                continue
            bbox_list.append(bbox)
            category_id_list.append(id)
            keypoints.append(keypoint)
        return bbox_list, category_id_list, keypoints

    def get_object_instance(self, imgIds, show=True):
        '''
        根据图像id，绘制分割图像
        :param imgIds:
        :param show:
        :return:
        '''
        # 根据图像id获得指定的anns
        img_anns = self.coco.loadImgs(imgIds)
        for img_ann in img_anns:
            # 根据图像id获得ann的id
            annIds = self.coco.getAnnIds(imgIds=img_ann['id'], iscrowd=None)
            # 根据ann的id获得anns
            anns = self.coco.loadAnns(annIds)
            if show:
                # show instance
                image_path = os.path.join(self.imge_dir, img_ann['file_name'])
                I = io.imread(image_path)
                plt.imshow(I), plt.axis('off')
                self.coco.showAnns(anns), plt.show()

    def get_object_mask(self, imgIds, show=True):
        '''
        根据图像id获得mask图
        :param imgIds:
        :param show:
        :return:
        '''
        # 根据图像id获得指定的anns
        img_anns = self.coco.loadImgs(imgIds)
        for img_ann in img_anns:
            # 根据图像id获得ann的id
            annIds = self.coco.getAnnIds(imgIds=img_ann['id'], iscrowd=None)
            # 根据ann的id获得anns
            anns = self.coco.loadAnns(annIds)
            # 根据anns获得mask
            mask = self.annsToMask(anns)
            if show:
                # 为了方便显示，mask*50
                image_processing.cv_show_image("object_mask", mask * 50)

    def annsToMask(self, anns):
        '''
        mask图像中像素0表示背景，像素1-80表示对应的label值，注意分割区域交叠部分会导致像素值超过80
        :param anns:
        :return:
        '''
        t = self.coco.loadImgs([anns[0]['image_id']])[0]
        h, w = t['height'], t['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            id = ann['category_id']
            name = self.category_id2name([id])[0]
            if id in self.COCO_ID and name in self.COCO_NAME:
                # label = self.COCO_ID.index(id)
                label = file_processing.encode_label(name_list=[name], name_table=self.COCO_NAME)[0]
            else:
                continue
            temp_mask = self.coco.annToMask(ann)
            if len(temp_mask.shape) < 3:
                mask[:, :] += (mask == 0) * (temp_mask * label)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(temp_mask, axis=2)) > 0) * label).astype(np.uint8)
        return mask

    def get_object_keypoints(self, imgIds, show=False):
        '''
        根据图像id,返回对应的目标检测框和标签名称
            "keypoints": {
                            0: "nose",
                            1: "left_eye",
                            2: "right_eye",
                            3: "left_ear",
                            4: "right_ear",
                            5: "left_shoulder",
                            6: "right_shoulder",
                            7: "left_elbow",
                            8: "right_elbow",
                            9: "left_wrist",
                            10: "right_wrist",
                            11: "left_hip",
                            12: "right_hip",
                            13: "left_knee",
                            14: "right_knee",
                            15: "left_ankle",
                            16: "right_ankle"
                            },
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
            :param imgIds:
            :param show:
        :return:
        '''
        info_list = []
        img_anns = self.coco.loadImgs(imgIds)
        for img_ann in img_anns:
            annIds = self.coco.getAnnIds(imgIds=img_ann['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            # 根据anns获得rects, category_id
            rects, category_id, keypoints = self.get_anns_keypoints(anns)
            if rects == [] and category_id == []:
                continue
            # 将id映射为对应的名称
            names = self.category_id2name(category_id)
            file_name = img_ann['file_name']
            info = {'file_name': file_name, "rects": rects, "label": names}
            info_list.append(info)
            if show:
                skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
                skeleton = np.asarray(skeleton) - 1
                image_path = os.path.join(self.imge_dir, file_name)
                I = io.imread(image_path)
                I = image_processing.draw_key_point_in_image(I, keypoints, pointline=skeleton)
                image_processing.show_image_rects_text("image", I, rects, names)
        return info_list

    def category_id2name(self, category_id_list):
        category_list = self.coco.loadCats(category_id_list)
        name_list = [cat['name'] for cat in category_list]
        return name_list

    def encode_info(self, info_list):
        '''
        编码，将string类型name字符信息编码成int类型的label信息，下标从0开始，编码关系由COCO_Name决定
        如根据COCO_Name，将['person', 'bicycle', 'car']编码为[0,1,2]
        :param info_list:
        :return:
        '''
        COCO_Label_list = []
        for info in info_list:
            names = info['label']
            rects = info['rects']
            file_name = info['file_name']
            labels = file_processing.encode_label(names, self.COCO_NAME)
            info = {"file_name": file_name, "rects": rects, "label": labels}
            COCO_Label_list.append(info)
        return COCO_Label_list

    def decode_info(self, info_list):
        '''
        解码，将int型label信息解码成对应的string类型的name名称信息，解码关系由COCO_Name决定
        如根据COCO_Name，将[0,1,2]解码成['person', 'bicycle', 'car']
        :param info_list:
        :return:
        '''
        COCO_Name_list = []
        for info in info_list:
            label = info['label']
            rects = info['rects']
            file_name = info['file_name']
            names = file_processing.decode_label(label, self.COCO_NAME)
            info = {"file_name": file_name, "rects": rects, "label": names}
            COCO_Name_list.append(info)
        return COCO_Name_list

    def filter_rects(self, rects, labels, minAreaTH=0):
        out_labels, out_rects = [], []
        for l, r in zip(labels, rects):
            x, y, w, h = r
            area = w * h
            if minAreaTH > 0 and area < minAreaTH:
                print("area< minAreaTH:{},{}<{}".format(r,area, minAreaTH))
                continue
            out_labels.append(l)
            out_rects.append(r)
        return out_rects, out_labels

    def write_info(self, save_dir, info_list):
        '''
        保存COCO的label，rects和file_name信息，类似于VOC数据集，每张图片的信息保存一个TXT文件，
        其中文件名为file_name.txt,文件内容格式为:
        label1 x y w h
        label2 x y w h
        :param save_dir:
        :param info_list:
        :return:
        '''
        file_processing.create_dir(save_dir)
        image_id = []
        for info in info_list:
            labels = info['label']
            rects = info['rects']
            file_name = info['file_name']
            if labels == [] or rects == []:
                continue
            basename = os.path.basename(file_name)[:-len('.jpg')]
            file_path = os.path.join(save_dir, basename + ".txt")
            content_list = [[l] + r for l, r in zip(labels, rects)]
            file_processing.write_data(file_path, content_list)
            image_id.append(basename)
        return image_id

    def save_train_val(self, info_list, out_train_val_path, label_out_dir, shuffle=True):
        if shuffle:
            seeds = 100  # 固定种子,只要seed的值一样，后续生成的随机数都一样
            random.seed(seeds)
            random.shuffle(info_list)
        # 分割成train和val数据集
        factor = 0.95
        train_num = int(factor * len(info_list))
        train_image_list = info_list[:train_num]
        val_image_list = info_list[train_num:]

        print("train_image_list:{}".format(len(train_image_list)))
        print("val_image_list  :{}".format(len(val_image_list)))
        train_image_id = self.write_info(label_out_dir, train_image_list)
        val_image_id = self.write_info(label_out_dir, val_image_list)
        # 保存图片id数据
        train_id_path = os.path.join(out_train_val_path, "train.txt")
        val_id_path = os.path.join(out_train_val_path, "val.txt")
        comment.save_id(train_id_path, train_image_id, val_id_path, val_image_id)


def label_test(image_dir, filename, class_names=None):
    basename = os.path.basename(filename)[:-len('.txt')] + ".jpg"
    image_path = os.path.join(image_dir, basename)
    image = image_processing.read_image(image_path)
    data = file_processing.read_data(filename, split=" ")
    label_list, rect_list = file_processing.split_list(data, split_index=1)
    label_list = [l[0] for l in label_list]
    if class_names:
        name_list = file_processing.decode_label(label_list, class_names)
    else:
        name_list = label_list
    show_info = ["id:" + str(n) for n in name_list]
    rgb_image = image_processing.show_image_rects_text("object2", image, rect_list, show_info, color=(0, 0, 255),
                                                       drawType="custom", waitKey=1)
    rgb_image = image_processing.resize_image(rgb_image, 900)
    image_processing.cv_show_image("object2", rgb_image)


def batch_label_test(label_dir, image_dir, classes):
    file_list = file_processing.get_files_list(label_dir, postfix=[".txt"])
    for filename in file_list:
        label_test(image_dir, filename, class_names=classes)


if __name__ == "__main__":
    coco_root = "/media/dm/dm/project/dataset/coco/"
    image_dir = coco_root + 'images/train2017/'
    # image_dir = coco_root + 'images/val2017/'
    # annFile = coco_root + 'annotations/instances_train2017.json'
    # annFile = coco_root + 'annotations/person_keypoints_train2017.json'
    annFile = coco_root + 'annotations/person_keypoints_train2017.json'
    # annFile = coco_root + 'annotations/person_keypoints_val2017.json'
    label_out_dir = coco_root + "label"

    co = COCO_Instances(annFile, image_dir, COCO_NAME)
    # 获得所有图像id
    imgIds = co.getImgIds()
    # test_imgIds = imgIds[0:20]
    test_imgIds = imgIds

    # 显示目标的bboxes
    info_list = co.get_object_rects(test_imgIds, show=False)
    # info_list = co.get_object_keypoints(test_imgIds, show=True)
    # 显示实例分割
    # co.get_object_instance(test_imgIds, show=True)
    # 显示语义分割的mask
    # co.get_object_mask(test_imgIds, show=True)
    # label编码
    info_list = co.encode_info(info_list)
    # label解码
    # info_list = co.decode_info(info_list)
    # print("nums:{}".format(len(info_list)))
    # 保存label等信息
    # co.write_info(save_dir="./data/coco", info_list=COCO_Label_list)
    co.save_train_val(info_list, coco_root, label_out_dir)
    batch_label_test(label_out_dir, image_dir, classes=COCO_NAME)
