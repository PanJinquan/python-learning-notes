# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : cocoDemo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-07 16:33:01
"""
#%%
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from utils import image_processing,file_processing
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

#第1个label是背景
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
COCO_NAME = ['background','person', 'bicycle', 'car', 'motorcycle']

class COCO_Instances(object):
    def __init__(self, annFile, image_dir, COCO_NAME=None):
        '''
        :param annFile: path/to/coco_dataset/instances_xxx.json -> instances文件路径
        :param image_dir: path/to/coco_dataset/image_dir        -> 图片目录
        '''
        # initialize COCO api for instance annotations
        self.coco=COCO(annFile)
        self.imge_dir=image_dir
        # display COCO categories and supercategories
        catIds=self.coco.getCatIds()      #获得所有种类的id,
        cats = self.coco.loadCats(catIds) #获得所有超类
        # COCO数据集catIds有1-90，但coco_name只有80个，因此其下标不一致对应的
        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

        self.COCO_ID = [0]
        if COCO_NAME is None:
            self.COCO_ID+=self.coco.getCatIds()
            self.COCO_NAME=['background']
            self.COCO_NAME+=[cat['name'] for cat in cats] #注意coco数据集中，label的id从1开始
        else:
            self.COCO_ID+=self.coco.getCatIds(COCO_NAME)
            self.COCO_NAME=COCO_NAME
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
            catIds=[]
        else:
            catIds = self.coco.getCatIds(catNms=catNms)
            # imgIds = self.coco.getImgIds(catIds=catIds)
        # 获得同时满足catNms的图片
        imgIds = self.coco.getImgIds(catIds=catIds)
        print("COCO image num:{}".format(len(imgIds)))
        return imgIds

    def get_object_rects(self, imgIds, show=False):
        '''
        根据图像id,返回对应的目标检测框和标签名称
        :param imgIds:
        :param show:
        :return:
        '''
        info_list=[]
        img_anns = self.coco.loadImgs(imgIds)
        for img_ann in img_anns:
            annIds = self.coco.getAnnIds(imgIds=img_ann['id'], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            # 根据anns获得rects, category_id
            rects, category_id = self.get_anns_bbox_list(anns)
            if rects==[] and category_id==[]:
                continue
            # 将id映射为对应的名称
            names = self.category_id2name(category_id)
            file_name=img_ann['file_name']
            info={'file_name':file_name,"rects":rects,"label":names}
            info_list.append(info)
            if show:
                image_path = os.path.join(self.imge_dir,file_name )
                I = io.imread(image_path)
                image_processing.show_image_rects_text(file_name, I, rects, names)
        return info_list

    def get_anns_bbox_list(self,anns):
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


    def get_object_mask(self,imgIds, show=True):
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
            mask=self.annsToMask(anns)
            if show:
                # 为了方便显示，mask*50
                image_processing.cv_show_image("object_mask", mask*50)
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
                label = file_processing.encode_label(name_list=[name],name_table=self.COCO_NAME)[0]
            else:
                continue
            temp_mask = self.coco.annToMask(ann)
            if len(temp_mask.shape) < 3:
                mask[:, :] += (mask == 0) * (temp_mask * label)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(temp_mask, axis=2)) > 0) * label).astype(np.uint8)
        return mask

    def category_id2name(self, category_id_list):
        category_list = self.coco.loadCats(category_id_list)
        name_list=[cat['name'] for cat in category_list]
        return name_list

    def encode_info(self, info_list):
        '''
        编码，将string类型name字符信息编码成int类型的label信息，下标从0开始，编码关系由COCO_Name决定
        如根据COCO_Name，将['person', 'bicycle', 'car']编码为[0,1,2]
        :param info_list:
        :return:
        '''
        COCO_Label_list=[]
        for info in info_list:
            names=info['label']
            rects=info['rects']
            file_name=info['file_name']
            labels=file_processing.encode_label(names,self.COCO_NAME)
            info={"file_name":file_name,"rects":rects,"label":labels}
            COCO_Label_list.append(info)
        return COCO_Label_list


    def decode_info(self,info_list):
        '''
        解码，将int型label信息解码成对应的string类型的name名称信息，解码关系由COCO_Name决定
        如根据COCO_Name，将[0,1,2]解码成['person', 'bicycle', 'car']
        :param info_list:
        :return:
        '''
        COCO_Name_list=[]
        for info in info_list:
            label=info['label']
            rects=info['rects']
            file_name=info['file_name']
            names=file_processing.decode_label(label,self.COCO_NAME)
            info={"file_name":file_name,"rects":rects,"label":names}
            COCO_Name_list.append(info)
        return COCO_Name_list


    def write_info(self,save_dir,info_list):
        '''
        保存COCO的label，rects和file_name信息，类似于VOC数据集，每张图片的信息保存一个TXT文件，
        其中文件名为file_name.txt,文件内容格式为:
        label1 x y w h
        label2 x y w h
        :param save_dir:
        :param info_list:
        :return:
        '''
        for info in info_list:
            labels = info['label']
            rects = info['rects']
            file_name = info['file_name']
            if labels==[] or rects==[]:
                continue
            basename=os.path.basename(file_name)[:-len('.jpg')]
            file_path=os.path.join(save_dir,basename+".txt")
            content_list=[[l] + r for l,r in zip(labels,rects)]
            file_processing.write_data(file_path,content_list)


def label_test(image_dir, filename,class_names):
    basename = os.path.basename(filename)[:-len('.txt')]+".jpg"
    image_path=os.path.join(image_dir, basename)
    image=image_processing.read_image(image_path)
    data=file_processing.read_data(filename,split=" ")
    label_list,rect_list = file_processing.split_list(data, split_index=1)
    label_list = [l[0] for l in label_list]
    name_list=file_processing.decode_label(label_list,class_names)
    image_processing.show_image_rects_text("object2", image, rect_list, name_list)

if __name__=="__main__":
    image_dir = 'D:/BaiduNetdiskDownload/COCO dataset/annotations_trainval2014/images/train2014/'
    annFile = 'D:/BaiduNetdiskDownload/COCO dataset/annotations_trainval2014/annotations/instances_train2014.json'
    co=COCO_Instances(annFile, image_dir,COCO_NAME)
    # 获得所有图像id
    imgIds=co.getImgIds()
    test_imgIds=imgIds[15:20]

    # 显示目标的bboxes
    info_list=co.get_object_rects(test_imgIds, show=True)
    # 显示实例分割
    # co.get_object_instance(test_imgIds, show=True)
    # 显示语义分割的mask
    co.get_object_mask(test_imgIds,show=True)
    # label编码
    COCO_Label_list=co.encode_info(info_list)
    # label解码
    COCO_Name_list=co.decode_info(COCO_Label_list)
    # print("nums:{}".format(len(COCO_Name_list)))
    # 保存label等信息
    co.write_info(save_dir="./data/coco", info_list=COCO_Label_list)

    filename='./data/coco/COCO_train2014_000000283524.txt'
    label_test(image_dir, filename,class_names=COCO_NAME)