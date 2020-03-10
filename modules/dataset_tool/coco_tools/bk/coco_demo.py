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
from utils import image_processing
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

annFile='D:/BaiduNetdiskDownload/COCO dataset/annotations_trainval2014/annotations/instances_train2014.json'
#
# initialize COCO api for instance annotations
coco=COCO(annFile)
#
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
coco_names=[cat['name'] for cat in cats] #注意coco数据集中，label的id从1开始
print('COCO categories: \n{}\n'.format(' '.join(coco_names)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [379520])
img_anns = coco.loadImgs(imgIds)[0]

# 显示样图
# load and display image_dict
coco_img_dir='D:/BaiduNetdiskDownload/COCO dataset/train2014/'
image_path=os.path.join(coco_img_dir, img_anns['file_name'])
I = io.imread(image_path)
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image_dict
# I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

# 显示目标信息样例
# load and display object annotations
def get_anns_bbox_list(anns):
    bbox_list=[]
    category_id_list=[]
    for ann in anns:
        bbox=ann['bbox']# x,y,w,h
        id =ann['category_id']
        bbox_list.append(bbox)
        category_id_list.append(id)
    return bbox_list,category_id_list

def get_category_name(category_id_list):
    category_list = coco.loadCats(category_id_list)
    name_list = [cat['name'] for cat in category_list]
    return name_list
annIds = coco.getAnnIds(imgIds=img_anns['id'], catIds=catIds, iscrowd=None)
# annIds = coco.getAnnIds(imgIds=img_anns['id'], iscrowd=None)

anns = coco.loadAnns(annIds)
rects_list,category_id_list=get_anns_bbox_list(anns)
name_list=get_category_name(category_id_list)
image_processing.show_image_rects_text("object", I, rects_list, name_list)


# 显示实例分割样例
# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img_anns['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()
mask =coco.annToMask(anns[0])
image_processing.show_image("mask",mask)
# 显示人体关键点信息
# initialize COCO api for person keypoints annotations
annFile='D:/BaiduNetdiskDownload/COCO dataset/annotations_trainval2014/annotations/person_keypoints_train2014.json'
coco_kps=COCO(annFile)
#
# load and display keypoints annotations
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img_anns['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()

# 打印标题信息
# initialize COCO api for caption annotations
# annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
annFile='D:/BaiduNetdiskDownload/COCO dataset/annotations_trainval2014/annotations/captions_train2014.json'

coco_caps=COCO(annFile)
#
# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img_anns['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()