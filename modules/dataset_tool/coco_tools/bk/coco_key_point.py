# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : coco_key_point.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-29 17:33:16
"""

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
from utils import image_processing

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
coco_root = "/media/dm/dm/project/dataset/coco/"
annFile = coco_root + 'annotations/instances_train2017.json'
#
# initialize COCO api for instance annotations
coco = COCO(annFile)
#
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
coco_names = [cat['name'] for cat in cats]  # 注意coco数据集中，label的id从1开始
print('COCO categories: \n{}\n'.format(' '.join(coco_names)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard']);
imgIds = coco.getImgIds(catIds=catIds);
imgIds = coco.getImgIds(imgIds=[379520])
img_anns = coco.loadImgs(imgIds)[0]

# 显示样图
# load and display image_dict
coco_img_dir = coco_root + 'images/train2017/'
image_path = os.path.join(coco_img_dir, img_anns['file_name'])
I = io.imread(image_path)

# 显示人体关键点信息
# initialize COCO api for person keypoints annotations
annFile = coco_root + '/annotations/person_keypoints_train2017.json'
coco_kps = COCO(annFile)
#
# load and display keypoints annotations
plt.imshow(I);
plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img_anns['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()

# 打印标题信息
# initialize COCO api for caption annotations
# annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
annFile = coco_root+'annotations/captions_train2014.json'

coco_caps = COCO(annFile)
#
# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img_anns['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I);
plt.axis('off');
plt.show()
