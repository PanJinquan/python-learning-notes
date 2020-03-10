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
https://blog.csdn.net/wc781708249/article/details/79615210
"""
import json
import cv2
import numpy as np
import glob
import PIL.Image
import os, sys


class PascalVOC2coco(object):
    def __init__(self, anno_dir, image_dir=None, seg_dir=None, save_json_path='./new.json'):
        """

        :param anno_dir:  for voc `Annotations`
        :param image_dir: for voc `JPEGImages`
        :param seg_dir:   for voc `SegmentationObject`
        :param save_json_path:
        """
        self.anno_dir = anno_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        filelist_xml = file_processing.get_files_list(anno_dir, postfix=["*.xml"])
        self.filelist_xml = filelist_xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []

        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.data_transfer(self.filelist_xml)
        # 保存json文件
        self.save_json(self.save_json_path)

    def data_transfer(self, filelist_xml):
        height = None
        width = None
        filename = ""
        for num, ann_file in enumerate(filelist_xml):
            self.xml_file = ann_file
            self.num = num
            path = os.path.dirname(self.xml_file)
            path = os.path.dirname(path)
            # path=os.path.split(self.ann_file)[0]
            # path=os.path.split(path)[0]
            with open(ann_file, 'r') as fp:
                for p in fp:
                    # if 'folder' in p:
                    #     folder =p.split('>')[1].split('<')[0]
                    if 'filename' in p:
                        filename = p.split('>')[1].split('<')[0]

                    if 'width' in p:
                        width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        height = int(p.split('>')[1].split('<')[0])

                    if width and height and filename:
                        self.images.append(self.image_dict(height, width, filename))
                        height = None
                        width = None

                    if '<object>' in p:
                        # 类别
                        d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                        self.supercategory = d[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)

                        # 边界框
                        x1 = int(d[-4])
                        y1 = int(d[-3])
                        x2 = int(d[-2])
                        y2 = int(d[-1])
                        rectangle = [x1, y1, x2, y2]
                        self.annotations.append(self.annotation(filename, rectangle, self.annID))
                        self.annID += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def image_dict(self, height, width, filename):
        image = {}
        image['height'] = height
        image['width'] = width
        image['id'] = self.num + 1
        image['file_name'] = filename
        return image

    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie

    @staticmethod
    def change_format(contour):
        contour2 = []
        length = len(contour)
        for i in range(0, length, 2):
            contour2.append([contour[i], contour[i + 1]])
        return np.asarray(contour2, np.int32)

    def annotation(self, filename, rectangle, annID):
        x1, y1, x2, y2 = rectangle
        bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO 对应格式[x,y,w,h]
        annotation = {}
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        # annotation['bbox'] = list(map(float, self.bbox))
        annotation['bbox'] = bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = annID

        # 计算轮廓面积
        seg = []
        if self.seg_dir:
            seg = self.getsegmentation(filename, rectangle)
        if seg:
            seg = [list(map(float, seg))]
            contour = PascalVOC2coco.change_format(seg[0])
            area = abs(cv2.contourArea(contour, True))
        else:
            seg = []
            area = 0
        annotation['segmentation'] = seg
        annotation['area'] = area
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getsegmentation(self, filename, rectangle):
        seg_path = os.path.join(self.seg_dir, filename.split('.')[0] + '.png')
        if not os.path.exists(seg_path):
            return []
        try:
            mask_1 = cv2.imread(seg_path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]

            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2

            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))

            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            return self.mask2polygons(mask)
        except:
            return []

    def mask2polygons(self, mask):
        '''从mask提取边界点'''
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox = []
        for cont in contours[0]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox  # list(contours[1][0].flatten())

    def getbbox(self, height, width, points):
        '''边界点生成mask，从mask提取定位框'''
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([height, width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        '''边界点生成mask'''
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def save_json(self, save_json_path):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        json.dump(data_coco, open(save_json_path, 'w'), indent=4)  # indent=4 更加美观显示

    @staticmethod
    def merge_coco_dataset(file_list, save_json_path):
        data_coco = {}
        for file in file_list:
            coco = file_processing.read_json_data(file)
            if not data_coco:
                data_coco = coco
                continue
            data_coco['images']+=coco['images']
            data_coco['categories']+=coco['categories']
            data_coco['annotations']+=coco['annotations']
        json.dump(data_coco, open(save_json_path, 'w'), indent=4)  # indent=4 更加美观显示


if __name__ == "__main__":
    """
    https://blog.csdn.net/wc781708249/article/details/79615210
    """
    # xml_file = glob.glob('./Annotations/*.xml')
    # xml_file=['./Annotations/000032.xml']

    # PascalVOC2coco(xml_file, './new.json')

    from utils import file_processing

    # anno_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations"
    # json_file = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json'  # 这是你要生成的json文件
    # seg_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/SegmentationObject"

    anno_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/Annotations"
    json_file = '/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json'  # 这是你要生成的json文件
    seg_dir = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/SegmentationObject"

    PascalVOC2coco(anno_dir, seg_dir=seg_dir, save_json_path=json_file)
    #
    # file_list = ["/media/dm/dm1/git/python-learning-notes/dataset/VOC/teacher_coco.json",
    #              "/media/dm/dm1/git/python-learning-notes/dataset/VOC/voc2coco2.json"]
    # save_json_path = "/media/dm/dm1/git/python-learning-notes/dataset/VOC/merge_coco_dataset.json"
    # PascalVOC2coco.merge_coco_dataset(file_list, save_json_path)
