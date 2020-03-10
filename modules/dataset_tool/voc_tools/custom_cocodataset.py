# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: AlphaPose
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-14 09:15:52
# --------------------------------------------------------
"""
import os
import numpy as np
import argparse
from utils import file_processing, image_processing
from modules.dataset_tool.voc_tools import build_voc

# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
#             [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# skeleton = np.asarray(skeleton) - 1

# skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
#             (8, 10), (11, 13), (12, 14), (13, 15), (14, 16), (11, 17), (12, 17)]

skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (11, 13), (12, 14), (13, 15), (14, 16), (11, 18), (12, 18), (18, 17)]


def cal_iou(box1, box2):
    """
    computing IoU
    :param box1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param box2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    cx1, cy1, cx2, cy2 = box1
    gx1, gy1, gx2, gy2 = box2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def cal_iou_list(box1, box2_list):
    iou_list = []
    for box2 in box2_list:
        iou = cal_iou(box1, box2)
        iou_list.append(iou)
    return iou_list


class CustomDataset():
    def __init__(self, anno_dir, image_dir, label_file):
        self.json_dir = anno_dir
        self.label_file = label_file
        self.image_dir = image_dir
        # json_dir = os.path.join(anno_dir, "json")
        self.boxes_label_lists = self.read_label(self.label_file)

    @staticmethod
    def get_anno(json_file):
        anno = file_processing.read_json_data(json_file)
        return anno

    def get_keypoints(self, anno, nums_joints=17):
        keypoints = []
        for a in anno:
            landmarks = a["label"]["landmark"]
            joints = []
            for i, l in enumerate(landmarks):
                point = [l[str(i)][str(i)]["x"], l[str(i)][str(i)]["y"]]
                if point[0] == -1 or point[1] == -1:
                    point = [0, 0]
                joints.append(point)
            keypoints.append(joints)
        return keypoints

    @staticmethod
    def read_label(filename):
        boxes_label_lists = file_processing.read_data(filename)
        return boxes_label_lists

    def get_joints_bbox(self, keypoints):
        joints_bbox = []
        for joints in keypoints:
            joints = [j for j in joints if not (j == [0, 0] or j == [-1, -1])]
            joints = np.asarray(joints)
            xmin = min(joints[:, 0])
            ymin = min(joints[:, 1])
            xmax = max(joints[:, 0])
            ymax = max(joints[:, 1])
            joints_bbox.append([xmin, ymin, xmax, ymax])

        return joints_bbox

    def match_keypoints(self, target_bbox, joints_bbox):
        iou_list = cal_iou_list(target_bbox, joints_bbox)
        index = np.argmax(iou_list)
        return index

    def flat_joints(self, joints):
        """
        :param joints:
        :return:
            keypoint关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
            其中x,y为Keypoint的坐标，v为可见标志
                v = 0 : 未标注点
                v = 1 : 标注了但是图像中不可见（例如遮挡）
                v = 2 : 标注了并图像可见
        """
        keypoint = []
        for p in joints:
            x, y = p
            v = 2
            if x == -1 or y == -1:
                v = 0
            if x == 0 and 0 == 0:
                v = 0
            keypoint += [x, y, v]
        return keypoint

    def convert_voc_dataset(self, filename, image_shape, bboxes, labels, keypoints, out_voc_ann):
        """
        bbox和keypoints并不是一一对应的,需要根据IOU进行匹配
        :param filename:
        :param image_shape:
        :param bboxes:[xmin,ymin,xmax,ymax]
        :param labels:
        :param keypoints:
        :param out_voc_ann:
        :return:
        """
        if not os.path.exists(out_voc_ann):
            os.makedirs(out_voc_ann)
        objects = []
        # 根据关节点获得轮廓的bbox
        joints_bbox = self.get_joints_bbox(keypoints)
        for bbox, name in zip(bboxes, labels):
            # 匹配bbox和对应的keypoint
            index = self.match_keypoints(bbox, joints_bbox)
            joints = keypoints[index]
            keypoint = self.flat_joints(joints)
            object = build_voc.create_object(name=name, bndbox=bbox, keypoint=keypoint)
            objects.append(object)

        id = filename[:-len(".jpg")]
        xml_path = os.path.join(out_voc_ann, "{}.xml".format(id))
        build_voc.covert_voc_xml(image_shape, filename, xml_path, objects)
        return joints_bbox

    def deme_test(self, out_voc_ann=None, vis=True):
        joints_bbox = []
        for lines in self.boxes_label_lists:
            # lines = ['lexue_teacher_LHui211_20190921145000_20190921172000_40_000143.jpg', 543.053254438, 527.147928994,
            #          456.710059172, 548.733727811, 'person']
            image_name = lines[0]
            rect = [lines[1], lines[2], lines[3], lines[4]]
            label = [lines[5]]
            image_path = os.path.join(self.image_dir, image_name)
            json_file = image_name[:-len(".jpg")] + ".json"
            json_path = os.path.join(self.json_dir, json_file)
            if not os.path.exists(image_path):
                print("Error:no path: {}".format(json_file))
                continue
            if not os.path.exists(json_path):
                print("Error:no path: {}".format(json_file))
                continue

            anno = CustomDataset.get_anno(json_path)
            if not anno:
                print("Error:empty path: {}".format(json_file))
                continue
            keypoints = self.get_keypoints(anno)
            sum = np.sum(np.abs(np.asarray(keypoints)))
            if keypoints == [] or sum == 0:
                print("Error:empty path: {}".format(json_file))
                continue

            image = image_processing.read_image(image_path)
            if out_voc_ann:
                try:
                    bboxes = image_processing.rects2bboxes([rect])
                    joints_bbox = self.convert_voc_dataset(image_name,
                                                           image_shape=image.shape,
                                                           bboxes=bboxes,
                                                           labels=label,
                                                           keypoints=keypoints,
                                                           out_voc_ann=out_voc_ann)
                except Exception as e:
                    print("Error:empty path: {}".format(json_file))
                    raise Exception("lines: {}".format(lines))

            save_image = True
            for i, joints in enumerate(keypoints):
                if np.sum(np.asarray(joints[5])) == 0 or np.sum(np.asarray(joints[6])) == 0 or \
                        np.sum(np.asarray(joints[11])) == 0 or np.sum(np.asarray(joints[12]))==0:
                    save_image = False
                else:
                    save_image = True
                chest_joint = (np.asarray(joints[5]) + np.asarray(joints[6])) / 2
                hip_joint = (np.asarray(joints[11]) + np.asarray(joints[12])) / 2
                keypoints[i].append(chest_joint.tolist())
                keypoints[i].append(hip_joint.tolist())

            if vis:
                image_processing.show_image_rects(None, image, [rect])
                # image_processing.show_image_boxes(None, image, joints_bbox, color=(255, 0, 0))
                image = image_processing.draw_key_point_in_image(image, keypoints, pointline=skeleton)
                image_processing.cv_show_image("Det", image, waitKey=1)
                if save_image:
                    out_dir = "/media/dm/dm2/project/dataset/COCO/HumanPose/LeXue_teacher/Posture/tmp1"
                    out_dir = file_processing.create_dir(out_dir)
                    out_image_path = os.path.join(out_dir, image_name)
                    image_processing.save_image(out_image_path, image)


parser = argparse.ArgumentParser(description="COCO Dataset")
parser.add_argument("-i", "--image_dir", help="path/to/image", type=str)
parser.add_argument("-a", "--anno_dir", help="path/to/anno_dir", type=str)
parser.add_argument("-l", "--label_file", help="path/to/label_file", type=str)
parser.add_argument("-o", "--out_voc_ann", help="out/to/out_voc_ann", type=str)
parser.add_argument("-v", "--vis", help="show image", default=True, type=str)
args = parser.parse_args()

if __name__ == "__main__":
    image_dir = args.image_dir
    anno_dir = args.anno_dir
    label_file = args.label_file
    out_voc_ann = args.out_voc_ann
    vis = args.vis

    image_dir = "/media/dm/dm2/project/dataset/COCO/HumanPose/LeXue_teacher/images/1"
    anno_dir = "/media/dm/dm2/project/dataset/COCO/HumanPose/LeXue_teacher/annotations/src/1"
    label_file = "/media/dm/dm2/project/dataset/COCO/HumanPose/LeXue_teacher/annotations/src/1_bbox.txt"
    out_voc_ann = "/media/dm/dm2/project/dataset/COCO/HumanPose/LeXue_teacher/annotations/voc/1"
    #
    # image_dir = "/media/dm/dm/project/dataset/coco/images/lexue_train"
    # anno_dir = "/media/dm/dm/project/dataset/coco/annotations/KeyPointsTeacher-label0220/lexue_train"
    # label_file = "/media/dm/dm/project/dataset/coco/annotations/KeyPointsTeacher-label0220/lexue_train.txt"
    # out_voc_ann = "/media/dm/dm/project/dataset/coco/annotations/KeyPointsTeacher-label0220/annotations/lexue_train"

    cd = CustomDataset(anno_dir, image_dir, label_file)
    cd.deme_test(out_voc_ann=out_voc_ann, vis=False)
