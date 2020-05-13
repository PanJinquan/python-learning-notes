# -*-coding: utf-8 -*-

import numpy as np
import cv2
import PIL.Image as Image
from net.mtcnn import MTCNN


def show_landmark_boxes(win_name, image, landmarks_list, boxes):
    '''
    显示landmark和boxes
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :param boxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    for landmarks in landmarks_list:
        for landmark in landmarks:
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, point_color, thickness)
    show_image_boxes(win_name, image, boxes)


def show_image_boxes(win_name, image, boxes_list):
    '''
    :param win_name:
    :param image:
    :param boxes_list:[[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    for box in boxes_list:
        x1, y1, x2, y2 = box
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    image = Image.fromarray(image)
    image.show(win_name)


if __name__ == "__main__":
    image_path = "./test.jpg"
    image = Image.open(image_path)
    min_face_size = 20.0
    thresholds = [0.6, 0.7, 0.8]
    nms_thresholds = [0.7, 0.7, 0.7]
    device = "cuda:0"
    pnet_path = './X2-face-detection-pnet.pth.tar'
    rnet_path = './X2-face-detection-rnet.pth.tar'
    onet_path = './X2-face-detection-onet.pth.tar'
    # init MTCNN
    mt = MTCNN(pnet_path, rnet_path, onet_path).to(device)
    # forward MTCNN and get bbox_score, bounding boxes
    bbox_score, landmarks = mt.forward(image, min_face_size, thresholds, nms_thresholds)
    bboxes = bbox_score[:, :4]
    scores = bbox_score[:, 4:]
    # show bbox_score and bounding boxes
    print("bbox_score:\n{}\nlandmarks:\n{}".format(bbox_score, landmarks))
    show_landmark_boxes("image", np.array(image), landmarks, bboxes)
