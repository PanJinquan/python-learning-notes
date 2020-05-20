# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-05-14 17:43:16
# --------------------------------------------------------
"""

import cv2
from utils import image_processing, file_processing


class OpenCVDNNTF(object):
    def __init__(self, model_path, pbtxt, class_names):
        self.class_names = class_names
        self.cvNet = cv2.dnn.readNetFromTensorflow(model_path, pbtxt)

    def forward(self, image):
        self.cvNet.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
        out = self.cvNet.forward()
        return out

    def detect(self, image, ishow=True):
        dets = self.forward(image)
        bboxes, scores, lables = self.post_process(image, dets)
        if ishow:
            self.show(image, bboxes, scores, lables)

    def post_process(self, image, dets):
        rows = image.shape[0]
        cols = image.shape[1]
        bboxes = []
        scores = []
        lables = []
        for detection in dets[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.3:
                object_cls = int(detection[1])
                xmin = detection[3] * cols
                ymin = detection[4] * rows
                xmax = detection[5] * cols
                ymax = detection[6] * rows
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (23, 230, 210), thickness=2)
                bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                scores.append(score)
                lables.append(object_cls)
        return bboxes, scores, lables

    def show(self, image, bboxes, scores, lables):
        boxes_name = ["{:.4f} {}".format(s, l) for s, l in zip(scores, lables)]
        image = image_processing.draw_image_bboxes_text(image, bboxes, boxes_name)
        # image_processing.cv_show_image("image", image)
        cv2.imshow("image", image)
        cv2.waitKey(0)

    def detect_image_dir(self, image_dir):
        image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])
        for image_path in image_list:
            image = cv2.imread(image_path)
            self.detect(image, ishow=True)


if __name__ == "__main__":
    model_path = 'pb/frozen_inference_graph.pb'
    pbtxt = 'pb/opencv_graph.pbtxt'
    class_names = ["background", "dog"]
    image_dir = "./person"
    image_path = 'person/1.jpg'
    image = cv2.imread(image_path)
    cvdnn = OpenCVDNNTF(model_path, pbtxt, class_names)
    # cvdnn.detect(image)
    cvdnn.detect_image_dir(image_dir)
