# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : main.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-09 18:06:47
"""
from utils import file_processing, image_processing


def decode_json(json_path, image_path):
    json_data = file_processing.read_json_data(json_path)
    bbox_list = json_data["bbox_id"]
    rgb_image = image_processing.read_image(image_path)
    # image_processing.show_image_boxes("image",image,bbox_id)
    boxes_name = [str(d) for d in bbox_list]
    print("processing image:image_path{},\nbbox_list{}".format(image_path,bbox_list))
    image_processing.show_image_bboxes_text(image_path, rgb_image, bbox_list, boxes_name, waitKey=0)


def batch_test(data_dir):
    image_list = file_processing.get_files_list(data_dir, postfix=["*.jpg"])
    for image_path in image_list:
        json_path = image_path[:-len("jpg")] + "json"
        decode_json(json_path, image_path)


if __name__ == "__main__":
    # json_path="./dataset/rec/0.json"
    # image_path="./dataset/rec/0.jpg"
    data_dir = "/media/dm/dm2/project/python-learning-notes/dataset/rec"
    batch_test(data_dir)
