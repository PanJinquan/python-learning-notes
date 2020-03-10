# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : face_eval.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-10 09:46:40
"""
import numpy
import numpy as np
import torch
import os
import itertools
from custom_insightFace.tools import image_processing, file_processing
from custom_insightFace.eval.iou import cal_iou_list
from custom_insightFace.core import faceRecognition
from custom_insightFace.eval import pr


def cal_face_recognition(pred_bboxes, pred_labels, true_bboxes, true_labels, iou_threshold, score_threshold):
    '''
    iou_mat shape=(num_pred_bboxes,num_true_bboxes)
    :param pred_bboxes:
    :param pred_labels:
    :param true_bboxes:
    :param true_labels:
    :param iou_threshold:
    :param score_threshold:
    :return:
    '''
    num_pred_bboxes = len(pred_bboxes)
    num_true_bboxes = len(true_bboxes)
    iou_mat = []
    for pred_bbox in pred_bboxes:
        iou = cal_iou_list(pred_bbox, true_bboxes)
        iou_mat.append(iou)
    iou_mat = np.asarray(iou_mat)
    # print(iou_mat)
    max_index = np.argmax(iou_mat, axis=1)
    max_iou = np.max(iou_mat, axis=1)
    # print(max_index)
    # print(max_iou)
    _true_labels = np.asarray(true_labels)[max_index]
    # print(_true_labels)
    tp = get_tp(pred_labels, _true_labels.tolist(), max_iou, iou_threshold=iou_threshold)
    fp = num_pred_bboxes - tp
    precision = tp / num_pred_bboxes
    recall = tp / num_true_bboxes
    print("precision:{}".format(precision))
    print("recall   :{}".format(recall))
    return precision, recall


def get_face_precision_recall_acc(true_labels, pred_labels, average="binary"):
    recision, recall, acc = pr.get_precision_recall_acc(true_labels, pred_labels, average)
    return recision, recall, acc


def get_tp(pred_labels, true_labels, iou_list, iou_threshold):
    assert isinstance(pred_labels, list), "must be list"
    assert isinstance(true_labels, list), "must be list"
    tp = 0
    for iou, pred_label, true_label in zip(iou_list, pred_labels, true_labels):
        if iou > iou_threshold and pred_label == true_label:
            tp += 1
    return tp


def get_pair_image_scores(faces_list1, faces_list2, issames_data, model_path, conf, image_dir=None, save_path=None):
    '''
    计算分数
    :param faces_data:
    :param issames_data:
    :param model_path: insightFace模型路径
    :param conf:
    :param save_path:
    :return:
    '''
    faceRec = faceRecognition.FaceRecognition(model_path, conf)
    pred_score = []
    i = 0
    image_size = [112, 112]
    for face1_path, face2_path, issame in zip(faces_list1, faces_list2, issames_data):
        # pred_id, pred_scores = faceRec.predict(faces)
        # 或者使用get_faces_embedding()获得embedding，再比较compare_embedding()
        if image_dir:
            face1_path = os.path.join(image_dir, face1_path)
            face2_path = os.path.join(image_dir, face2_path)
        face1 = image_processing.read_image_gbk(face1_path)
        face2 = image_processing.read_image_gbk(face2_path)
        face1 = image_processing.resize_image(face1, resize_height=image_size[0], resize_width=image_size[1])
        face2 = image_processing.resize_image(face2, resize_height=image_size[0], resize_width=image_size[1])

        face_emb1 = faceRec.get_faces_embedding([face1])
        face_emb2 = faceRec.get_faces_embedding([face2])
        diff = face_emb1 - face_emb2
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        dist = dist.detach().cpu().numpy()
        pred_score.append(dist)
        i += 1
        if i % 100 == 0:
            print('processing data :', i)
    pred_score = np.array(pred_score).reshape(-1)
    if not isinstance(issames_data, numpy.ndarray):
        issames_data = np.asarray(issames_data)
    issames_data = issames_data  # 将true和false转为1/0
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        issames_path = os.path.join(save_path, "issames.npy")
        pred_score_path = os.path.join(save_path, "pred_score.npy")
        np.save(issames_path, issames_data)
        np.save(pred_score_path, pred_score)
    return pred_score, issames_data


def load_npy(dir_path):
    issames_path = os.path.join(dir_path, "issames.npy")
    pred_score_path = os.path.join(dir_path, "pred_score.npy")
    issames = np.load(issames_path)
    pred_score = np.load(pred_score_path)
    return pred_score, issames


def read_pair_data(filename, split=True):
    content_list = file_processing.read_data(filename)
    if split:
        content_list = np.asarray(content_list)
        faces_list1 = content_list[:, :1].reshape(-1)
        faces_list2 = content_list[:, 1:2].reshape(-1)
        # convert to 0/1
        issames_data = np.asarray(content_list[:, 2:3].reshape(-1), dtype=np.int)
        issames_data = np.where(issames_data > 0, 1, 0)
        faces_list1 = faces_list1.tolist()
        faces_list2 = faces_list2.tolist()
        issames_data = issames_data.tolist()
        return faces_list1, faces_list2, issames_data
    return content_list


def get_combinations_pair_data(image_dir):
    '''
    get image_dir image_dict list,combinations image_dict
    :param image_dir:
    :return:
    '''
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])

    pair_issame = []
    for paths in itertools.combinations(image_list, 2):
        image_path1, image_path2 = paths
        name1 = os.path.basename(image_path1)
        name2 = os.path.basename(image_path2)

        label1 = image_path1.split(os.sep)[-2]
        label2 = image_path2.split(os.sep)[-2]
        if label1 == label2:
            issame = 1
        else:
            issame = 0
        image_id1 = os.path.join(label1, name1)
        image_id2 = os.path.join(label2, name2)
        # pair_issame.append([image_id1, image_id2, issame])
        pair_issame.append([image_path1, image_path2, issame])
    pair_issame = np.asarray(pair_issame)
    pair_issame = pair_issame[np.lexsort(pair_issame.T)]
    pair_issame_0 = pair_issame[pair_issame[:, -1] == "0", :]
    pair_issame_1 = pair_issame[pair_issame[:, -1] == "1", :]
    num_pair_issame_1 = len(pair_issame_1)
    per = np.random.permutation(pair_issame_0.shape[0])[:num_pair_issame_1]  # 打乱后的行号
    pair_issame_0 = pair_issame_0[per, :]  # 获取打乱后的训练数据

    pair_issame = np.concatenate([pair_issame_0, pair_issame_1], axis=0)
    image_list1 = pair_issame[:, 0]
    image_list2 = pair_issame[:, 1]
    issame_list = pair_issame[:, 2]
    print("have images:{},combinations :{} pairs".format(len(image_list), len(pair_issame)))
    return image_list1, image_list2, issame_list


if __name__ == "__main__":
    image_path = "/media/dm/dm2/project/python-learning-notes/dataset/VOC/JPEGImages/000001.jpg"
    image = image_processing.read_image(image_path)
    iou_threshold = 0.5
    score_threshold = 0.5
    # pred_bboxes=[[50,50,100,100],[150,50,200,100],[50,150,100,200],[150,150,200,200]]
    # pred_labels=[1,2,3,4]
    pred_bboxes = [[151, 152, 200, 200], [49, 49, 100, 100], [150, 50, 200, 100]]
    pred_labels = ["D", "A", "B"]
    true_bboxes = [[50, 50, 100, 100], [150, 50, 200, 100], [50, 150, 100, 200], [150, 150, 200, 200]]
    true_labels = ["A", "B", "C", "D"]
    # image_processing.show_image_bboxes_text("image_dict",image_dict,pred_bboxes,pred_labels)
    # cal_face_recognition(pred_bboxes, pred_labels, true_bboxes, true_labels, iou_threshold, score_threshold)

    filename = "/media/dm/dm/project/dataset/face_recognition/AgeDB-30/agedb_30_pair.txt"
    faces_list1, faces_list2, issames_data = pair_data = read_pair_data(filename)
