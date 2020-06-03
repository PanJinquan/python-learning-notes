# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: ultra-light-fast-generic-face-detector-1MB
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-03 18:38:34
# --------------------------------------------------------
"""
from __future__ import print_function
import os, sys

sys.path.append("..")
sys.path.append(os.path.dirname(__file__))
sys.path.append("../..")
sys.path.append(os.getcwd())

import argparse
import torch
import cv2
import numpy as np

from models.config.config import cfg_mnet, cfg_slim, cfg_rfb
from models.nets.retinaface import RetinaFace
from models.nets.net_slim import Slim
from models.nets.net_rfb import RFB
from models.layers.functions.prior_box import PriorBox
from models.layers.box_utils import decode, decode_landm
from models.nms.py_cpu_nms import py_cpu_nms
from utils import image_processing, debug


def get_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-m', '--model_path', default='./face_detection_rbf.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
    parser.add_argument('--confidence_threshold', default=0.8, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')

    args = parser.parse_args()
    return args


class UltraLightFaceDetector(object):
    def __init__(self, model_path, network="RFB", confidence_threshold=0.6,
                 nms_threshold=0.4, top_k=5000, keep_top_k=750, device="cuda:0"):
        """
        :param model_path:
        :param network: Backbone network mobile0.25 or slim or RFB
        :param confidence_threshold: confidence_threshold
        :param nms_threshold: nms_threshold
        :param top_k:
        :param keep_top_k:
        :param device:
        """
        self.device = device
        self.network = network
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.net, self.cfg, self.prior_data = self.build_net(self.network)
        self.net = self.load_model(self.net, model_path)
        self.net = self.net.to(self.device)
        torch.set_grad_enabled(False)
        self.net.eval()
        print('Finished loading model!')

    def build_net(self, network):
        """
        :param network: <dict> mobile0.25,slim or RFB
        :return:net,cfg,prior_data
        """
        net = None
        if network == "mobile0.25":
            cfg = cfg_mnet
            net = RetinaFace(cfg=cfg, phase='test')
        elif network == "slim":
            cfg = cfg_slim
            net = Slim(cfg=cfg, phase='test')
        elif network == "RFB":
            cfg = cfg_rfb
            net = RFB(cfg=cfg, phase='test')
        else:
            print("Don't support network!")
            exit(0)
        prior_data = self.get_priorbox(cfg)
        return net, cfg, prior_data

    def load_model(self, model, model_path):
        """
        :param model:
        :param model_path:
        :param load_to_cpu:
        :return:
        """
        print('Loading pretrained model from {}'.format(model_path))
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    @debug.run_time_decorator("pre_process")
    def pre_process(self, image, image_size, img_mean=(104, 117, 123)):
        """
        :param image:
        :param img_mean:
        :return:out_image
        """
        out_image = cv2.resize(image, dsize=(image_size, image_size))
        out_image = np.float32(out_image)
        out_image -= img_mean
        out_image = out_image.transpose(2, 0, 1)
        out_image = torch.from_numpy(out_image).unsqueeze(0)
        return out_image

    @debug.run_time_decorator("get_priorbox")
    def get_priorbox(self, cfg):
        # get priorbox
        image_size = cfg["image_size"]
        priorbox = PriorBox(cfg, image_size=(image_size, image_size))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        # get boxes and scores
        prior_data = priors.data
        return prior_data

    @debug.run_time_decorator("pose_process")
    def pose_process(self, loc, conf, landms, im_height, im_width, variance):
        """
        :param loc:
        :param conf:
        :param landms:
        :param im_height:
        :param im_width:
        :param cfg:
        :return:
        """
        scale = [im_width, im_height]
        bboxes_scale = torch.Tensor(scale * 2)
        landms_scale = torch.Tensor(scale * 5)
        bboxes_scale = bboxes_scale.to(self.device)
        landms_scale = landms_scale.to(self.device)
        boxes = decode(loc.data.squeeze(0), self.prior_data, variance)
        boxes = boxes * bboxes_scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # get landmarks
        landms = decode_landm(landms.data.squeeze(0), self.prior_data, variance)
        landms = landms * landms_scale
        landms = landms.cpu().numpy()
        dets, landms = self.nms_process(boxes,
                                        scores,
                                        landms,
                                        confidence_threshold=self.confidence_threshold,
                                        nms_threshold=self.nms_threshold,
                                        top_k=self.top_k,
                                        keep_top_k=self.keep_top_k)
        return dets, landms

    @staticmethod
    @debug.run_time_decorator("nms_process")
    def nms_process(boxes, scores, landms, confidence_threshold, nms_threshold, top_k, keep_top_k):
        """
        :param boxes:
        :param scores:
        :param landms:
        :param confidence_threshold:
        :param nms_threshold:
        :param top_k:
        :param keep_top_k:
        :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
                 landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
        """
        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        return dets, landms

    @debug.run_time_decorator("inference")
    def inference(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            loc, conf, landms = self.net(img_tensor)  # forward pass
        return loc, conf, landms

    @debug.run_time_decorator("adapter_bbox_score_landmarks")
    def adapter_bbox_score_landmarks(self, dets, landms):
        if len(dets) > 0:
            landms = landms.reshape(len(landms), -1, 2)
            bboxes = dets[:, 0:4]
            scores = dets[:, 4:5]
            # bboxes = self.get_square_bboxes(bboxes,fixed="H")
        else:
            bboxes, scores, landms = [], [], []
        return bboxes, scores, landms

    @staticmethod
    def get_square_bboxes(bboxes, fixed="W"):
        '''
        :param bboxes:
        :param fixed: (W)width (H)height
        :return:
        '''
        new_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            cx, cy = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            if fixed == "H":
                dd = h / 2
            elif fixed == 'W':
                dd = w / 2
            elif fixed > 0:
                dd = h / 2 * fixed
            fxmin = int(cx - dd)
            fymin = int(cy - dd)
            fxmax = int(cx + dd)
            fymax = int(cy + dd)
            new_bbox = (fxmin, fymin, fxmax, fymax)
            new_bboxes.append(new_bbox)
        new_bboxes = np.asarray(new_bboxes)
        return new_bboxes

    @debug.run_time_decorator("detect")
    def detect(self, bgr_image, isshow=False):
        """
        :param bgr_image:
        :return:
        bboxes: <np.ndarray>: (num_boxes, 4)
        scores: <np.ndarray>: (num_boxes, 1)
        scores: <np.ndarray>: (num_boxes, 5, 2)
        """
        im_height, im_width, _ = bgr_image.shape
        img_tensor = self.pre_process(bgr_image, image_size=self.cfg["image_size"])
        loc, conf, landms = self.inference(img_tensor)
        dets, landms = self.pose_process(loc,
                                         conf,
                                         landms,
                                         im_height,
                                         im_width,
                                         variance=self.cfg["variance"])
        bboxes, scores, landms = self.adapter_bbox_score_landmarks(dets, landms)
        # print(bboxes)
        if isshow:
            self.show_landmark_boxes("Det", bgr_image, bboxes, scores, landms)
        return bboxes, scores, landms

    @staticmethod
    def show_landmark_boxes(win_name, image, bboxes, scores, landms):
        '''
        显示landmark和boxes
        :param win_name:
        :param image:
        :param landmarks_list: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        '''
        image = image_processing.draw_landmark(image, landms, vis_id=True)
        image = image_processing.draw_image_bboxes_text(image, bboxes, scores, color=(0, 0, 255))
        cv2.imshow(win_name, image)
        cv2.waitKey(0)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


if __name__ == '__main__':
    # model_path = args.model_path
    # network = args.network
    # confidence_threshold = args.confidence_threshold
    # nms_threshold = args.nms_threshold
    # top_k = args.top_k
    # keep_top_k = args.keep_top_k
    model_path = "./face_detection_rbf.pth"
    network = "RFB"
    confidence_threshold = 0.85
    nms_threshold = 0.1
    top_k = 20
    keep_top_k = 50
    device = "cuda:0"
    image_path = "./test.jpg"
    # image_path = "./data/11.jpg"
    # image_path = "/media/dm/dm1/git/python-learning-notes/libs/ultra_ligh_face/data/9.jpg"
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image_processing.read_image(image_path, colorSpace="BGR")
    detector = UltraLightFaceDetector(model_path,
                                      network=network,
                                      confidence_threshold=confidence_threshold,
                                      nms_threshold=nms_threshold,
                                      top_k=top_k,
                                      keep_top_k=keep_top_k)
    detector.detect(image, isshow=False)
    detector.detect(image, isshow=False)
    detector.detect(image, isshow=False)
    detector.detect(image, isshow=False)
    bboxes, scores, landms = detector.detect(image, isshow=True)
    print("bboxes:\n{}\nscores:\n{}\nlandms:\n{}".format(bboxes, scores, landms))
