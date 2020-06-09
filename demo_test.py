# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""

import os
import PIL.Image as Image
import numpy as np
import cv2
import random
from utils import image_processing, file_processing, numpy_tools

import glob

import logging

import logging.handlers

import os


class LabelSmoothing(object):
    def __init__(self, eps=0.1, p=0.5):
        self.p = p
        self.eps = eps

    def __call__(self, img_dict):
        if np.random.rand() < self.p:
            img_dict['label'] = np.abs(img_dict['label'] - self.eps)

        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + '(eps={0}, p={1})'.format(self.eps, self.p)


if __name__ == "__main__":
    ls = LabelSmoothing()
    print(repr(ls))
