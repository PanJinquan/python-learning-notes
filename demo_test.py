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


def set_format(handler, format):
    # handler.suffix = "%Y%m%d"
    if format:
        logFormatter = logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(levelname)s: %(message)s",
                                         "%Y-%m-%d %H:%M:%S")
    else:
        logFormatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(logFormatter)


def set_logging(name, level="info", logfile=None, format=False):
    """
    debug.py
    url:https://cuiqingcai.com/6080.html
    level级别：debug>info>warning>error>critical
    :param level: 设置log输出级别
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :return:
    """
    logger = logging.getLogger(name)
    if logfile and os.path.exists(logfile):
        os.remove(logfile)
    # define a FileHandler write messages to file
    if logfile:
        # filehandler = logging.handlers.RotatingFileHandler(filename="./log.txt")
        filehandler = logging.handlers.TimedRotatingFileHandler(logfile, when="midnight", interval=1)
        set_format(filehandler, format)
        logger.addHandler(filehandler)

    # define a StreamHandler print messages to console
    console = logging.StreamHandler()
    set_format(console, format)
    logger.addHandler(console)
    # set initial log level
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    if level == 'info':
        logger.setLevel(logging.INFO)
    if level == 'warning':
        logger.setLevel(logging.WARN)
    if level == 'critical':
        logger.setLevel(logging.CRITICAL)
    if level == 'fatal':
        logger.setLevel(logging.FATAL)
    logger.info("Init log in %s level", level)
    return logger


logger = set_logging(name="LOG", level="debug", logfile="log.txt", format=False)

if __name__ == "__main__":
    msg = "this is just a test"
    logger.info(msg)
    logger.debug(msg)
    logger.error(msg)
