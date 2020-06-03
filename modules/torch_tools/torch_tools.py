# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-06-02 16:00:47
# --------------------------------------------------------
"""
import torch
import random
import os
import numpy as np


def set_env_random_seed(seed=2020):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
