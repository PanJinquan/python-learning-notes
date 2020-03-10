# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""


import glob


def get_gpu_device():
    gpu_id = []
    for i in glob.glob('/dev/nvidia[0-7]'):
        gpu_id.append(int(i[-1]))
    # gpu_id = reversed(gpu_id)
    gpu_id = sorted(gpu_id)
    return gpu_id


if __name__ == "__main__":
    gpu_id = get_gpu_device()
    # gpu_id=sorted(gpu_id)
    # print(gpu_id)
