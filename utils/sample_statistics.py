# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : sample_statistics.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-13 15:09:33
"""
from utils import image_processing, file_processing, plot_utils
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def count_data_info(data_list, _print=True, plot=True, title="data count info", line_names="data"):
    '''
    statis sample nums
    print(pd.value_counts(label_list))
    :param data_list:
    :return: label_set  : label set
             label_count: label nums
    '''
    data_set = list(set(data_list))
    data_set.sort()
    count_list = []
    for s in data_set:
        nums = data_list.count(s)
        count_list.append(nums)
    print("mean count :{}/{}={}".format(len(data_list), len(data_set), len(data_list) / len(data_set)))
    if plot:
        plot_utils.plot_bar(x_data=data_set, y_data=count_list, title=title, xlabel="ID", ylabel="COUNT")
        # plot_utils.plot_multi_line([data_set], [count_list], [line_names], title=title, xlabel="ID", ylabel="COUNT")

    return count_list, data_set


def count_data_info_pd(data_list, _print=True, plot=True, title="data count info", line_names="data"):
    p = pd.value_counts(data_list, sort=False)
    if _print:
        print(p)
    data_set = []
    count_list = []
    for key, count in p.items():
        # count=p[key]
        data_set.append(key)
        count_list.append(count)
    print("mean count :{}/{}={}".format(len(data_list), len(data_set), len(data_list) / len(data_set)))
    if plot:
        plot_utils.plot_bar(x_data=data_set, y_data=count_list, title=title, xlabel="ID", ylabel="COUNT")
        # plot_utils.plot_multi_line([data_set], [count_list], [line_names], title=title, xlabel="ID", ylabel="COUNT")
    # return count_list, data_set
    return p


if __name__ == "__main__":
    image_dir = "/media/dm/dm2/project/dataset/face_recognition/NVR/facebank/NVR_3_20190605_1005_VAL"
    image_list, label_list = file_processing.get_files_labels(image_dir)
    label_list = [int(l) for l in label_list]
    label_list.sort()
    # count = Counter(label_list)
    # count = label_list.count()
    # print(count)
    count_list, label_set = count_data_info_pd(label_list)
