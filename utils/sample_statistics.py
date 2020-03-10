# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : sample_statistics.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-13 15:09:33
"""
from utils import file_processing, plot_utils
import numpy as np
import pandas as pd
from modules.pandas_json import pandas_tools


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


def count_data_dict(data_list):
    '''
    statis sample nums
    print(pd.value_counts(label_list))
    :param data_list:
    :return: label_set  : label set
             label_count: label nums
    '''
    data_set = list(set(data_list))
    data_set.sort()
    count_dict = []
    for s in data_set:
        nums = data_list.count(s)
        count_dict[s] = nums
    return count_dict


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
        data_range = list(range(0, len(data_set)))
        # data_range=data_set
        plot_utils.plot_bar(x_data=data_range, y_data=count_list, title=title, xlabel="ID", ylabel="COUNT")
        # plot_utils.plot_multi_line([data_set], [count_list], [line_names], title=title, xlabel="ID", ylabel="COUNT")
    # return count_list, data_set
    return p


if __name__ == "__main__":
    # image_dir = "/media/dm/dm2/project/dataset/face_recognition/NVR/facebank/NVR_3_20190605_1005_VAL"
    # dataset="/media/dm/dm2/project/dataset/face_recognition/CASIA-FaceV5/"
    # image_dir = dataset+"CASIA-Faces"
    # dataset="/media/dm/dm2/project/dataset/face_recognition/celebs_add_movies/"
    # image_dir = dataset+"Asian_Faces"
    image_dir = '/media/dm/dm1/project/dataset/face_recognition/X2T/X2T_Face233/val'
    # image_dir = '/media/dm/dm1/project/dataset/face_recognition/NVR/face/NVR1/trainval'
    image_list, label_list = file_processing.get_files_labels(image_dir)
    name_table = list(set(label_list))
    label_list = file_processing.encode_label(name_list=label_list, name_table=name_table)
    label_list = [int(l) for l in label_list]
    label_list.sort()
    # count = Counter(label_list)
    # count = label_list.count()
    # print(count)
    pd_data = count_data_info_pd(label_list)
    filename = "my_test2.csv"
    pd = pandas_tools.construct_pd(index=None, columns_name=["A"], content=pd_data, filename=filename)
    print(pd)
