# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : pr.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-10 17:10:56
"""
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import sklearn.model_selection as cross_validation

def plot_pr_curve(recall_list, precision_list, auc_list, line_names):
    '''
    绘制roc曲线
    :param recall_list:
    :param precision_list:
    :param auc_list:
    :param line_names:曲线名称
    :return:
    '''
    # 绘图
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors=["b","r","c","m","g","y","k","w"]
    for r, p ,roc_auc, color,line_name in zip(recall_list, precision_list, auc_list, colors, line_names):
        plt.plot(r, p, color=color,lw=lw, label='{} PR curve (area = {:.3f})'.format(line_name,roc_auc))  #假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--') # 绘制y=1-x的直线

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    plt.xlabel('Precision',font)
    plt.ylabel('Recall',font)

    plt.title('PR curve')
    plt.legend(loc="lower right")#"upper right"
    # plt.legend(loc="upper right")#"upper right"

    plt.show()

def get_classification_precision_recall(y_true, probas_pred):
    '''
    对于二分类问题，可以直接调用sklearn.metrics的precision_recall_curve()
    :param y_true: 真实样本的的正负标签
    :param probas_pred: 预测的分数
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.
    :return:
    '''

    precision, recall, _thresholds = metrics.precision_recall_curve(y_true, probas_pred)
    AUC = metrics.auc(recall, precision)
    return precision, recall,AUC

def get_object_detection_precision_recall(iou_data, probas_pred, iou_threshold):
    '''
    在二分类中，recall的分母是测试样本的的正样本的个数，因此可认为二分类recall是召回正样本的比率
    sklearn.metrics的precision_recall_curve()是根据二分类返回的recall；
    但对于目标检测问题，召回率recall与二分类的情况不同，目标检测的召回率recall的分母是groundtruth boundingbox的个数
    可以认为目标检测的recall是召回groundtruth boundingbox的比率；
    显示将precision_recall_curve()返回recall*正样本的个数/groundtruth boundingbox个数就是目标检测的recall了
    :param iou_data:
    :param probas_pred:
    :param iou_threshold:
    :return:
    '''
    true_label=np.where(iou_data > iou_threshold, 1, 0)
    t_nums=np.sum(true_label)  # 测试样本中正样本个数
    gb_nums=len(iou_data)      # groundtruth boundingbox的个数
    precision, recall, _thresholds = metrics.precision_recall_curve(true_label, probas_pred)
    recall=recall*t_nums/gb_nums
    AUC = metrics.auc(recall, precision)
    return precision, recall,AUC

def plot_classification_pr_curve(true_label, prob_data, plot_pr=True):
    precision, recall,AUC=get_classification_precision_recall(y_true=true_label, probas_pred=prob_data)
    # 绘制ROC曲线
    if plot_pr:
        recall_list = [recall]
        precision_list = [precision]
        auc_list = [AUC]
        line_names = ["line_name"]
        plot_pr_curve(recall_list, precision_list, auc_list, line_names)

def plot_object_detection_pr_curve(iou_data, prob_data, iou_threshold, plot_pr=True):
    precision, recall,AUC=get_object_detection_precision_recall(iou_data=iou_data, probas_pred=prob_data,
                                                                iou_threshold=iou_threshold)
    # 绘制ROC曲线
    if plot_pr:
        recall_list = [recall]
        precision_list = [precision]
        auc_list = [AUC]
        line_names = [""]
        plot_pr_curve(recall_list, precision_list, auc_list, line_names)
if __name__=="__main__":
    iou_data=[0.88,0.4,0.70]
    prob_data=[0.9,0.8,0.7]
    iou_data=np.array(iou_data)
    prob_data=np.array(prob_data)
    # 阈值
    iou_threshold=0.5
    # true_label=np.where(iou_data > iou_threshold, 1, 0)
    # pr_classification_test(true_label, prob_data)
    plot_object_detection_pr_curve(iou_data, prob_data, iou_threshold, plot_pr=True)