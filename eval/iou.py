# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : iou.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-10 10:14:56
"""


def cal_iou(box1, box2):
    """
    computing IoU
    :param box1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param box2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    cx1, cy1, cx2, cy2 = box1
    gx1, gy1, gx2, gy2 = box2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def cal_iou_list(box1, box2_list):
    iou_list = []
    for box2 in box2_list:
        iou = cal_iou(box1, box2)
        iou_list.append(iou)
    return iou_list


if __name__ == '__main__':
    box1 = [661, 27, 679, 47]
    box2 = [661, 27, 679, 47]
    box2_list=[]
    box2_list.append(box2)
    box2_list.append(box2)
    iou = cal_iou_list(box1, box2_list)
    print(iou)
