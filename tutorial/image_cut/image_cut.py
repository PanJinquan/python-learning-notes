# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : bn_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-18 17:01:33
"""

import cv2
import os
import re
from PIL import Image
import numpy as np

def contours_combine(image,contours):
    start_i = 0
    sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    max_c = sorted_cnts[0]
    max_x,max_y,max_w,max_h = cv2.boundingRect(max_c)
    if max_w == image.shape[1] and max_h == image.shape[0]:
        max_c = sorted_cnts[1]
        max_x, max_y, max_w, max_h = cv2.boundingRect(max_c)
        start_i = 1
    max_x_end = max_x + max_w
    max_y_end = max_y + max_h
    result_x = max_x
    result_y = max_y
    result_x_end = max_x + max_w
    result_y_end = max_y + max_h
    result_x_tmp = result_x
    result_y_tmp = result_y
    result_x_end_tmp = result_x_end
    result_y_end_tmp = result_y_end
    for contour in sorted_cnts[start_i:]:
        x,y,w,h = cv2.boundingRect(contour)
        x_end = x + w
        y_end = y + h
        if max_x == x and max_y == y and max_w == w and max_h == h:
            continue
        if max_x > x_end or max_x_end < x or max_y_end < y or max_y > y_end:
            continue
        if x_end > result_x_end:
            result_x_end_tmp = x_end
        if x < result_x:
            result_x_tmp = x
        if y < result_y:
            result_y_tmp = y
        if y_end > result_y_end:
            result_y_end_tmp = y_end
        if (result_x_end_tmp - result_x_tmp) == image.shape[1] and (result_y_end_tmp - result_y_tmp) == image.shape[0]:
            break
        result_x = result_x_tmp
        result_y = result_y_tmp
        result_x_end = result_x_end_tmp
        result_y_end = result_y_end_tmp
    return result_x,result_y,(result_x_end - result_x),(result_y_end -result_y)


def object_field_detect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (_,cnts,_) = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(c)
    if w == img.shape[1] and h == img.shape[0]:
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[1]
        x, y, w, h = cv2.boundingRect(c)
    return (x,y,w,h,th2)

def to_png_image(origin_image):
    w = origin_image.shape[1]
    h = origin_image.shape[0]
    png_img = Image.new("RGBA", (w, h))
    new_data = []
    for i in range(0, h):
        for j in range(0, w):
            b = origin_image[i][j][0]
            g = origin_image[i][j][1]
            r = origin_image[i][j][2]
            if b == 0 and g == 0 and r == 0:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append((r, g, b))
    png_img.putdata(new_data)
    return png_img

def image_cut(image):
    x, y, w, h, th2 = object_field_detect(image)
    rect = (x, y, w, h)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    out = image * mask2[:, :, np.newaxis]
    return out

def image_cut_combine(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    (_, cnts, _) = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = contours_combine(image, cnts)
    rect = (x, y, w, h)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    out = image * mask2[:, :, np.newaxis]
    return out


def image_cut_recurrence(image_path):
    if os.path.isdir(image_path):
        for file_name in os.listdir(image_path):
            file_path = os.path.join(image_path, file_name)
            image_cut_recurrence(file_path)
    else:
        try:
            save_path = re.sub("download", "cut", image_path)
            if os.path.exists(save_path):
                return
            img = cv2.imread(image_path)
            out = image_cut(img)
            save_dir = save_path[:save_path.rfind("/")]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_path,out)
        except:
            return

def image_cut_combine_recurrence(image_path):
    if os.path.isdir(image_path):
        for file_name in os.listdir(image_path):
            file_path = os.path.join(image_path, file_name)
            image_cut_combine_recurrence(file_path)
    else:
        try:
            save_path = re.sub("download", "cut_combine", image_path)
            if os.path.exists(save_path):
                return
            img = cv2.imread(image_path)
            out = image_cut_combine(img)
            save_dir = save_path[:save_path.rfind("/")]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_path,out)
        except:
            return

def main():
    file_dir = "/home/base/aidesign/download/SocialCommunity"
    image_cut_combine_recurrence(file_dir)

def test():
    try:
        file_dir = '/home/base/aidesign/origin/1035_261'
        for file_name in os.listdir(file_dir):
            image_path = os.path.join(file_dir,file_name)
            save_path = re.sub("download", "cut_combine", image_path)
            if os.path.exists(save_path):
                continue
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            (_, cnts, _) = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = contours_combine(image,cnts)
            rect = (x, y, w, h)
            mask = np.zeros(image.shape[:2], np.uint8)
            bgModel = np.zeros((1, 65), np.float64)
            fgModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            out = image * mask2[:, :, np.newaxis]
            
            save_dir = save_path[:save_path.rfind("/")]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_path, out)
    except:
        pass

def one_image_cut():
    image_path = "/home/base/image_cut.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    (_, cnts, _) = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = contours_combine(image, cnts)
    rect = (x, y, w, h)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    out = image * mask2[:, :, np.newaxis]
    # mask2 = cv2.GaussianBlur(mask2, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    mask2 = cv2.medianBlur(mask2,3)
    out2 = image * mask2[:,:,np.newaxis]
    cv2.imshow("out1", out)
    cv2.imshow("out2", out2)
    cv2.waitKey()

def one_image_cut_for_pjq():
    image_path = "/media/dm/dm1/git/python-learning-notes/dataset/test_image/image_cut.jpg"
    image = cv2.imread(image_path)
    # image_dict=cv2.resize(image_dict,(100,100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # (_, cnts, _) = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # by zengding
    cnts, _ = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)        # by pjq

    x, y, w, h = contours_combine(image, cnts)
    rect = (x, y, w, h)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    # convert to float32
    mask2=np.asarray(mask2/255,dtype=np.float32)
    image=np.asarray(image/255,dtype=np.float32)
    # erode image_dict
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.erode(mask2, kernel)

    # convert mask from 1-channel to 3-channel so as to multiply matrices
    # mask2=cv2.cvtColor(mask2,cv2.COLOR_GRAY2BGR)
    out1 = image*mask2[:, :, np.newaxis]

    # blur mask by Gaussian filter
    ksize=9
    mask3 = cv2.GaussianBlur(mask2, (ksize, ksize), 0, 0, cv2.BORDER_DEFAULT)
    # mask3 = cv2.blur(mask2, (ksize, ksize))

    # can not use medianBlur
    # mask3 = cv2.medianBlur(mask2*255,15)
    out2 = image * mask3[:, :, np.newaxis]


    # seamlessClone,have some bug to fix
    # dest_image = cv2.seamlessClone(out2,brg, mask2, center, cv2.NORMAL_CLONE)

    # split bgr-image_dict to 3-channels
    b_channel, g_channel, r_channel = cv2.split(image)
    # merge imgae by 4 channels[r,g,b,a],mask as transparent channel
    out3 = cv2.merge((b_channel, g_channel, r_channel, mask3))
    out3=np.asarray(out3*255,dtype=np.uint8)
    cv2.imwrite("1.png",out3)

    cv2.imshow("mask2", mask2)
    cv2.imshow("mask3", mask3)

    cv2.imshow("out1", out1)
    cv2.imshow("out2", out2)
    cv2.imshow("out3", out3)
    cv2.waitKey(0)

if __name__ == '__main__':
    print(cv2.__version__)
    one_image_cut_for_pjq()