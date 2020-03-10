from utils import file_processing, image_processing
import warnings
import numpy as np
import pandas as pd
import os
import  cv2


def read_csv(path):
    data = pd.read_csv(path)
    return data

def extract_bbox(f, label, cartoon = False, inside = False, group = False , IsTruncated = False, ):
    u = f.loc[f['LabelName'] == label]
    # j = j.loc[j['IsTruncated'] ==  & j['IsGroupOf'] == 0 & j['IsDepiction'] == 0 & j['IsInside'] == 0]
    u = u.loc[u['IsTruncated'] == 0]
    u = u.loc[u['IsGroupOf'] == 0]
    u = u.loc[u['IsDepiction'] == 0]
    u = u.loc[u['IsInside'] == 0 ]

    keep_col = ['ImageID', 'XMin', 'XMax', 'YMin', 'YMax']

    # For multiple classes use the below, adding as many new LabelNames as needed
    # numClasses = ['/m/04hgtk','/m/0k65p']
    # u = f.loc[df['LabelName'].isin(numClasses)]
    # keep_col = ['LabelName', ImageID','XMin','XMax','YMin','YMax']

    new_f = u[keep_col]
    new_f['width'] = new_f['XMax'] - new_f['XMin']
    new_f['height'] = new_f['YMax'] - new_f['YMin']
    new_f['x'] = new_f['XMin']
    new_f['y'] =  new_f['YMin']
    keep_col = ['ImageID', 'x', 'y', 'width', 'height']
    new_f_2 = new_f[keep_col]
    return new_f_2


def copy_file(filename, srcfile_dir, dstfile_dir):
    """
    从原始数据集目录中移动图片到目标目录
    :param info_list: 每张图片列表内容
    :param srcfile_dir: 原始目录
    :param dstfile_dir: 目标目录
    :return:
    """
    file_processing.copyfile_dstdir(filename, srcfile_dir, dstfile_dir)

def convert2voc(data ,image_dir, dst_dir):
    for index in range(data.shape[0]):
        print(index)
        for i in range(9):
            image_dir_list = os.path.join(image_dir+"0%d"%i, data.iloc[index, :]["ImageID"] + ".jpg")
            if  os.path.isfile(image_dir_list):
                break
            print(image_dir_list)

            # if not os.path.isfile(os.path.join(image_dir ,data.iloc[index, :]["ImageID"] + ".jpg")):
            #     print(" not exist!" )
            # continue
        if not os.path.isfile(image_dir_list):
            continue
        # image_dict = image_processing.read_image(os.path.join(image_dir, data.iloc[index,:]["ImageID"]+".jpg"))
        image = image_processing.read_image(image_dir_list)
        print(data.iloc[index,:]["ImageID"])
        x, y, w, h = (data.iloc[index,:]["x"]*image.shape[1] ,
                     data.iloc[index, :]["y"] * image.shape[0],
                      data.iloc[index, :]["width"] * image.shape[1],
                      data.iloc[index, :]["height"] * image.shape[0])
        label_path = os.path.join(label_dir, data.iloc[index, :]["ImageID"]+".txt")
        if w * h < 1500:
            continue
        #
        with open(label_path , "a+") as f:
                f.write("2")
                f.write(" ")
                f.write("{:<.0f} {:<.0f} {:<.0f} {:<.0f}".format(x,y,w,h))
                f.write("\n")
        # copy_file(data.iloc[index, :]["ImageID"]+".jpg", image_dir, dst_dir)
        copy_file(data.iloc[index, :]["ImageID"]+".jpg", image_dir+"0%d"%i, dst_dir)

def write_img_name(img_dir, save_path):
    img_list = os.listdir(img_dir)
    with open(save_path ,'w') as f:
        for name in img_list:
            f.write(name[:len(name.split(".")[0])])
            f.write("\n")

# def merge_label(file_path1, file_path2):

if __name__ == '__main__':
    image_dir = "/home/dm/data/openimage/train_"
    bbox_path = "/home/dm/data/openimage/train-annotations-bbox.csv"
    # image_label_path = "/home/dm/data/openimage/validation-annotations-human-imagelabels-boxable.csv"
    # rotation_path = "/home/dm/data/openimage/validation-images-with-rotation.csv"
    label_dir = "../openimage/openimage_pro/train/label"
    dst_dir = "../openimage/openimage_pro/train/image_dict"
    img_name_path = '../openimage/openimage_pro/train/train.txt'

    label_list = ["/m/02p0tk3", "/m/0dzct"]
    human_body = "/m/02p0tk3"
    human_face = "/m/0dzct"
    human_head = "/m/04hgtk"

    bbox = read_csv(bbox_path)
    body = extract_bbox(bbox ,human_body)
    # print(body)
    # print(body.shape)
    # body.to_csv("../openimage/body.csv", index = False)
    # # print(body.shape)
    # face = extract_bbox(bbox , human_face)
    # face.to_csv("../openimage/face.csv", index = False)
    # head = extract_bbox(bbox , human_head)
    # head.to_csv("../openimage/head.csv", index = False)

    # body = read_csv("../openimage/body.csv")
    # print(body.shape)
    # print(body.head(5))
    # face = read_csv("../openimage/face.csv")
    convert2voc(body ,image_dir ,dst_dir)

    write_img_name(dst_dir, img_name_path)
    # for i,index in enumerate(body["ImageID"][:50]):
    #     image_path = os.path.join(image_dir, index)+ ".jpg"
    #     print(image_path)
    #     print(index)
    # image_processing.show_image_rects("image_dict", image_dict, rect_list)
    # cv2.imshow("image_dict", image_dict)
    # cv2.waitKey(0)
    #     image_dict = image_processing.read_image(image_path)
    #     rect_list = [[body.iloc[i,:]["x"]*image_dict.shape[1] ,
    #                   body.iloc[i,:]["y"]*image_dict.shape[0] ,
    #                   body.iloc[i,:]["width"]*image_dict.shape[1],
    #                   body.iloc[i,:]["height"]*image_dict.shape[0]]]
    #     image_processing.show_image_rects("image_dict", image_dict, rect_list)
    #     print(rect_list)