# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : rename_tool.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-08-09 18:14:37
"""
import os
import shutil
import os.path
from utils import file_processing

def rename_image_dir(image_list, prefix="ID_"):
    for image_path in image_list:
        dirname = os.path.dirname(image_path)
        label = image_path.split(os.sep)[-2]
        # basename=os.path.basename(image_path)
        index = 0
        newName = prefix + label + '_{}.jpg'.format(index)
        newpath = os.path.join(dirname, newName)
        while os.path.exists(newpath):
            index += 1
            newName = prefix + label + '_{}.jpg'.format(index)
            newpath = os.path.join(dirname, newName)

        print(image_path)
        print(newName)
        os.rename(image_path, newpath)

def synch_rename_image_dir(src_dir1, src_dir2, dest_dir):
    '''
    synchronize rename image directory
    :param src_dir1:
    :param src_dir2:
    :param dest_dir:
    :return:
    '''
    image_list,image_id = file_processing.get_files_labels(src_dir1, postfix=['*.jpg'])
    class_set =list(set(image_id))
    class_set.sort()
    print(class_set)
    for cls_name in class_set:
        id=class_set.index(cls_name)+1
        s1=os.path.join(src_dir1,str(cls_name))
        s2=os.path.join(src_dir2,str(cls_name))
        if not os.path.exists(s1):
            print("no s1 dir:{}".format(s1))
            continue
        if not os.path.exists(s2):
            print("no s2 dir:{}".format(s2))
            continue
        d1=file_processing.create_dir(dest_dir,'val')
        # shutil.copytree(s1, os.path.join(d1,str(id)+"_{}".format(cls_name)))
        shutil.copytree(s1, os.path.join(d1,str(id)))


        d2=file_processing.create_dir(dest_dir,'facebank')
        # shutil.copytree(s2, os.path.join(d2,str(id)+"_{}".format(cls_name)))
        shutil.copytree(s2, os.path.join(d2,str(id)))




if __name__ == '__main__':
    # src_dir1='/media/dm/dm1/FaceDataset/lexue/lexue2/val'
    # src_dir2='/media/dm/dm1/FaceDataset/lexue/lexue2/facebank'
    # dest_dir='/media/dm/dm1/FaceDataset/lexue/lexue2/dest'
    # synch_rename_image_dir(src_dir1,src_dir2,dest_dir)

    # # dir = '/media/dm/dm/project/dataset/face_recognition/NVR/face/NVRS/trainval'
    # # dataset_dir='F:/clear_data_bzl/val'
    # # dataset_dir='/media/dm/dm/XMC/FaceData/X4/X4_Face132/val'
    # # dataset_dir = '/media/dm/dm1/FaceDataset/lexue/lexue/facebank'
    dataset_dir='/media/dm/dm1/FaceDataset/lexue/lexue2/val'
    image_list = file_processing.get_files_list(dataset_dir, postfix=['*.jpg'])
    rename_image_dir(image_list, prefix="val_")


