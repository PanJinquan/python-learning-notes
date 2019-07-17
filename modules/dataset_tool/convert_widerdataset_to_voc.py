# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : convert_wider_to_voc.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-19 13:55:05
"""
import cv2
import shutil
import os
from xml.dom.minidom import Document


def writexml(filename, saveimg, bboxes, xmlpath):
    """
    write to xml style of VOC dataset
    :param filename: xml filename
    :param saveimg: the image data with shape [H, W, C]
    :param bboxes: bounding boxes
    :param xmlpath: xml file save path
    :return: None
    """

    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('wider face Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)
    flikerid = doc.createElement('flikerid')
    flikerid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flikerid)

    owner = doc.createElement('owner')
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('kinhom'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(ymax)

    f = open(xmlpath, 'w')
    f.write(doc.toprettyxml(indent=' '))
    f.close()

def convertimgset(wider_face_root,img_set):
    imgdir = wider_face_root + "/WIDER_" + img_set + "/images"
    gtfilepath = wider_face_root + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"

    if not os.path.exists(wider_face_root + "/ImageSets/Main/"):
         os.makedirs(wider_face_root + "/ImageSets/Main/")
    if not os.path.exists(wider_face_root + "/Annotations"):
        os.makedirs(wider_face_root + "/Annotations")
    if not os.path.exists(wider_face_root + "/JPEGImages"):
        os.makedirs(wider_face_root + "/JPEGImages")

    fwrite = open(wider_face_root + "/ImageSets/Main/" + img_set + ".txt", 'w')
    index = 0

    with open(gtfilepath, 'r') as gtfiles:
        while True:  # True
            filename = gtfiles.readline()[:-1]
            # if filename == "":
            #     continue
            if not filename:
                break
            imgpath = imgdir + "/" + filename
            # print(imgpath)
            img = cv2.imread(imgpath)
            if img is None:
                continue
            if not img.data:
                continue
            numbbox = int(gtfiles.readline())

            bboxes = []
            for i in range(numbbox):
                line = gtfiles.readline()
                lines = line.split()
                lines = lines[0: 4]
                bbox = (int(lines[0]), int(lines[1]), int(lines[0]) + int(lines[2]), int(lines[1]) + int(lines[3]))
                bboxes.append(bbox)

            filename = filename.replace("/", "_")

            if len(bboxes) == 0:
                print("no face")
                continue

            cv2.imwrite("{}/JPEGImages/{}".format(wider_face_root, filename), img)
            fwrite.write(filename.split('.')[0] + '\n')

            xmlpath = '{}/Annotations/{}.xml'.format(wider_face_root, filename.split('.')[0])
            writexml(filename, img, bboxes, xmlpath)
            if index % 100 == 0:
                print("success NO." + str(index))
            index += 1
    print(img_set + " total: " + str(index))
    fwrite.close()


if __name__ == "__main__":
    # wider face dataset folder path
    wider_face_root = "/media/dm/dm2/project/dataset/face"
    img_sets = ['train', 'val']
    for img_set in img_sets:
        print("handling " + img_set)
        convertimgset(wider_face_root,img_set)

    shutil.move(wider_face_root + "/ImageSets/Main/" + "train.txt", wider_face_root + "/ImageSets/Main/" + "trainval.txt")
    shutil.move(wider_face_root + "/ImageSets/Main/" + "val.txt", wider_face_root + "/ImageSets/Main/" + "test.txt")
