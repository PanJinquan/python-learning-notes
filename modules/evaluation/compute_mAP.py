from voc_eval import voc_eval
import os

pwd = os.getcwd()

settings = ["TinyYolov2", "TinyYolov3", "Yolov2", "Yolov3"] # change result dir if you need
labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
annoFile = '../VOCdevkit/VOC2007/Annotations/{}.xml' # annotion file name template
valFile = '../VOCdevkit/VOC2007/ImageSets/Main/test.txt' # validation set list
annoDir = "./anno" # extracted annotation directory, create first if doesn't exist

for setting in settings:
    print("setting = " + setting)
    resultPath = os.path.join(pwd, 'results', setting)
    resultFile = os.path.join(resultPath, 'comp4_det_test_{}.txt') #[image_id prob xmin ymin xmax ymax]
    mAP = 0
    for label in labels:
        rec, prec, ap = voc_eval(resultFile, annoFile, valFile, label, os.path.join(annoDir, label))
        mAP = mAP + ap
        print('{:s}: {:.3f}'.format(label, ap))
    mAP = mAP / len(labels)
    print('{:s} mAP: {:.3f}'.format(setting, mAP))
