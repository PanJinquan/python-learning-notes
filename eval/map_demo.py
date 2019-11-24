from eval.voc_eval import voc_eval
import os

pwd = os.getcwd()

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
annoFile = '../dataset/VOC/Annotations/{}.xml'  # annotion file name template
valFile = '../dataset/VOC/test.txt'  # validation set list
annoDir = "./Annotations"  # extracted annotation directory, create first if doesn't exist
resultFile = '../dataset/VOC/detect_result.txt'  # validation set list
mAP = 0
for label in labels:
    rec, prec, ap = voc_eval(resultFile, annoFile, valFile, label, os.path.join(annoDir, label))
    mAP = mAP + ap
    print('{:s}: {:.3f}'.format(label, ap))
mAP = mAP / len(labels)
print('mAP: {:.3f}'.format(mAP))
print("==========================================================")
