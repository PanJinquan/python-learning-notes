###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################
import sys
import os

sys.path.append(os.getcwd())
import shutil
import argparse
from lib.Evaluator import *
from lib import check


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description='This project applies the most popular metrics used to evaluate object detection '
                    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
                    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")

    # Optional
    parser.add_argument(
        '-t',
        '--threshold',
        dest='iouThreshold',
        type=float,
        default=0.5,
        metavar='',
        help='IOU threshold. Default 0.5')
    parser.add_argument(
        '-gtformat',
        dest='gtFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the ground truth bounding boxes: '
             '(\'xywh\': <left> <top> <width> <height>)'
             ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-detformat',
        dest='detFormat',
        metavar='',
        default='xywh',
        help='format of the coordinates of the detected bounding boxes '
             '(\'xywh\': <left> <top> <width> <height>) '
             'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument(
        '-gtcoords',
        dest='gtCoordinates',
        default='abs',
        metavar='',
        help='reference of the ground truth bounding box coordinates: absolute '
             'values (\'abs\') or relative to its image_dict size (\'rel\')')
    parser.add_argument(
        '-detcoords',
        default='abs',
        dest='detCoordinates',
        metavar='',
        help='reference of the ground truth bounding box coordinates: '
             'absolute values (\'abs\') or relative to its image_dict size (\'rel\')')
    parser.add_argument(
        '-imgsize',
        dest='imgSize',
        metavar='',
        help='image_dict size. Required if -gtcoords or -detcoords are \'rel\'')

    parser.add_argument(
        '-np',
        '--noplot',
        dest='showPlot',
        action='store_false',
        help='no plot is shown during execution')
    args = parser.parse_args()

    # Arguments validation
    errors = []
    # Validate formats
    args.gtFormat = check.ValidateFormats(args.gtFormat, '-gtformat', errors)
    args.detFormat = check.ValidateFormats(args.detFormat, '-detformat', errors)

    # Coordinates types
    args.gtCoordType = check.ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    args.detCoordType = check.ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
    args.imgSize = (0, 0)
    if args.gtCoordType == CoordinatesType.Relative:  # Image size is required
        args.imgSize = check.ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if args.detCoordType == CoordinatesType.Relative:  # Image size is required
        args.imgSize = check.ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
    # If error, show error messages
    if len(errors) is not 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()
    return args


def get_bboxes_class(gtFolder, gtFormat, gtCoordType, detFolder, detFormat, detCoordType, imgSize):
    # Get groundtruth boxes
    allBoundingBoxes, allClasses = check.getBoundingBoxes(gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = check.getBoundingBoxes(detFolder, False, detFormat, detCoordType, allBoundingBoxes,
                                                          allClasses, imgSize=imgSize)
    allClasses.sort()
    return allBoundingBoxes, allClasses


def evaluate(allBoundingBoxes, savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0
    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')
    f.flush()
    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            f.write('\n\nClass: %s' % cl)
            f.write('\nAP: %s' % ap_str)
            f.write('\nPrecision: %s' % prec)
            f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)


if __name__ == "__main__":
    # gtFolder = "/media/dm/dm1/git/python-learning-notes/eval/map_tools/groundtruths"
    # detFolder = "/media/dm/dm1/git/python-learning-notes/eval/map_tools/detections"
    # dataroot="/media/dm/dm2/project/pytorch-learning-tutorials/object_detection/Pytorch-SSD/models/mb2-ssd-lite/VOC2007_VOC2012_xmc_det_v3.1.1_voc/mb2-ssd-lite-2019-11-10-10-07/models/VOC2007"
    # dataroot = "/media/dm/dm2/project/pytorch-learning-tutorials/object_detection/Pytorch-SSD/models/mb2-ssd-lite/voc/models/VOC2007"
    # dataroot = "/media/dm/dm2/project/pytorch-learning-tutorials/object_detection/Pytorch-SSD/models/mb2-ssd-lite/voc/models/VOC2007"
    dataroot = "/media/dm/dm2/project/dataset/xmc/xmc_det_banchmark_v2.1/xmc_det_banchmark_v2.1_voc"

    gtFolder = os.path.join(dataroot, "result/gt_result")
    detFolder = os.path.join(dataroot, "result/dt_result")
    savePath = os.path.join(dataroot)

    args = get_parser()
    showPlot = args.showPlot
    iouThreshold = args.iouThreshold

    gtFormat = args.gtFormat
    detFormat = args.detFormat

    detCoordType = args.detCoordType
    gtCoordType = args.gtCoordType
    imgSize = args.imgSize
    allBoundingBoxes, allClasses = get_bboxes_class(gtFolder, gtFormat, gtCoordType, detFolder,
                                                    detFormat, detCoordType, imgSize)
    evaluate(allBoundingBoxes, savePath)
