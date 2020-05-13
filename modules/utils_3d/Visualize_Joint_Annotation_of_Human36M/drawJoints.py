import os
import numpy as np
import cv2

JoName='0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,' \
       ' 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist'
JoName=JoName.split(',')

def DrawJoints(I,joints,CONNECTED_PAIRS=None,Jo=JoName):
    '''
    Assume joints Dim : P x 2
    :param I:
    :param joints:
    :param CONNECTED_PAIRS:
    :return:
    '''
    joints = np.reshape(joints,[16,2])
    txt='Missing Joints:\n'
    # Background rectangle for text
    cv2.rectangle(I, (25, 25), (300, 250), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(joints.shape[0]):
        if joints[i,0]<5 and joints[i,1]<5 :
            txt=txt+ str(Jo[i]) +'\n'
        cv2.circle(I, (joints[i,0], joints[i,1]), 2, (0, 255,0 ), -1)
    if CONNECTED_PAIRS:
        I=drawLine(I,joints,CONNECTED_PAIRS)

    # show text
    y0, dy = 50, 25
    for i, txt in enumerate(txt.split('\n')):
        y = y0 + i * dy
        cv2.putText(I, txt, (50, y), font, .8, (0, 255, 0), 1, 2)
    # print(txt)
    cv2.line(I, (100,100), (900,100), (0, 0, 255), 2)
    cv2.line(I, (900, 900), (900, 100), (0, 0, 255), 2)
    cv2.line(I, (100, 100), (100, 900), (0, 0, 255), 2)
    cv2.line(I, (900, 900), (100, 900), (0, 0, 255), 2)
    return I

def drawLine(I,joints,CONNECTED_PAIRS):
    N=len(CONNECTED_PAIRS)
    for i in range(N):
        p1=joints[CONNECTED_PAIRS[i][0]]
        p2=joints[CONNECTED_PAIRS[i][1]]
        p1=totuple(p1)
        p2 = totuple(p2)
        cv2.line(I,p1,p2,(255,0,0),2)
    return I

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

