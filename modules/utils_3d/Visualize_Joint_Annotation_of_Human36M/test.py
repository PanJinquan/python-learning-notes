import numpy as np
import cv2
import h5py
import time
import os
from modules.utils_3d.Visualize_Joint_Annotation_of_Human36M.drawJoints import DrawJoints

CONNECTED_PAIRS=[[10,11],[11,12],[13,14],[14,15],[0,1],[1,2],[3,4],[4,5]]


def showVid(cap, pred,title='frame'):
    idx = 0
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 50
    while (True):
        # t1=time.time()
        if idx >= pred.shape[0]:
            return
        ret, frame = cap.read()
        # frame = DrawJoints(frame, pred[idx, ::], CONNECTED_PAIRS)
        if not frame is None:
            frame = DrawJoints(frame, pred[idx, ::])
            cv2.imshow(title, frame)
        idx = idx + 1
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord(' '):
            while True:
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    return
                if (cv2.waitKey(1) & 0xFF == ord(' ')):
                    break
        if ( cap.get(cv2.CAP_PROP_FRAME_COUNT) <= cap.get(cv2.CAP_PROP_POS_FRAMES) ):
            return
        if (cv2.waitKey(2) & 0xFF == ord('q')) :
            return

def getFileList(data_dir,pred_dir,idx):
    '''
    Assume idx in ['S9','S11']
    :param data_dir:
    :param pred_dir:
    :param idx:
    :return:
    '''
    V=[] #Video
    P=[] #Predict
    video_files=[i for i in os.listdir(os.path.join(data_dir,idx,'Videos')) if (i.split('.')[-1]=='mp4' and (not i[0]=='_') )]
    pred_files= [i for i in os.listdir(os.path.join(pred_dir,idx,'StackedHourglassFineTuned240')) if i.split('.')[-1]=='h5']
    video_files.sort()
    pred_files = [p.replace('_',' ') for p in pred_files]
    pred_files.sort()
    pred_files = [p.replace(' ', '_') for p in pred_files]

    ip=0
    for i in range(len(video_files)):
        cur_file=video_files[i]
        if idx=='S11' and cur_file=='Directions.54138969.mp4':
            continue
        else:
            V.append(os.path.join(data_dir,idx,'Videos',cur_file))
            P.append(os.path.join(os.path.join(pred_dir,idx,'StackedHourglassFineTuned240'),pred_files[ip]))
            ip=ip+1
    return V,P



def getFileListTest(data_dir,pred_dir,idx):
    '''
    Assume idx in ['S9','S11']
    :param data_dir:
    :param pred_dir:
    :param idx:
    :return:
    '''
    ACTIONS=['Phoning','Sitting','Photo']
    V=[] #Video
    P=[] #Predict
    video_files=[i for i in os.listdir(os.path.join(data_dir,idx,'Videos')) if (i.split('.')[-1]=='mp4' and (not i[0]=='_'))]
    pred_files= [i for i in os.listdir(os.path.join(pred_dir,idx,'StackedHourglassFineTuned240')) if i.split('.')[-1]=='h5']
    video_files.sort()
    pred_files = [p.replace('_',' ') for p in pred_files]
    pred_files.sort()
    pred_files = [p.replace(' ', '_') for p in pred_files]



    ip=0
    for i in range(len(video_files)):
        cur_file=video_files[i]
        if idx=='S11' and cur_file=='Directions.54138969.mp4':
            continue
        else:
            for act in ACTIONS:
                if act in cur_file:
                    V.append(os.path.join(data_dir,idx,'Videos',cur_file))
                    P.append(os.path.join(os.path.join(pred_dir,idx,'StackedHourglassFineTuned240'),pred_files[ip]))
        ip=ip+1
    return V,P


def test():
    '''
    pred_name = '/data2/guoyu/workspace/3d-pose-baseline/temporal/data/h36m/S9/StackedHourglassFineTuned240/Greeting.54138969.h5'
    filename = '/data3/Human36M/raw_data/S9/Videos/Greeting.54138969.mp4'
    pred_data = h5py.File(pred_name)
    pred = np.array(pred_data['poses'], dtype=np.int)
    cap = cv2.VideoCapture(filename)
    # print cap.get(cv2.CAP_PROP_FRAME_COUNT)- pred.shape[0] ,S,filename.split('/')[-1]
    # cap.release()

    fps = cap.get(cv2.CAP_PROP_FPS)
    showVid(cap, pred, title=filename.split('/')[-1])
    cap.release()
    '''
    vidDir='/data3/Human36M/raw_data'
    predDir='/data2/guoyu/workspace/3d-pose-baseline/temporal/data/h36m'
    for S in ['S9','S11']:
        V,P = getFileListTest(vidDir,predDir,S)
        for i in range(len(V)):
            pred_name=P[i]
            filename=V[i]
            pred_data = h5py.File(pred_name)
            pred = np.array(pred_data['poses'], dtype=np.int)
            cap = cv2.VideoCapture(filename)
            # print int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'\t',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)- pred.shape[0]),'\t' ,S,filename.split('/')[-1].split('.')[0]
            # cap.release()
            fps = cap.get(cv2.CAP_PROP_FPS)
            showVid(cap, pred,title=filename.split('/')[-1])
            cap.release()
            cv2.destroyAllWindows()

def main():

    # Change file path below
    vidDir='/data3/Human36M/raw_data'
    predDir='/data2/guoyu/workspace/pytorch-3dpose/End2End/Train/DEBUG/data/pred_data_v3'
    # predDir='/data2/guoyu/workspace/3d-pose-baseline/temporal/data/h36m'
    for S in ['S9','S11']:
        V,P = getFileList(vidDir,predDir,S)
        for i in range(len(V)):
            pred_name=P[i]
            filename=V[i]
            pred_data = h5py.File(pred_name)
            pred = np.array(pred_data['poses'], dtype=np.int)
            cap = cv2.VideoCapture(filename)
            # cap.release()
            fps = cap.get(cv2.CAP_PROP_FPS)
            showVid(cap, pred,title=filename.split('/')[-1])
            cap.release()
            cv2.destroyAllWindows()

def test0813():

        pred_name = '/data2/guoyu/workspace/3d-pose-baseline/3d-pose-baseline/data/h36m/S9/StackedHourglassFineTuned240/Eating.54138969.h5'
        filename = '/data3/Human36M/raw_data/S9/Videos/Eating.54138969.mp4'
        pred_data = h5py.File(pred_name)
        pred = np.array(pred_data['poses'], dtype=np.int)
        cap = cv2.VideoCapture(filename)
        # print int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'\t',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)- pred.shape[0]),'\t' ,S,filename.split('/')[-1].split('.')[0]
        # cap.release()
        fps = cap.get(cv2.CAP_PROP_FPS)
        showVid(cap, pred, title=filename.split('/')[-1])
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # test0813()
    main()