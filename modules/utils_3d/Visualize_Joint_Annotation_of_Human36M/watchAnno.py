import numpy as np
import cv2
import h5py
import time
import os
from drawJoints import DrawJoints
from spacepy import pycdf
os.environ["CDF_LIB"] = "/usr/local/cdf/lib"
# CONNECTED_PAIRS=[[10,11],[11,12],[13,14],[14,15],[0,1],[1,2],[3,4],[4,5]]
CONNECTED_PAIRS=None


def showVid(cap, pred,title='frame'):
    idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while (True):
        # t1=time.time()
        if idx >= pred.shape[0]:
            return
        ret, frame = cap.read()
        frame = DrawJoints(frame, pred[idx, ::], CONNECTED_PAIRS)
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
    pred_files= [i for i in os.listdir(os.path.join(pred_dir,idx,'MyPoseFeatures','D2_Positions')) if i.split('.')[-1]=='cdf']
    video_files.sort()
    # pred_files = [p.replace('_',' ') for p in pred_files]
    pred_files.sort()
    # pred_files = [p.replace(' ', '_') for p in pred_files]

    ip=0
    for i in range(len(video_files)):
        cur_file=video_files[i]
        if idx=='S11' and cur_file=='Directions.54138969.mp4':
            continue
        else:
            V.append(os.path.join(data_dir,idx,'Videos',cur_file))
            P.append(os.path.join(os.path.join(pred_dir,idx,'MyPoseFeatures','D2_Positions'),pred_files[ip]))
            ip=ip+1
    return V,P


def main():
    vidDir='/data3/Human36M/raw_data'
    predDir='/data3/Human36M/raw_data/'
    for S in ['S9','S11']:
        V,P = getFileList(vidDir,predDir,S)
        for i in range(len(V)):
            pred_name=P[i]
            filename=V[i]
            cdf=pycdf.CDF(pred_name)
            pred_data=cdf.copy()
            cdf.close()
            pred = np.array(pred_data['Pose'], dtype=np.int)
            pred=np.reshape(pred,[-1,32,2])
            # print(pred.shape)
            # return
            cap = cv2.VideoCapture(filename)
            showVid(cap, pred,title=filename.split('/')[-1])
            cap.release()
            cv2.destroyAllWindows()




if __name__ == '__main__':
    # test()
    main()






