import os
from easydict import EasyDict as edict
import torch
from datetime import datetime


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def print_dict(dict_data, save_path):
    list_config = []
    for key in dict_data:
        info = "conf.{}={}".format(key, dict_data[key])
        print(info)
        list_config.append(info)
    if save_path is not None:
        with open(save_path, "w") as f:
            for info in list_config:
                f.writelines(info + "\n")


def get_config(conf):
    conf.DEVICE = torch.device("cuda:{}".format(conf.GPU_ID[0]) if torch.cuda.is_available() else "cpu")
    DATASET_NAME = "_".join([os.path.basename(path) for path in conf.DATA_ROOT])
    conf.PREFIX = "{}_{}_{}_{}_{}_{}".format(conf.BACKBONE_NAME, conf.HEAD_NAME, conf.LOSS_NAME, conf.INPUT_SIZE[0],
                                             conf.EMBEDDING_SIZE, conf.WIDTH_MULT)
    if conf.TIME:
        conf.WORK_DIR = "work_space/{}_{}_{}_{}".format(conf.PREFIX, conf.TRANSFORM_TYPE, DATASET_NAME, get_time())
    else:
        conf.WORK_DIR = "work_space/{}_{}_{}".format(conf.PREFIX, conf.TRANSFORM_TYPE, DATASET_NAME)
    conf.MODEL_ROOT = os.path.join(conf.WORK_DIR, "models")  # the root to buffer your checkpoints
    conf.LOG_ROOT = os.path.join(conf.WORK_DIR, "log")  # the root to log your train/val status
    conf.CONFIG_TRAIN = os.path.abspath(__file__)
    conf.CONFIG_TRAIN_BK = os.path.join(conf.LOG_ROOT, os.path.basename(__file__))
    conf.CONFIG_PATH = os.path.join(conf.LOG_ROOT, os.path.basename(__file__)[:-len(".py")] + ".txt")
    if not os.path.exists(conf.MODEL_ROOT):
        os.makedirs(conf.MODEL_ROOT)
    if not os.path.exists(conf.LOG_ROOT):
        os.makedirs(conf.LOG_ROOT)
    return conf


def config():
    conf = edict()
    # set model parameter
    conf.SEED = 1337  # random seed for reproduce results
    # local
    conf.DATA_ROOT1 = '/media/dm/dm/project/InsightFace_Pytorch/custom_insightFace/data/faces_emore/imgs'
    conf.DATA_ROOT2 = '/media/dm/dm/project/face.evoLVe.PyTorch/data/dataset'
    conf.DATA_ROOT = [conf.DATA_ROOT1, conf.DATA_ROOT2]
    # conf.DATA_ROOT = [conf.DATA_ROOT1]
    conf.VAL_ROOT = "/media/dm/dm/project/dataset/face_recognition/faces_vgg_112x112/faces_emore"
    # service
    # conf.DATA_ROOT1 = "/data/panjinquan/FaceData/ms1m_align_112"
    # conf.DATA_ROOT2 = "/data/panjinquan/FaceData/DMFR_V1"
    # conf.DATA_ROOT3 = "/data/panjinquan/FaceData/Asian_Celeb"
    # conf.DATA_ROOT = [conf.DATA_ROOT3]
    # conf.DATA_ROOT = [conf.DATA_ROOT1, conf.DATA_ROOT2]
    # conf.DATA_ROOT = [conf.DATA_ROOT1, conf.DATA_ROOT2,conf.DATA_ROOT3]

    # conf.VAL_ROOT = "/data/panjinquan/FaceData/val"
    # conf.VAL_DATASET = ["X4", "NVR1", "lfw", "agedb_30"]  # val data
    # conf.VAL_DATASET = ["NVR1", "lfw"]  # val data
    conf.VAL_DATASET = ['X4']  # val data

    conf.BACKBONE_NAME = 'IR_18'
    # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152',
    # 'IR_MB_V1','IR_MB_V2','MB_V1','MB_V2','ResNet_18','IR_18','IR_SE_18','MixNet_M']
    conf.WIDTH_MULT = None  # [1.5,1.4,1.3,1.2,1.0,0.75,0.5,0.35]
    conf.HEAD_NAME = 'Softmax'  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    conf.LOSS_NAME = 'Softmax'  # support: ['Focal', 'Softmax']
    conf.INPUT_SIZE = [64, 64]  # support: [112, 112] and [224, 224]
    conf.EMBEDDING_SIZE = 8  # feature dimension
    conf.BATCH_SIZE = 512
    conf.MULTI_GPU = True
    conf.NUM_WORKERS = 24
    # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
    # conf.GPU_ID = [1,0]  # specify your GPU ids
    conf.GPU_ID = [0]  # specify your GPU ids
    conf.DEVICE = torch.device("cuda:{}".format(conf.GPU_ID[0]) if torch.cuda.is_available() else "cpu")
    conf.TRANSFORM_TYPE = "default"  # [default,scale20_50,scale20_112,scale30] TRANSFORM_TYPE

    # conf.val_data_resize_list = [112, 50, 30]
    conf.VAL_DATA_RESIZE = [64, 30]  # VAL_DATA_RESIZE
    conf.TIME = True
    conf.PIN_MEMORY = True
    conf.RGB_MEAN = [0.5, 0.5, 0.5]  # for normalize inputs to [-1, 1]
    conf.RGB_STD = [0.5, 0.5, 0.5]
    conf.DROP_LAST = True  # whether drop the last batch to ensure consistent batch_norm statistics
    conf.LR = 0.1  # initial LR
    conf.NUM_EPOCH = 200  # total epoch number (use the firt 1/25 epochs to warm up)
    conf.WEIGHT_DECAY = 5e-4  # do not apply to batch_norm parameters
    conf.MOMENTUM = 0.9
    conf.STAGES = [35, 65, 95, 150]  # epoch stages to decay learning rate

    conf.RESUME = False
    conf.BACKBONE_RESUME_ROOT = 'work_space/'
    conf.HEAD_RESUME_ROOT = 'work_space/'  # the root to resume training from a saved checkpoint
    return conf

if __name__ == "__main__":
    config = get_config()
    print_dict(config, config.CONFIG_PATH)
