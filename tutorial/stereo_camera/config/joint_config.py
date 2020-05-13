# -*-coding: utf-8 -*-
"""
    @Project: DepthPose
    @File   : body_joint.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-24 16:44:28
"""


class Joints():
    def __init__(self, type="openpose"):
        self.joint_nose = None
        self.joint_nect = None
        self.joint_head = None
        # joint_waist = [8, 11]  # 腰,RHip,LHip
        self.joint_shoulder_left = None
        self.joint_shoulder_right = None
        self.joint_lines = None
        self.joint_count = None
        self.get_joints_type(type)

    def get_joints_type(self, type=""):
        '''
        :param type: 'kinect2','openpose',"custom_mpii"
        :return:
        '''
        if type == "kinect2":
            self.kinect2_joints()
        elif type == "openpose":
            self.openpose_joints()
        elif type == "custom_mpii":
            self.custom_mpii_joints()
        else:
            raise Exception("Error:{}".format(type))

    def kinect2_joints(self):
        '''
        kinect2
        :return:
        '''
        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        # kinect2
        JointType_SpineBase = 0  # 脊柱底
        JointType_SpineMid = 1  # 脊柱中间
        JointType_Neck = 2  # 脖子
        JointType_Head = 3  # 额头
        JointType_ShoulderLeft = 4
        JointType_ElbowLeft = 5
        JointType_WristLeft = 6
        JointType_HandLeft = 7
        JointType_ShoulderRight = 8
        JointType_ElbowRight = 9
        JointType_WristRight = 10
        JointType_HandRight = 11
        JointType_HipLeft = 12
        JointType_KneeLeft = 13
        JointType_AnkleLeft = 14
        JointType_FootLeft = 15
        JointType_HipRight = 16
        JointType_KneeRight = 17
        JointType_AnkleRight = 18
        JointType_FootRight = 19
        JointType_SpineShoulder = 20
        JointType_HandTipLeft = 21
        JointType_ThumbLeft = 22
        JointType_HandTipRight = 23
        JointType_ThumbRight = 24
        JointType_Count = 25
        # kinect2
        self.joint_count = 25
        self.joint_head = 3
        self.joint_nect = 20  # or JointType_Neck = 2
        self.joint_waist = [0, 1]  # 腰
        self.joint_shoulder_left = 4
        self.joint_shoulder_right = 5
        self.joint_arm_left = [4, 5, 6]  # 手臂
        self.joint_arm_right = [8, 9, 10]  # 手臂
        self.joint_lines = [[0, 1], [1, 20], [20, 2], [2, 3],  # Spine
                            [20, 4], [4, 5], [5, 6], [6, 7], [7, 21], [7, 22],  # Left arm and hand
                            [20, 8], [8, 9], [9, 10], [10, 11], [11, 23], [11, 24],  # Right arm and hand
                            [0, 12], [12, 13], [13, 14], [14, 15],  # Left leg
                            [0, 16], [16, 17], [17, 18], [18, 19]]  # Right leg

    def openpose_joints(self):
        '''
        for openpose and RGBD-Pose
        openPose,RGBD-Pose
        :return:
        '''

        self.joint_count = 19
        self.joint_nose = 0
        self.joint_nect = 1
        self.joint_head = self.joint_nose
        joint_waist = [8, 11]  # 腰,RHip,LHip
        self.joint_shoulder_left = 2
        self.joint_shoulder_right = 5
        self.joint_arm_left = [2, 3, 4]  # 手臂
        self.joint_arm_right = [5, 6, 7]  # 手臂
        self.joint_lines = [[1, 2], [1, 5], [2, 3],
                            [3, 4], [5, 6], [6, 7],
                            [1, 8], [8, 9], [9, 10], [1, 11],
                            [11, 12], [12, 13], [1, 0], [0, 14],
                            [14, 16], [0, 15], [15, 17]]

    def custom_mpii_joints(self):
        '''
        custom mpii dataset
        :return:
        '''
        self.joint_nose = None
        self.joint_nect = 2
        self.joint_head = 3
        # joint_waist = [8, 11]  # 腰,RHip,LHip
        self.joint_shoulder_left = 4
        self.joint_shoulder_right = 5
        self.joint_lines = [[0, 1], [1, 2], [2, 3], [4, 1], [1, 5]]
        self.joint_count = 6


# 'kinect2','openpose',"custom_mpii"
# joint_config = Joints(type="kinect2")
# joint_config = Joints(type="openpose")
joint_config = Joints(type="custom_mpii")
