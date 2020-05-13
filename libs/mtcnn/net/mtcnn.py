# -*-coding: utf-8 -*-

import torch
import os, math
import PIL.Image as Image
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from net.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):

    def __init__(self,pnet_path):
        super(PNet, self).__init__()
        '''
        suppose we have input with size HxW, then
        after first layer: H - 2,
        after pool: ceil((H - 2)/2),
        after second conv: ceil((H - 2)/2) - 2,
        after last conv: ceil((H - 2)/2) - 4,
        and the same for W
        '''

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        # weights = np.load("./X2-face-detection-pnet.pth.tar", allow_pickle=True)[()]
        weights = np.load(pnet_path, allow_pickle=True)[()]

        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class RNet(nn.Module):

    def __init__(self,rnet_path):
        super(RNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        # weights = np.load("./X2-face-detection-rnet.pth.tar", allow_pickle=True)[()]
        weights = np.load(rnet_path, allow_pickle=True)[()]

        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class ONet(nn.Module):

    def __init__(self,onet_path):
        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        # weights = np.load("./X2-face-detection-onet.pth.tar", allow_pickle=True)[()]
        weights = np.load(onet_path, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim=1)
        return c, b, a


class MTCNN(nn.Module):
    def __init__(self,pnet_path,rnet_path,onet_path):
        super(MTCNN, self).__init__()
        # LOAD MODELS
        self.pnet = PNet(pnet_path)
        self.rnet = RNet(rnet_path)
        self.onet = ONet(onet_path)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.to()

    def to(self, *args, **kwargs):
        self.device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None:
            if not dtype.is_floating_point:
                raise TypeError('nn.Module.to only accepts floating point '
                                'dtypes, but got desired dtype={}'.format(dtype))

        def convert(t):
            return t.to(self.device, dtype if t.is_floating_point() else None, non_blocking)

        return self._apply(convert)

    def _generate_bboxes(self, probs, offsets, scale, threshold):
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a float numpy array of shape [n, m].
            offsets: a float numpy array of shape [1, 4, n, m].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            a float numpy array of shape [n_boxes, 9]
        """

        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # indices of boxes where there is probably a face
        inds = np.where(probs > threshold)

        if inds[0].size == 0:
            return np.array([])

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = np.vstack([
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score, offsets
        ])
        # why one is added?

        return bounding_boxes.T

    def _preprocess(self, img):
        """Preprocessing step before feeding the network.

        Arguments:
            img: a float numpy array of shape [h, w, c].

        Returns:
            a float numpy array of shape [1, c, h, w].
        """
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = (img - 127.5) * 0.0078125
        return img

    def run_first_stage(self, image, net, scale, threshold):
        """Run P-Net, generate bounding boxes, and do NMS.

        Arguments:
            image: an instance of PIL.Image.
            net: an instance of pytorch's nn.Module, P-Net.
            scale: a float number,
                scale width and height of the image by this number.
            threshold: a float number,
                threshold on the probability of a face when generating
                bounding boxes from predictions of the net.

        Returns:
            a float numpy array of shape [n_boxes, 9],
                bounding boxes with scores and offsets (4 + 1 + 4).
        """

        # scale the image and convert it to a float array
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')

        img = torch.FloatTensor(self._preprocess(img)).to(self.device)
        with torch.no_grad():
            output = net(img)
            probs = output[1].cpu().data.numpy()[0, 1, :, :]
            offsets = output[0].cpu().data.numpy()
            # probs: probability of a face at each sliding window
            # offsets: transformations to true bounding boxes

            boxes = self._generate_bboxes(probs, offsets, scale, threshold)
            if len(boxes) == 0:
                return None

            keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]

    def convert_to_square(self, bboxes):
        """Convert bounding boxes to a square form.

        Arguments:
            bboxes: a float numpy array of shape [n, 5].

        Returns:
            a float numpy array of shape [n, 5],
                squared bounding boxes.
        """

        square_bboxes = np.zeros_like(bboxes)
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = np.maximum(h, w)
        square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
        square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
        return square_bboxes

    def forward(self, rgb_image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7]):
        '''
        :param rgb_image: RGB PIL image
        :param min_face_size: a float number.
        :param thresholds: a list of length 3.
        :param nms_thresholds: a list of length 3.
        :return: bounding_boxes, a float numpy with shape=(num_boxes,5),[xmin,ymin,xmax,ymax,score]
                 landmarks, 5 facial landmarks points(x,y),a float numpy with shape=(num_boxes,5,2)
        '''
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        # BUILD AN IMAGE PYRAMID
        width, height = rgb_image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = self.run_first_stage(rgb_image, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(self.device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = self.convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(self.device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]
            # combine landmark points
            landmarks_list = []
            for landmark in landmarks:
                face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
                landmarks_list.append(face_landmarks)
            landmarks = np.asarray(landmarks_list)
        return bounding_boxes, landmarks
