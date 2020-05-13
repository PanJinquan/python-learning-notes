# face-detection
基于ultra-light-fast-generic-face-detector-1MB的人脸检测和5个关键点检测模型
## 1. 目录结构

```
├── net 
│   ├── mtcnn.py      # MTCNN网络模型
│   └── box_utils.py  # MTCNN依赖的处理函数
├── data
│   ├── test_image                 # 测试图片,共10张
│   └── test_image_detection.json  # 测试图片的检测结果
├── demo.py           # MTCNN测试Demo文件
├── md5sum.txt        # MTCNN权重MD5文件
├── test.jpg          # 测试图片
├── X2-face-detection-onet.pth.tar  # MTCNN的onet权重文件
├── X2-face-detection-pnet.pth.tar  # MTCNN的pnet权重文件
├── X2-face-detection-rnet.pth.tar  # MTCNN的rnet权重文件
└── README.md
```
## 2. Platform
- hardware: Intel Core i7-8700 CPU @ 3.20GHz × 12, GPU GeForce RTX 2070 8G
- Python3.6
- Pillow-6.0
- Pytorch-1.0.1
- torchvision-0.2.2
- numpy-1.16.3
- opencv-python 3.4.1



## 3. I/O

```
Input: 处理待检测的BGR图像,type:PIL.Image,size:任意
Output: 输出人脸检测框和分数dets,type:numpy,shape=(num_boxes,5),[xmin,ymin,xmax,ymax,score]
        人脸landms的5个关键点(x,y):type:numpy,shape=(num_boxes,10),[x0,y0,x1,y1,...,x4,y4]
```

## 4. Run a demo

```bash
python demo.py 
```

输出结果

```  
bboxes:
[[233.34984  82.8728  342.6103  243.22647]
 [130.23299 139.56438 237.01491 277.63983]]
scores:
[[0.99997807]
 [0.9995683 ]]
landms:
[[[255.09235 145.65901]
  [303.036   144.15025]
  [272.47324 178.9415 ]
  [258.99652 199.27878]
  [304.8047  198.61676]]

 [[178.06424 188.51749]
  [222.07262 198.14188]
  [206.06091 224.33795]
  [165.17015 234.75359]
  [208.24365 243.11685]]]
```


