# face-detection
基于MTCNN的人脸检测和5个关键点检测模型
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
Input: 处理待检测的RGB图像,type:PIL.Image,size:任意
Output: 输出人脸检测框和分数bbox_score,type:numpy,shape=(num_boxes,5),[xmin,ymin,xmax,ymax,score]
        人脸landmarks的5个关键点(x,y):type:numpy,shape=(num_boxes,5,2)
```

## 4. Run a demo

```bash
python demo.py 
```

输出结果

```  
bbox_score:
[[ 69.48486808  58.12609892 173.92575279 201.95947894   0.99979943]]
landmarks:
[[[103.97721 119.6718 ]
  [152.35837 113.06249]
  [136.67535 142.62952]
  [112.62607 171.1305 ]
  [154.60092 165.12515]]]
```

## 5. 人脸检测和关键点测试结果
- 测试图片在`data/test_image`,
- 人脸检测和关键点测试结果保存在`data/test_image_detection.json`,格式为

```json
{
  "1.jpg": {
    "bbox_score": [[xmin,ymin,xmax,ymax,score],...],
    "landmarks": [[[x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4]],...]
  }
}

```
例如:

```json
{
  "1.jpg": {
    "bbox_score": [[69.4848680794239,58.12609891593456,173.92575278878212,201.95947894454002,0.9997994303703308]],
    "landmarks": [
      [
        [103.97721099853516,119.67179870605469],
        [152.35836791992188,113.06249237060547],
        [136.67535400390625,142.6295166015625],
        [112.62606811523438,171.1304931640625],
        [154.60092163085938,165.12515258789062]
      ]
    ]
  },
}
```

