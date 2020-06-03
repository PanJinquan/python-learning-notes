# face-detection
基于ultra-light-fast-generic-face-detector-1MB的人脸检测和5个关键点检测模型
## 1. 目录结构

```
├── models 
├── data              # 测试图片,共10张
├── demo.py           # 测试Demo文件
├── md5sum.txt        # 权重MD5文件
├── test.jpg          # 测试图片
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
Input: 处理待检测的BGR图像,Size:任意
Output: bboxes, scores, landmarks:
        bboxes: <np.ndarray>: (num_boxes, 4)
        scores: <np.ndarray>: (num_boxes, 1)
        scores: <np.ndarray>: (num_boxes, 5, 2)
```

## 4. Run a demo

```bash
python demo.py 
```

输出结果

```  
bboxes:
[[332.7389  108.80746 476.60403 390.30975]
 [186.29468 199.22787 324.7784  439.92715]]
scores:
[[0.9999976]
 [0.9999739]]
landms:
[[[361.7588  224.60507]
  [420.83823 224.77036]
  [380.9847  285.49857]
  [366.53976 317.433  ]
  [421.47122 315.85056]]

 [[254.00157 289.82516]
  [304.01807 310.66592]
  [287.20694 352.56915]
  [235.70898 362.96088]
  [283.40564 381.3235 ]]]
```


