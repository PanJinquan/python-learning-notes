# Object-Detection SSD(TF-slim)

- TensorFlow object detection API: https://github.com/tensorflow/models

## 1. Convert TF graph to OpenCV DNN pbtxt

- 转换工具
- `pb`目录下是基于TensorFlow object detection API训练的MobileNet-SSD人体检测模型
- 训练过程参考: https://blog.csdn.net/linolzhang/article/details/87121875

```bash
python convert_tools/tf_text_graph_ssd.py --input pb/frozen_inference_graph.pb --output pb/opencv_graph.pbtxt --config pb/ssd_mobilenet_v2_coco.config
```

## 2.Run Demo
```bash
python ssd_detector.py
```


## Ref
- https://blog.csdn.net/qq_39267907/article/details/88994218



