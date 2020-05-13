# 双目相机标定和恢复深度
- 参考github: https://github.com/aliyasineser/stereoDepth

## 1.相关依赖包
- opencv-contrib-python
```bash
pip install opencv-contrib-python # ximgproc
```

- PCL-Python
> https://github.com/strawlab/python-pcl/issues/317

```bash
sudo apt-get install libpcl-dev -y
pip install python-pcl
```

## 2.基本原理

### (1)视差图
pass

### (2)视差图转换为深度图
- 视差的单位是像素（pixel），深度的单位往往是毫米（mm）表示。
- 而根据平行双目视觉的几何关系，可以得到下面的视差与深度的转换公式：
> depth = ( f * baseline) / disp
- f表示归一化的焦距，也就是内参中的fx； 
- baseline是两个相机光心之间的距离，称作基线距离；
- disp是视差值。等式后面的均已知，深度值即可算出
- depth表示深度图


## 3.相关博客资料

- <真实场景的双目立体匹配（Stereo Matching）获取深度图详解> : https://www.cnblogs.com/riddick/p/8486223.html
- <双目测距理论及其python实现> (五分推荐) https://blog.csdn.net/dulingwen/article/details/98071584


