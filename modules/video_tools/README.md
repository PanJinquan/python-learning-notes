
#

## 图像合成视频：
```bash
ffmpeg -loop 1 -f image2 -i solvenew/%04d.jpg -vcodec libx264 -r 15 -t 7 video.mp4
```
注：-r 帧率， 
-t 总时长，
video.mp4表示输出视频，
-vcodec 视频编码选项，libx264 视频编码格式，
-i 图像目录，注意需以0001.jpg开始，
其他的开始无法生成视频，且以0001方式命名，
-f 生成视频来源只是图像，
-loop 循环从图像列表中提取图像直到图像提取完为止。