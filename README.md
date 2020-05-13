# python-learning-notes

```
    ├── dataset
    ├── modules
    ├── tutorial
    ├── script
    ├── utils
    ├── coco_demo.py
    ├── convert_voc_to_line_dataset.py
    ├── convert_voc_to_text_dataset.py
    ├── face_body_text_dataset.py
    ├── demo_test.py
    ├── README.md
    ├── setup.py
    └── voc_demo.py
```
##
- line_dataset
> [image_path,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id,...]
- text_dataset
> [label_id,x,y,w,h]

## Linux CMD

- remove git cached
> git rm -r --cached .              <br/>
> git add .                         <br/>
> git commit -m 'update .gitignore' <br/>

- 统计当前文件夹下文件的个数，包括子文件夹里的
> ls -lR|grep "^-"|wc -l
- 统计文件夹下目录的个数，包括子文件夹里的
> ls -lR|grep "^d"|wc -l
- 软链接文件:“ln –s 源文件 目标文件”(类似与windows的快捷方式)
> ln -s / /home/good/linkname
- get `val2017_gt` `file list` and save in val2017_gt.txt
> ls val2017_gt|less > val2017_gt.txt
- get `val2017_gt` `file list ID` and save in val2017_gt.txt
> ls val2017_gt|awk -F "." '{print $1}'|less > val2017_gt.txt
-
> ls facebank_DMFR_V1|head -n 3000 |awk '{print "cp -rf facebank_DMFR_V1/"$0, "DMAI" }'|sh
- Bash shell中的位置参数
```bash
$0是脚本本身的名字
$1是传递给该shell脚本的第一个参数
$2是传递给该shell脚本的第二个参数
```

## Train
- OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python demo.py

## pyinstaller打包
### 常用命令：pyinstaller -F -w  pyinstaller_demo.py
> -F 表示生成单个可执行文件(注意大小写) <br/>
> -w 表示去掉控制台窗口，这在GUI界面时非常有用。不过如果是命令行程序的话那就把这个选项删除吧！ <br/>
> -p 表示你自己自定义需要加载的类路径，一般情况下用不到 <br/>
> -i 表示可执行文件的图标 <br/>

### 常见的错误
- problem:'No module named 'setuptools._vendor'' OR  'str' object has no attribute 'items'
> pip install --upgrade setuptools
