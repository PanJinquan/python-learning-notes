# python-learning-notes

```
    ├── dataset
    ├── modules
    ├── demo
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


## Linux CMD

- remove git cached
> git rm -r --cached .              <br/>
> git add .                         <br/>
> git commit -m 'update .gitignore' <br/>

- 统计当前文件夹下文件的个数，包括子文件夹里的
> ls -lR|grep "^-"|wc -l
- 统计文件夹下目录的个数，包括子文件夹里的
> ls -lR|grep "^d"|wc -l




## pyinstaller打包
### 常用命令：pyinstaller -F -w  pyinstaller_demo.py
> -F 表示生成单个可执行文件(注意大小写) <br/>
> -w 表示去掉控制台窗口，这在GUI界面时非常有用。不过如果是命令行程序的话那就把这个选项删除吧！ <br/>
> -p 表示你自己自定义需要加载的类路径，一般情况下用不到 <br/>
> -i 表示可执行文件的图标 <br/>

### 常见的错误
- problem:'No module named 'setuptools._vendor'' OR  'str' object has no attribute 'items'
> pip install --upgrade setuptools
