# tools
> 自己封装的工具类函数

## Linux
```bash
nohup python custom_train.py --batch_size=8 1>> train.log &
jobs -l
ps aux|grep python
kill 7080
nvidia-smi
```



## pyinstaller打包
### 常用命令：pyinstaller -F -w  pyinstaller_demo.py
> -F 表示生成单个可执行文件(注意大小写) <br/>
> -w 表示去掉控制台窗口，这在GUI界面时非常有用。不过如果是命令行程序的话那就把这个选项删除吧！ <br/>
> -p 表示你自己自定义需要加载的类路径，一般情况下用不到 <br/>
> -i 表示可执行文件的图标 <br/>

### 常见的错误
- problem:'No module named 'setuptools._vendor'' OR  'str' object has no attribute 'items'
> pip install --upgrade setuptools
