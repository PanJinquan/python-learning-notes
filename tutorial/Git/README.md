# Git instructions

## Create Repository 

- Git global setup

```bash
git config --global user.name "潘金全"
git config --global user.email "panjinquan@dm-ai.cn"
```

- Create a new repository

```bash
git clone https://gitlab.dm-ai.cn/panjinquan/lightweight-face-recognition-tf.git
cd lightweight-face-recognition-tf
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

- Existing folder

```bash
cd existing_folder
git init
git remote add origin https://gitlab.dm-ai.cn/panjinquan/lightweight-face-recognition-tf.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

- Existing Git repository
```bash
cd existing_repo
git remote rename origin old-origin
git remote add origin https://gitlab.dm-ai.cn/panjinquan/lightweight-face-recognition-tf.git
git push -u origin --all
git push -u origin --tags
```

- updata `.gitignore` or remove git cached

```bash
git rm -r --cached .              
git add .                         
git commit -m 'update .gitignore' 
```

## Common Command line instructions


