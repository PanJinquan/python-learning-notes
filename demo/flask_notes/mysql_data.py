# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : mysql_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-03 19:48:09
"""
import pymysql

def my_db(sql):
    conn=pymysql.connect(
        host='118.24.3.40',
        user='jxz',
        password='123456',
        db='jxz',
        charset='utf8',
        autocommit=True# 自动提交

    )
    cur=conn.cursor(cursor=pymysql.cursors.DictCursor)# 建立游标；默认返回二维数组，DictCursor指定返回字典；
    cur.execute(sql)#execute帮你执行sql
    res=cur.fetchall()#拿到全部sql执行结果
    cur.close()# 关闭游标
    conn.close()# 关闭数据库
    return res # 返回sql执行的结果