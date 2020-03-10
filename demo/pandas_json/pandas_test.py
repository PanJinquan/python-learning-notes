# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : pandas_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-29 17:05:43
"""
import pandas as pd
# df = pd.read_csv('data/test.csv')


import pandas as pd
import numpy as np
import json

def read_csv(filename,usecols=None):
    file = pd.read_csv(filename,usecols=usecols)
    df = pd.DataFrame(file)
    return df

def save_csv(filename,df):
    df.to_csv(filename, index=False, sep=',', header=True)


def pd_json_test(df):
    print("row,col={}".format(df.shape))
    print("{}".format(df))
    # pandas to json
    json_data=df.to_json()
    print("json_data={}".format(json_data))

    # json to pandas
    re_df=pd.DataFrame(json.loads(json_data))
    print("{}".format(re_df))

    save_csv(filename="data/xmc_for_student_re.csv", df=re_df)

if __name__=="__main__":
    save_path = "data/test2.csv"
    filename="data/xmc_for_student2.csv"
    df=read_csv(filename,usecols=None)
    pd_json_test(df)
    # print(df[0:3]) # get 0-2 row data
    # df['x']=[1,2,3,4]
    # print(df)
    #
    # df[1]=["cat",10,10,10]
    # print(df)

    # print(df.loc[0:2,["name","x"]])
    # dd=df.iloc[0:2,0:2]
    # print(df.iloc[0:2,0:2])


    # save_csv(save_path, df)