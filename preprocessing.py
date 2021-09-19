#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107170104
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(preprocessing.py)对原始的*.txt数据进行预处理。
'''

import gc
import os
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

def parse_target_id(str_list=None):
    '''解析字符串形式的targid的list'''
    str_list = str_list[1:-1]
    str_list = str_list.split(',')
    return [int(i) for i in str_list]


def parse_time(str_list=None):
    '''解析字符串形式的time的list'''
    str_list = str_list[1:-1]
    str_list = str_list.split(',')
    return [int(np.float64(i)) for i in str_list]


if __name__ == '__main__':
    # 读入原始的训练与测试数据
    # -------------------------
    start_time = time.time()

    NROWS = None
    IS_SAVE_DATA = False
    TRAIN_PATH = './data/train/'
    TEST_PATH = './data/test/'

    train_df = pd.read_csv(
        TRAIN_PATH+'train.txt', header=None, nrows=NROWS,
        names=['pid', 'label',
               'gender', 'age',
               'targid', 'time',
               'province', 'city',
               'model', 'make'])
    test_df = pd.read_csv(
        TEST_PATH+'apply_new.txt', header=None, nrows=NROWS,
        names=['pid', 'gender',
               'age', 'targid',
               'time', 'province',
               'city', 'model', 'make'])
    test_df['label'] = np.nan

    curr_time = time.time()
    print('[INFO] took {} loading end...'.format(
        np.round(curr_time - start_time, 10)))

    total_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_df, test_df
    gc.collect()

    # 数据预处理与数据解析
    # -------------------------
    # 编码One-hot类型特征
    # *****************
    start_time = time.time()

    total_df['make'] = total_df['make'].apply(lambda x: x.split(' ')[-1])

    for feat_name in ['province', 'city', 'model', 'make']:
        encoder = LabelEncoder()
        total_df[feat_name] = encoder.fit_transform(total_df[feat_name].values)

    curr_time = time.time()
    print('[INFO] took {} oht encoding end...'.format(
        np.round(curr_time - start_time, 10)))

    # 处理字符串类型特征
    # *****************
    start_time = time.time()

    total_targid_list = total_df['targid'].apply(
        parse_target_id).values.tolist()
    total_timestamp_list = total_df['time'].apply(
        parse_time).values.tolist()
    timestamp_argidx = [np.argsort(item) for item in total_timestamp_list]

    unmatch_idx = 0
    for i in range(len(total_targid_list)):
        if len(total_targid_list[i]) == len(timestamp_argidx[i]):
            total_targid_list[i] = np.array(total_targid_list[i])[timestamp_argidx[i]]
        else:
            total_targid_list[i] = np.array(total_targid_list[i])
            unmatch_idx += 1
    total_timestamp_list = [np.array(item)[sorted_idx] for item, sorted_idx in \
                            zip(total_timestamp_list, timestamp_argidx)]

    total_df.drop(['targid', 'time'], axis=1, inplace=True)
    total_df['targid_len'] = [len(item) for item in total_targid_list]

    curr_time = time.time()
    print('[INFO] took {} str processing end...'.format(
        np.round(curr_time - start_time, 10)))

    # 基础指标的抽取
    # -------------------------
    start_time = time.time()

    train_df = total_df[total_df['label'].notnull()].reset_index(drop=True)
    test_df = total_df[total_df['label'].isnull()].reset_index(drop=True)
    total_df.fillna(-1, axis=1, inplace=True)

    curr_time = time.time()
    print('[INFO] took {} train test split processing...'.format(
        np.round(curr_time - start_time, 10)))

    # *****************
    start_time = time.time()

    for feat_name in ['gender', 'age', 'province', 'city', 'model', 'make']:
        tmp_val = train_df.groupby(feat_name)['label'].sum().values \
            / train_df.groupby(feat_name)['label'].count().values
        tmp_df = train_df.groupby(feat_name)['label'].count().reset_index()
        tmp_df['label_dist'] = tmp_val

        tmp_df.sort_values(
            by=['label_dist'], inplace=True, ascending=False)
        tmp_df.reset_index(inplace=True, drop=True)
        # print('++++++++++++')
        # print(tmp_df.iloc[:5])

    for feat_name in [['gender', 'age'], ['province', 'city'], ['model', 'make']]:
        tmp_val = train_df.groupby(feat_name)['label'].sum().values \
            / train_df.groupby(feat_name)['label'].count().values
        tmp_df = train_df.groupby(feat_name)['label'].count().reset_index()
        tmp_df['label_dist'] = tmp_val

        tmp_df.sort_values(
            by=['label_dist'], inplace=True, ascending=False)
        tmp_df.reset_index(inplace=True, drop=True)
        # print('++++++++++++')
        # print(tmp_df.iloc[:5])

    curr_time = time.time()
    print('[INFO] took {} groupby...'.format(
        np.round(curr_time - start_time, 10)))

    # 预处理数据的存储
    # -------------------------
    if IS_SAVE_DATA:
        file_processor = LoadSave(dir_name='./cached_data/')
        file_processor.save_data(
            file_name='total_df.pkl', data_file=total_df)
        file_processor.save_data(
            file_name='total_targid_list.pkl', data_file=total_targid_list)
        file_processor.save_data(
            file_name='total_timestamp_list.pkl', data_file=total_timestamp_list)
