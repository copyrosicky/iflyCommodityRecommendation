#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107202143
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
EDA part
'''

import gc
import os
import sys
import warnings
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder

from utils import LoadSave

np.random.seed(860919)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)

if __name__ == '__main__':
    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
  #   file_processor.save_data(dir_name='./cached_data/', file_name='total_df.pkl', data_file='./cached_data/train.txt')
    total_df = file_processor.load_data(file_name='total_df.pkl')
    total_df.fillna(-1, axis=1, inplace=True)

    total_feat_mat = None
    
    # tarid 的pdf
    fig, ax = plt.subplots(figsize=(10, 4))
    ax = sns.distplot(total_df['targid_len'], rug=False, kde=True, bins=150)

    ax.grid(True)
    # ax.set_ylabel("Score", fontsize=10)
    # ax.legend(fontsize=10)
    ax.set_xlabel('targid_len', fontsize=10)
    ax.set_ylabel('density', fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax = sns.boxplot(total_df['targid_len'])

    ax.grid(True)
    # ax.set_ylabel("Score", fontsize=10)
    # ax.legend(fontsize=10)
    ax.set_xlabel('targid_len', fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()


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

    # 不同的feature下label的分布
    # feature值   对应取值的feature数量 其中label为1的比率
    for feat_name in ['gender', 'age', 'province', 'city', 'model', 'make']:
        # 根据feature的取值分类
        tmp_val = train_df.groupby(feat_name)['label'].sum().values \
                  / train_df.groupby(feat_name)['label'].count().values
        tmp_df = train_df.groupby(feat_name)['label'].count().reset_index()
        tmp_df['label_dist'] = tmp_val

        tmp_df.sort_values(
            by=['label_dist'], inplace=True, ascending=False)
        tmp_df.reset_index(inplace=True, drop=True)
        print('++++++++++++')
        print(tmp_df.iloc[:5])

    for feat_name in [['gender', 'age'], ['province', 'city'], ['model', 'make']]:
        tmp_val = train_df.groupby(feat_name)['label'].sum().values \
                  / train_df.groupby(feat_name)['label'].count().values
        tmp_df = train_df.groupby(feat_name)['label'].count().reset_index()
        tmp_df['label_dist'] = tmp_val

        tmp_df.sort_values(
            by=['label_dist'], inplace=True, ascending=False)
        tmp_df.reset_index(inplace=True, drop=True)
        print('++++++++++++')
        print(tmp_df.iloc[:5])

    curr_time = time.time()
    print('[INFO] took {} groupby...'.format(
        np.round(curr_time - start_time, 10)))

