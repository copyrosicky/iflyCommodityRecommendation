#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107220043
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(compute_item_property.py)对item序列的一些基本property进行累计。
'''

import gc
import os
import sys
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

if __name__ == '__main__':
    IS_DEBUG = False

    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')

    if IS_DEBUG:
        total_df = file_processor.load_data(
            file_name='total_df.pkl').sample(5000)
        sampled_idx = list(total_df.index)

        total_targid_list = file_processor.load_data(
            file_name='total_targid_list.pkl')

        total_df.reset_index(drop=True, inplace=True)
        total_targid_list = [total_targid_list[i] for i in sampled_idx]
    else:
        total_df = file_processor.load_data(
            file_name='total_df.pkl')
        total_targid_list = file_processor.load_data(
            file_name='total_targid_list.pkl')
    total_df.fillna(-1, axis=1, inplace=True)

    # seq processing
    for i in range(len(total_targid_list)):
        total_targid_list[i] = [str(item) for item in total_targid_list[i]]

    # 累积每个item的特征分布特性
    # -------------------------
    item_meta_dist = {}
    item_meta_dist_tmp = {}

    # item to gender分布特征
    # *****************
    print('[INFO] {} gender processing...'.format(
        str(datetime.now())[:-7]))
    print('****************')

    encoder = OneHotEncoder(sparse=False)
    tmp_feat_mat = encoder.fit_transform(
        total_df['gender'].values.reshape(-1, 1))

    for idx, seq in tqdm(enumerate(total_targid_list), total=len(total_targid_list)):
        for item in seq:
            if item not in item_meta_dist:
                item_meta_dist[item] = tmp_feat_mat[idx]
            else:
                item_meta_dist[item] += tmp_feat_mat[idx]

    for key in item_meta_dist.keys():
        item_meta_dist[key] = \
            item_meta_dist[key] / np.sum(item_meta_dist[key])

    print('****************')
    print('[INFO] {} gender processing end...'.format(
        str(datetime.now())[:-7]))

    # item to age分布特征
    # *****************
    print('[INFO] {} age processing...'.format(
        str(datetime.now())[:-7]))
    print('****************')

    encoder = OneHotEncoder(sparse=False)
    tmp_feat_mat = encoder.fit_transform(
        total_df['age'].values.reshape(-1, 1))

    for idx, seq in tqdm(enumerate(total_targid_list), total=len(total_targid_list)):
        for item in seq:
            if item not in item_meta_dist_tmp:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]
            else:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]

    for key in item_meta_dist.keys():
        item_meta_dist[key] = np.concatenate(
            [item_meta_dist[key],
             item_meta_dist_tmp[key] / np.sum(item_meta_dist_tmp[key])])

    print('****************')
    print('[INFO] {} age processing end...'.format(
        str(datetime.now())[:-7]))

    # item to model分布特征
    # *****************
    print('[INFO] {} model processing...'.format(
        str(datetime.now())[:-7]))
    print('****************')

    encoder = OneHotEncoder(sparse=False)
    tmp_feat_mat = encoder.fit_transform(
        total_df['model'].values.reshape(-1, 1))

    for idx, seq in tqdm(enumerate(total_targid_list), total=len(total_targid_list)):
        for item in seq:
            if item not in item_meta_dist_tmp:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]
            else:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]

    for key in item_meta_dist.keys():
        item_meta_dist[key] = np.concatenate(
            [item_meta_dist[key],
             item_meta_dist_tmp[key] / np.sum(item_meta_dist_tmp[key])])

    print('****************')
    print('[INFO] {} model processing end...'.format(
        str(datetime.now())[:-7]))
    '''
    # item to province分布特征
    # *****************
    print('[INFO] {} province processing...'.format(
        str(datetime.now())[:-7]))
    print('****************')

    encoder = OneHotEncoder(sparse=False)
    tmp_feat_mat = encoder.fit_transform(
        total_df['province'].values.reshape(-1, 1))

    for idx, seq in tqdm(enumerate(total_targid_list), total=len(total_targid_list)):
        for item in seq:
            if item not in item_meta_dist_tmp:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]
            else:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]

    for key in item_meta_dist.keys():
        item_meta_dist[key] = np.concatenate(
            [item_meta_dist[key],
             item_meta_dist_tmp[key] / np.sum(item_meta_dist_tmp[key])])

    print('****************')
    print('[INFO] {} province processing end...'.format(
        str(datetime.now())[:-7]))

    # item to make分布特征
    # *****************
    print('[INFO] {} make processing...'.format(
        str(datetime.now())[:-7]))
    print('****************')

    encoder = OneHotEncoder(sparse=False)
    tmp_feat_mat = encoder.fit_transform(
        total_df['make'].values.reshape(-1, 1))

    for idx, seq in tqdm(enumerate(total_targid_list), total=len(total_targid_list)):
        for item in seq:
            if item not in item_meta_dist_tmp:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]
            else:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]

    for key in item_meta_dist.keys():
        item_meta_dist[key] = np.concatenate(
            [item_meta_dist[key],
             item_meta_dist_tmp[key] / np.sum(item_meta_dist_tmp[key])])

    print('****************')
    print('[INFO] {} province make end...'.format(
        str(datetime.now())[:-7]))
    '''
    # item to label分布特征
    # *****************
    print('[INFO] {} LABEL processing...'.format(
        str(datetime.now())[:-7]))
    print('****************')

    encoder = OneHotEncoder(sparse=False)
    tmp_feat_mat = encoder.fit_transform(
        total_df['label'].values.reshape(-1, 1))

    for idx, seq in tqdm(enumerate(total_targid_list), total=len(total_targid_list)):
        for item in seq:
            if item not in item_meta_dist_tmp:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]
            else:
                item_meta_dist_tmp[item] = tmp_feat_mat[idx]

    for key in item_meta_dist.keys():
        item_meta_dist[key] = np.concatenate(
            [item_meta_dist[key],
             item_meta_dist_tmp[key] / np.sum(item_meta_dist_tmp[key])])

    print('****************')
    print('[INFO] {} LABEL processing end...'.format(
        str(datetime.now())[:-7]))

    # 保存分布特性字典
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    file_processor.save_data(
        file_name='total_dist_dict.pkl',
        data_file=item_meta_dist)
