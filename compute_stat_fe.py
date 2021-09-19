#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107170104
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(compute_stat_fe.py)进行统计特征工程。
'''

import gc
import os
import sys
import warnings

import pandas as pd
import numpy as np
from numba import njit
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import LoadSave

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################

@njit
def njit_compute_stat_feats(input_array=None):
    '''计算输入的array的一系列统计特征'''
    if len(input_array) == 1:
        return np.zeros((1, 6))
    stat_feats = np.zeros((1, 6))

    time_diff_array = input_array[1:] - input_array[:-1]
    stat_feats[0, 0] = np.mean(time_diff_array)
    stat_feats[0, 1] = np.std(time_diff_array)
    stat_feats[0, 2] = np.min(time_diff_array)
    stat_feats[0, 3] = np.max(time_diff_array)
    stat_feats[0, 4] = np.median(time_diff_array)
    stat_feats[0, 5] = input_array[-1] - input_array[0]

    return stat_feats


def comput_timestamp_stat_feats(input_list=None):
    '''接口方法，将input_list转为array并进行特征抽取'''
    array_list = np.array(input_list)
    return njit_compute_stat_feats(array_list)


def compute_targid_stat_feats(input_list=None):
    '''对targid序列抽取统计特征'''
    stat_feats = np.zeros((1, 2))

    stat_feats[0, 0] = len(np.unique(input_list))
    stat_feats[0, 1] = len(input_list)

    return stat_feats


def compute_tfidf_feats(corpus=None, max_feats=100, ngram_range=None):
    '''计算稀疏形式的TF-IDF特征'''
    if ngram_range is None:
        ngram_range = (1, 1)

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2',
                                 max_features=max_feats, max_df=1.0,
                                 analyzer='word', ngram_range=ngram_range,
                                 token_pattern=r'(?u)\b\w+\b')
    tfidf_array = vectorizer.fit_transform(corpus)

    return tfidf_array, vectorizer


if __name__ == '__main__':
    IS_DEBUG = False
    TFIDF_DIM = 200

    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')

    if IS_DEBUG:
        total_df = file_processor.load_data(
            file_name='total_df.pkl').sample(5000)
        sampled_idx = list(total_df.index)

        total_targid_list = file_processor.load_data(
            file_name='total_targid_list.pkl')
        total_timestamp_list = file_processor.load_data(
            file_name='total_timestamp_list.pkl')

        total_df.reset_index(drop=True, inplace=True)
        total_targid_list = [total_targid_list[i] for i in sampled_idx]
        total_timestamp_list = [total_timestamp_list[i] for i in sampled_idx]
    else:
        total_df = file_processor.load_data(
            file_name='total_df.pkl')
        total_targid_list = file_processor.load_data(
            file_name='total_targid_list.pkl')
        total_timestamp_list = file_processor.load_data(
            file_name='total_timestamp_list.pkl')
    total_df.fillna(-1, axis=1, inplace=True)

    total_feat_mat = None

    # 对时间戳做特征
    # -------------------------
    tmp_feat_list = list(
        map(comput_timestamp_stat_feats, total_timestamp_list))
    tmp_feat_array = np.vstack(tmp_feat_list)
    tmp_feat_array = tmp_feat_array / 1000 / 3600

    if total_feat_mat is None:
        total_feat_mat = csr_matrix(tmp_feat_array)

    tmp_feat_list = list(
        map(compute_targid_stat_feats, total_targid_list))
    tmp_feat_array = np.vstack(tmp_feat_list)
    total_feat_mat = hstack([total_feat_mat, tmp_feat_array]).tocsr()

    # targid特征抽取部分
    # -------------------------
    # seq processing
    for i in range(len(total_targid_list)):
        total_targid_list[i] = [str(item) for item in total_targid_list[i]]

    # TF-IDF
    # *****************
    for i in range(len(total_targid_list)):
        total_targid_list[i] = [str(item) for item in total_targid_list[i]]
        total_targid_list[i] = ' '.join(total_targid_list[i])
    tmp_feat_sp_array, encoder = compute_tfidf_feats(
        total_targid_list, max_feats=TFIDF_DIM)
    total_feat_mat = hstack([total_feat_mat, tmp_feat_sp_array]).tocsr()

    # 转csr矩阵形式进行处理
    # -------------------------
    feat_name_list = ['gender', 'model']
    for feat_name in feat_name_list:
        encoder = OneHotEncoder()
        tmp_sp_mat = encoder.fit_transform(
            total_df[feat_name].values.reshape(-1, 1))

        if total_feat_mat is None:
            total_feat_mat = tmp_sp_mat
        else:
            total_feat_mat = hstack([total_feat_mat, tmp_sp_mat]).tocsr()

    feat_name_list = ['age']
    for feat_name in feat_name_list:
        tmp_sp_mat = csr_matrix(total_df[feat_name].values.reshape(-1, 1))
        total_feat_mat = hstack([total_feat_mat, tmp_sp_mat]).tocsr()

    # 存储统计特征工程结果
    # -------------------------
    file_processor.save_data(
        file_name='total_sp_stat_mat.pkl', data_file=total_feat_mat)
