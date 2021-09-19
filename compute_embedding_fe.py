#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107171211
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(compute_embedding_fe.py)计算item embedding的特征工程。
'''

import gc
import os
import sys
import warnings
import multiprocessing as mp
from datetime import datetime

import pandas as pd
import numpy as np
import seaborn as sns
from numba import njit
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from gensim.models import FastText, word2vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import LoadSave, GensimCallback

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2022
np.random.seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
sns.set(style='ticks', font_scale=1.2, palette='deep', color_codes=True)
###############################################################################


def compute_sg_embedding(corpus=None, is_save_model=True,
                         model_name='skip_gram_model',
                         **kwargs):
    '''利用gensim的SKip-Gram模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] {} Skip-Gram embedding start.'.format(
        str(datetime.now())[:-4]))
    print('-------------------------------------------')
    model = word2vec.Word2Vec(corpus, sg=1,
                              workers=mp.cpu_count(),
                              compute_loss=True,
                              callbacks=[GensimCallback(verbose_round=3)],
                              **kwargs)
    print('-------------------------------------------')
    print('[INFO] {} Skip-Gram embedding end. \n'.format(
        str(datetime.now())[:-4]))

    # 保存Embedding模型
    # ---------------------------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1)

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name),
            data_file=model)
    return model


def compute_cbow_embedding(corpus=None, is_save_model=True,
                           model_name='cbow_model',
                           **kwargs):
    '''利用gensim的CBOW模型训练并保存词向量。语料输入形式为：
        [['1', '2', '3'],
        ...,
        ['10', '23', '65', '9', '34']]
    '''
    print('\n[INFO] CBOW embedding start at {}'.format(
        str(datetime.now())[:-4]))
    print('-------------------------------------------')
    model = word2vec.Word2Vec(corpus, sg=0,
                              workers=mp.cpu_count(),
                              compute_loss=True,
                              callbacks=[GensimCallback(verbose_round=3)],
                              **kwargs)
    print('-------------------------------------------')
    print('[INFO] CBOW embedding end at {}\n'.format(
        str(datetime.now())[:-4]))

    # 保存Embedding模型
    # ---------------------------
    file_processor = LoadSave(
        dir_name='./pretraining_models/', verbose=1)

    if is_save_model:
        file_processor.save_data(
            file_name='{}.pkl'.format(model_name),
            data_file=model)
    return model


def compute_embedding(corpus, word2vec, embedding_size):
    '''将句子转化为embedding vector'''
    embedding_mat = np.zeros((len(corpus), embedding_size))

    for ind, seq in enumerate(corpus):
        seq_vec, word_count = np.zeros((embedding_size, )), 0
        for word in seq:
            if word in word2vec:
                seq_vec += word2vec[word]
                word_count += 1

            if word_count != 0:
                embedding_mat[ind, :] = seq_vec / word_count
    return embedding_mat


if __name__ == '__main__':
    CBOW_MODEL_NAME = 'cbow_model'
    SG_MODEL_NAME = 'skip_gram_model'
    EMBEDDING_DIM = 128
    TFIDF_DIM = 200

    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    total_targid_list = file_processor.load_data(
        file_name='total_targid_list.pkl')
    total_timestamp_list = file_processor.load_data(
        file_name='total_timestamp_list.pkl')

    # targid特征抽取部分
    # -------------------------

    # word2vec embedding
    # *****************
    for i in range(len(total_targid_list)):
        total_targid_list[i] = total_targid_list[i].split(' ')

    if CBOW_MODEL_NAME:
        file_processor = LoadSave(dir_name='./pretraining_models/')
        cbow_model = file_processor.load_data(
            file_name=CBOW_MODEL_NAME+'.pkl')
    else:
        cbow_model = compute_cbow_embedding(
            corpus=total_targid_list, negative=20,
            min_count=2, window=5,
            vector_size=EMBEDDING_DIM, epochs=30)

    if SG_MODEL_NAME:
        file_processor = LoadSave(dir_name='./pretraining_models/')
        sg_model = file_processor.load_data(
            file_name=SG_MODEL_NAME+'.pkl')
    else:
        sg_model = compute_sg_embedding(
            corpus=total_targid_list, negative=20,
            min_count=2, window=5,
            vector_size=EMBEDDING_DIM, epochs=15)

    # 计算句子向量
    # *****************
    cbow_embedding_mat = compute_embedding(
        total_targid_list, cbow_model.wv, EMBEDDING_DIM)
    cbow_embedding_mat = csr_matrix(cbow_embedding_mat)

    sg_embedding_mat = compute_embedding(
        total_targid_list, sg_model.wv, EMBEDDING_DIM)
    sg_embedding_mat = csr_matrix(sg_embedding_mat)

    # Spare matrix
    total_feat_mat = hstack(
        [cbow_embedding_mat, sg_embedding_mat]).tocsr()

    # 存储Embedding特征工程结果
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    file_processor.save_data(
        file_name='total_sp_embedding_mat.pkl', data_file=total_feat_mat)
