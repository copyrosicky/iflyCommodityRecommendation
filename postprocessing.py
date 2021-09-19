#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107172055
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(postprocessing.py)利用预测的oof概率进行后处理，包括stacking与阈值搜索。
'''
import os
import gc
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

import numba
from numba import njit

from utils import LoadSave, njit_f1, load_csv

# 设定全局随机种子，并且屏蔽warnings
GLOBAL_RANDOM_SEED = 2048
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
###############################################################################

def njit_search_best_threshold_f1(
        y_true, y_pred_proba, low, high, n_search=500):
    '''通过阈值枚举搜索最优的F1的阈值, 采用@njit技术加速计算'''
    # 依据y_pred_proba,确定可能的搜索范围
    unique_proba = np.unique(y_pred_proba)
    if len(unique_proba) < n_search:
        threshold_array = np.sort(unique_proba)
    else:
        threshold_array = np.linspace(low, high, n_search)

    # 便利threshold_array, 进行阈值搜索
    best_f1, best_threshold = 0, 0
    f1_list = []
    precision_list = []
    recall_list = []

    for threshold in threshold_array:
        f1_tmp, precision_tmp, recall_tmp = njit_f1(
            y_true, y_pred_proba, threshold)

        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_threshold = threshold

        f1_list.append(f1_tmp)
        precision_list.append(precision_tmp)
        recall_list.append(recall_tmp)

    return best_f1, best_threshold, f1_list, precision_list, recall_list, threshold_array


if __name__ == '__main__':
    # Load submissions
    # -------------------------
    NAME = '4_nfolds_5_valf1_73014_valacc_73018'

    file_processor = LoadSave(dir_name='./submissions_oof/')
    valid_oof_df = file_processor.load_data(
        file_name='{}_oof.pkl'.format(NAME))
    valid_oof_df['pred_label'] = valid_oof_df['oof_pred_1'] > 0.5

    test_pred_df = file_processor.load_data(
        file_name='{}_ypred.pkl'.format(NAME))
    test_pred_df['pred_label'] = test_pred_df['y_pred_1'] > 0.5

    '''
    # 是否搜索最优F1切分阈值
    # -------------------------
    if IS_SEARCH_THRESHOLD:
        oof_pred_proba = oof_pred_df['oof_pred_1'].values

        best_f1, best_threshold, _, _, _ = njit_search_best_threshold_f1(
            train_targets.reshape(-1, 1), oof_pred_proba,
            low=0.3, high=0.7, n_search=2000)
        print('[INFO] {} searched best f1: {:.4f}, threshold: {:.4f}'.format(
            str(datetime.now())[:-4], best_f1, best_threshold))
    else:
        best_threshold = 0.5
    '''

