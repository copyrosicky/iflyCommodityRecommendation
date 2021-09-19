#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107171925
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(xgb_training.py)利用特征工程结果，训练XGBoost模型。
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
GLOBAL_RANDOM_SEED = 4096
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')
###############################################################################

def xgb_clf_training_sparse(train_sp=None,
                            test_sp=None,
                            train_targets=None,
                            **kwargs):
    '''
    @Description:
    ----------
    训练XGBoost分类器，其中输入数据为CSR类型的稀疏矩阵。

    @Parameters:
    ----------
    train_sp: {csr-matrix}
        CSR格式的训练数据。
    test_sp: {csr-matirx}
        CSR格式的测试数据。
    train_targets: {array-like}
        训练集标签。

    @Return:
    ----------
    分类器的训练结果。
    '''
    if sparse.issparse(train_sp) != True or sparse.issparse(test_sp) != True:
        raise ValueError('Inputs are dense matrix, while',
                         'sparse matrixs are required !')

    # Initializing parameters
    # --------------------
    n_folds = kwargs.pop('n_folds', 5)
    params = kwargs.pop('params', None)
    shuffle = kwargs.pop('shuffle', True)
    n_classes = kwargs.pop('n_classes', 2)
    stratified = kwargs.pop('stratified', False)
    feat_names = kwargs.pop('feat_names', None)
    random_state = kwargs.pop('random_state', 2022)
    early_stop_rounds = kwargs.pop('early_stop_rounds', 200)

    if params is None:
        raise ValueError('Invalid training parameters !')

    # Preparing
    # --------------------
    if stratified == True:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle,
                                random_state=random_state)
    else:
        folds = KFold(n_splits=n_folds, shuffle=shuffle,
                      random_state=random_state)

    # Initializing oof prediction and feat importances dataframe
    # ---------------------------------
    feat_importance = pd.DataFrame(None)
    feat_importance['feat_name'] = feat_names
    scores = np.zeros((n_folds, 4))

    oof_pred = np.zeros((train_sp.shape[0], n_classes))
    y_pred = np.zeros((test_sp.shape[0], n_classes))

    # Training the Lightgbm Classifier
    # ---------------------------------
    print('\n[INFO] {} XGBoost training start(CSR)...'.format(
        str(datetime.now())[:-4]))
    print('==================================')
    print('-- train shape: {}, test shape: {}, total folds: {}'.format(
        train_sp.shape, test_sp.shape, n_folds))
    for fold, (tra_id, val_id) in enumerate(folds.split(train_sp, train_targets)):
        d_train, d_valid = train_sp[tra_id], train_sp[val_id]
        t_train, t_valid = train_targets[tra_id], train_targets[val_id]

        # Training the model
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            d_train, t_train,
            eval_set=[(d_valid, t_valid)],
            early_stopping_rounds=early_stop_rounds,
            verbose=0)

        feat_importance['fold_{}'.format(fold+1)] = clf.feature_importances_
        valid_pred_proba = clf.predict_proba(
            d_valid, ntree_limit=clf.best_iteration)
        y_pred += clf.predict_proba(
            test_sp, ntree_limit=clf.best_iteration) / n_folds

        oof_pred[val_id] = valid_pred_proba
        valid_pred_label = np.argmax(valid_pred_proba, axis=1).reshape((-1, 1))

        valid_f1 = f1_score(
            t_valid.reshape((-1, 1)), valid_pred_label, average='macro')
        valid_auc = roc_auc_score(
            t_valid.reshape((-1, 1)), valid_pred_proba[:, 1].reshape((-1, 1)))

        scores[fold, 0] = fold
        scores[fold, 1], scores[fold, 2] = valid_f1, valid_auc
        scores[fold, 3] = clf.best_iteration

        print('-- {} folds {}({}), valid f1: {:.5f}, auc: {:.5f}'.format(
            str(datetime.now())[:-4], fold+1, n_folds, valid_f1, valid_auc))
        params['random_state'] = params['random_state'] + 10086

    oof_pred_label = np.argmax(oof_pred, axis=1).reshape((-1, 1))
    total_f1 = f1_score(
        train_targets.reshape((-1, 1)),
        oof_pred_label.reshape((-1, 1)), average='macro')
    total_auc = roc_auc_score(
        train_targets.reshape((-1, 1)),
        oof_pred[:, 1].reshape((-1, 1)))
    print('-- total valid f1: {:.5f}, auc: {:.5f}'.format(
        total_f1, total_auc))
    print('==================================')
    print('[INFO] {} XGBoost training end(CSR)...'.format(
        str(datetime.now())[:-4]))

    scores = pd.DataFrame(
        scores, columns=['folds', 'valid_f1', 'valid_roc_auc', 'best_iters'])
    y_pred = pd.DataFrame(
        y_pred, columns=['y_pred_{}'.format(i) for i in range(n_classes)])
    oof_pred = pd.DataFrame(
        oof_pred, columns=['oof_pred_{}'.format(i) for i in range(n_classes)])

    return scores, feat_importance, oof_pred, y_pred


if __name__ == '__main__':
    # 全局参数列表
    # -------------------------
    N_FOLDS = 5
    IS_STRATIFIED = True

    # 检测是否有GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')

    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    total_df = file_processor.load_data(file_name='total_df.pkl')
    total_df = total_df[['pid', 'label']]
    sub_df = pd.read_csv('./data/submit_sample.csv')

    total_feats_stat = file_processor.load_data(
        file_name='total_sp_stat_mat.pkl')
    total_feats_embedding = file_processor.load_data(
        file_name='total_sp_embedding_mat.pkl')
    # total_feats_mat = hstack([total_feats_stat, total_feats_embedding]).tocsr()
    total_feats_mat = total_feats_embedding

    train_idx = np.arange(0, len(total_df))[total_df['label'].notnull()]
    test_idx = np.arange(0, len(total_df))[total_df['label'].isnull()]

    train_df = total_df.iloc[train_idx].reset_index(drop=True)
    test_df = total_df.iloc[test_idx].reset_index(drop=True)

    train_feats_mat = total_feats_mat[train_idx]
    test_feats_mat = total_feats_mat[test_idx]
    train_targets = train_df['label'].values.astype(int)

    del total_feats_stat, total_feats_embedding, total_feats_mat, total_df
    gc.collect()

    # 读入原始的训练与测试数据
    # -------------------------
    xgb_params = {
        'n_estimators': 5000,
        'max_depth': 5,
        'learning_rate': 0.07,
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric':'auc',
        'booster': 'gbtree',
        'colsample_bytree': 0.98,
        'colsample_bylevel': 0.98,
        'subsample': 0.985,
        'random_state': GLOBAL_RANDOM_SEED
    }

    # 模型训练若有GPU资源则采用GPU计算
    if gpus:
        xgb_params['tree_method'] = 'gpu_hist'
        xgb_params['gpu_id'] = 0

    val_scores_df, val_feat_importance_df, oof_pred_df, y_pred_df = \
        xgb_clf_training_sparse(
            train_feats_mat, test_feats_mat, train_targets,
            n_folds=N_FOLDS, stratified=IS_STRATIFIED,
            params=xgb_params, random_state=GLOBAL_RANDOM_SEED)

    # 保存预测的结果
    # -------------------------
    sub_file_name = '{}_xgb_nfolds_{}_valroc_auc_{}_valf1_{}'.format(
        len(os.listdir('./submissions/')) + 1,
        N_FOLDS,
        str(np.round(val_scores_df['valid_roc_auc'].mean(), 5))[2:],
        str(np.round(val_scores_df['valid_f1'].mean(), 5))[2:])

    test_sub_df = test_df.copy()
    test_sub_df.rename(
        {'pid': 'user_id', 'label': 'category_id'}, axis=1, inplace=True)
    test_sub_df['category_id'] = np.argmax(
        y_pred_df.values, axis=1).reshape((-1, 1))
    test_sub_df['user_id'] = test_sub_df['user_id'].astype(int)
    test_sub_df['category_id'] = test_sub_df['category_id'].astype(int)

    test_sub_df.to_csv('./submissions/{}.csv'.format(sub_file_name), index=False)
    print('[INFO] {} save sub to {}.csv'.format(
        str(datetime.now())[:-4], sub_file_name))
    print(test_sub_df['category_id'].value_counts() / len(test_sub_df))

    # 保存训练的oof的结果
    # -------------------------
    oof_pred_df['pid'] = train_df['pid'].values
    oof_pred_df['label'] = train_df['label'].values
    y_pred_df['pid'] = test_df['pid'].values

    file_processor = LoadSave(dir_name='./submissions_oof/')
    file_processor.save_data(
        file_name='{}_oof.pkl'.format(sub_file_name), data_file=oof_pred_df)
    file_processor.save_data(
        file_name='{}_ypred.pkl'.format(sub_file_name), data_file=y_pred_df)
