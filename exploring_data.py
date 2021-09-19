#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107202143
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(exploring_data.py)。
'''

import gc
import os
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    # 读入原始的训练与测试数据
    # -------------------------
    file_processor = LoadSave(dir_name='./cached_data/')
    total_df = file_processor.load_data(file_name='total_df.pkl')
    total_df.fillna(-1, axis=1, inplace=True)

    total_feat_mat = None
    
    # dis plot
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
