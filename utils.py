#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107170103
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
数据处理与特征工程辅助代码。
'''

import pickle
import warnings
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import njit
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec

warnings.filterwarnings('ignore')
###############################################################################


class GensimCallback(CallbackAny2Vec):
    '''计算每一个Epoch的词向量训练损失的回调函数。

    @Attributes:
    ----------
    epoch: {int-like}
    	当前的训练的epoch数目。
    verbose_round: {int-like}
    	每隔verbose_round轮次打印一次日志。
	loss: {list-like}
		保存每个epoch的Loss的数组。

    @References:
    ----------
    [1] https://stackoverflow.com/questions/54888490/gensim-word2vec-print-log-loss
    '''
    def __init__(self, verbose_round=3):
        self.epoch = 0
        self.loss = []

        if verbose_round == 0:
            verbose_round = 1
        self.verbose_round = verbose_round

    def on_epoch_end(self, model):
        '''在每个epoch结束的时候计算模型的Loss并且打印'''

        # 获取该轮的Loss值
        loss = model.get_latest_training_loss()
        self.loss.append(loss)

        if len(self.loss) == 1:
            pass
        else:
            loss_decreasing_precent = \
                (loss - self.loss[-2]) / self.loss[-2] * 100

            if divmod(self.epoch, self.verbose_round)[1] == 0:
                print('[{}]: word2vec loss: {:.2f}, decreasing {:.4f}%.'.format(
                    self.epoch, loss, loss_decreasing_precent))
        self.epoch += 1


class LoadSave():
    '''以*.pkl格式，利用pickle包存储各种形式（*.npz, list etc.）的数据。

    @Attributes:
    ----------
        dir_name: {str-like}
            数据希望读取/存储的路径信息。
        file_name: {str-like}
            希望读取与存储的数据文件名。
        verbose: {int-like}
            是否打印存储路径信息。
    '''
    def __init__(self, dir_name=None, file_name=None, verbose=1):
        if dir_name is None:
            self.dir_name = './data_tmp/'
        else:
            self.dir_name = dir_name
        self.file_name = file_name
        self.verbose = verbose

    def save_data(self, dir_name=None, file_name=None, data_file=None):
        '''将data_file保存到dir_name下以file_name命名。'''
        if data_file is None:
            raise ValueError('LoadSave: Empty data_file !')

        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 保存数据以指定名称到指定路径
        full_name = dir_name + file_name
        with open(full_name, 'wb') as file_obj:
            pickle.dump(data_file, file_obj, protocol=4)

        if self.verbose:
            print('[INFO] {} LoadSave: Save to dir {} with name {}'.format(
                str(datetime.now())[:-4], dir_name, file_name))

    def load_data(self, dir_name=None, file_name=None):
        '''从指定的dir_name载入名字为file_name的文件到内存里。'''
        if dir_name is None or not isinstance(dir_name, str):
            dir_name = self.dir_name
        if file_name is None:
            file_name = self.file_name
        if not isinstance(file_name, str) or not file_name.endswith('.pkl'):
            raise ValueError('LoadSave: Invalid file_name !')

        # 从指定路径导入指定文件名的数据
        full_name = dir_name + file_name
        with open(full_name, 'rb') as file_obj:
            data_loaded = pickle.load(file_obj)

        if self.verbose:
            print('[INFO] {} LoadSave: Load from dir {} with name {}'.format(
                str(datetime.now())[:-4], dir_name, file_name))
        return data_loaded


class CategoricalEncoder():
    '''对于输入的array对象的元素进行重新encoding，赋予新的编号。

    扫描数组内容，构建category2id的字典，通过字典替换的方法进行编码。

    @Attributes:
    ----------
    is_encoding_nan: {bool-like}
        是否对能够被np.isnan方法检测的对象进行编码，默认为不需要编码。
    cat2id: {dict-like}
        原始的category对象到新编码id的映射表。
    id2cat: {dict-like}
        新编码id到原始的category对象的映射表。
    default_nan_id: {int-like}
        默认的NaN表示的ID。

    '''
    def __init__ (self, is_encoding_nan=False):
        self.is_encoding_nan = is_encoding_nan
        self.default_nan_id = -999

    def fit(self, raw_data=None):
        '''扫描raw_data，利用字典记录raw_data的unique的id对象，返回None值。'''
        if isinstance(raw_data, (list, np.ndarray)):
            pass
        else:
            raise TypeError('Invalid input data type !')

        # 类别到id与id到类别的映射
        self.cat2id, self.id2cat = {}, {}

        global_id = 0
        # TODO(zhuoyin94@163.com): 此处NaN检测方法需要做测试
        for item in raw_data:
            if item not in self.cat2id and item is not None and ~np.isnan(item):
                self.cat2id[item] = global_id
                self.id2cat[global_id] = item
                global_id += 1
            elif (item is None or ~np.isnan(item)) and self.is_encoding_nan:
                pass

        return None

    def fit_transform(self, raw_data=None, inplace=True):
        '''扫描raw_data，利用字典记录raw_data的unique的id对象，返回转换后的数组。'''
        if isinstance(raw_data, (list, np.ndarray)):
            pass
        else:
            raise TypeError('Invalid input data type !')

        if inplace is False:
            transformed_data = np.zeros(len(raw_data), )

        # 类别到id与id到类别的映射
        self.cat2id, self.id2cat = {}, {}

        global_id = 0
        for idx, item in enumerate(raw_data):
            # FIXME(zhuoyin94@163.com): 不安全的检测方法
            if item not in self.cat2id and item is not np.nan:
                self.cat2id[item] = global_id
                self.id2cat[global_id] = item
                global_id += 1

            if item is not np.nan and inplace is True:
                raw_data[idx] = global_id
            elif item is not np.nan and inplace is False:
                transformed_data[idx] = global_id

        # Returen results
        if inplace:
            return raw_data
        else:
            return transformed_data

    def transform(self, raw_data=None):
        '''转换新输入的对象，并返回转换后的对象。'''
        if len(self.cat2id) == 0 or len(self.id2cat) == 0:
            raise ValueError('Please fit first !')
        if len(raw_data) == 0:
            return np.array([])

        for idx in range(len(raw_data)):
            if raw_data[idx] in self.cat2id:
                raw_data[idx] = self.cat2id[raw_data[idx]]
        return raw_data

    def reverse_transform(self, transformed_data=None):
        '''将被转换的数据恢复为原有的编码。'''
        if len(self.cat2id) == 0 or len(self.id2cat) == 0:
            raise ValueError('Please fit first !')
        if len(transformed_data) == 0:
            return np.array([])

        for idx in range(len(transformed_data)):
            if transformed_data[idx] in self.id2cat:
                transformed_data[idx] = self.id2cat[transformed_data[idx]]
        return transformed_data


def load_csv(dir_name, file_name, nrows=100, **kwargs):
    '''从指定路径dir_name读取名为file_name的*.csv文件，nrows指定读取前nrows行。'''
    if dir_name is None or file_name is None or not file_name.endswith('.csv'):
        raise ValueError('Invalid dir_name or file_name !')

    full_name = dir_name + file_name
    data = pd.read_csv(full_name, nrows=nrows, **kwargs)
    return data


def basic_feature_report(data_table, quantile=None):
    '''抽取Pandas的DataFrame的基础信息。'''
    if quantile is None:
        quantile = [0.25, 0.5, 0.75, 0.95, 0.99]

    # 基础统计数据
    data_table_report = data_table.isnull().sum()
    data_table_report = pd.DataFrame(data_table_report, columns=['#missing'])

    data_table_report['#uniques'] = data_table.nunique(dropna=False).values
    data_table_report['types'] = data_table.dtypes.values
    data_table_report.reset_index(inplace=True)
    data_table_report.rename(columns={'index': 'feature_name'}, inplace=True)

    # 分位数统计特征
    data_table_description = data_table.describe(quantile).transpose()
    data_table_description.reset_index(inplace=True)
    data_table_description.rename(
        columns={'index': 'feature_name'}, inplace=True)
    data_table_report = pd.merge(
        data_table_report, data_table_description,
        on='feature_name', how='left')

    return data_table_report


def save_as_csv(df, dir_name, file_name):
    '''将Pandas的DataFrame以*.csv的格式保存到dir_name+file_name路径。'''
    if dir_name is None or file_name is None or not file_name.endswith('.csv'):
        raise ValueError('Invalid dir_name or file_name !')

    full_name = dir_name + file_name
    df.to_csv(full_name, index=False)


class ReduceMemoryUsage():
    '''通过pandas的column的强制类型转换降低pandas数据表的内存消耗。
    返回经过类型转换后的DataFrame。

    @Attributes:
    ----------
    data_table: pandas DataFrame-like
        需要降低内存消耗的pandas的DataFrame对象。
    verbose: bool
        是否打印类型转换前与转换后的相关信息。

    @Return:
    ----------
    类型转换后的pandas的DataFrame数组。

    @References:
    ----------
    [1] https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
    [2] https://wizard
    forcel.gitbooks.io/ts-numpy-tut/content/3.html
    '''
    def __init__(self, data_table=None, verbose=True):
        self.data_table = data_table
        self.verbose = verbose

    def get_dataframe_types(self, data_table):
        '''获取pandas的DataFrame的每一列的数据类型，并返回类型的字典'''
        data_table_types = list(map(str, data_table.dtypes.values))

        type_dict = {}
        for ind, name in enumerate(data_table.columns):
            type_dict[name] = data_table_types[ind]
        return type_dict

    def reduce_memory_usage(self):
        '''对self.data_table的每一列进行类型转换，返回经过转换后的DataFrame。'''
        memory_usage_before_transformed = self.data_table.memory_usage(
            deep=True).sum() / 1024**2
        type_dict = self.get_dataframe_types(self.data_table)

        if self.verbose is True:
            print('[INFO] {} Reduce memory usage:'.format(
                str(datetime.now())[:-4]))
            print('----------------------------------')
            print('[INFO] {} Memory usage of data is {:.5f} MB.'.format(
                str(datetime.now())[:-4], memory_usage_before_transformed))

        # 扫描每一个column，若是属于float或者int类型，则进行类型转换
        for name in tqdm(list(type_dict.keys())):
            feat_type = type_dict[name]

            if 'float' in feat_type or 'int' in feat_type:
                feat_min = self.data_table[name].min()
                feat_max = self.data_table[name].max()

                if 'int' in feat_type:
                    if feat_min > np.iinfo(np.int8).min and \
                        feat_max < np.iinfo(np.int8).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int8)
                    elif feat_min > np.iinfo(np.int16).min and \
                        feat_max < np.iinfo(np.int16).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int16)
                    elif feat_min > np.iinfo(np.int32).min and \
                        feat_max < np.iinfo(np.int32).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int32)
                    else:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.int64)
                else:
                    if feat_min > np.finfo(np.float32).min and \
                        feat_max < np.finfo(np.float32).max:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.float32)
                    else:
                        self.data_table[name] = \
                            self.data_table[name].astype(np.float64)

        memory_usage_after_reduced = self.data_table.memory_usage(
            deep=True).sum() / 1024**2
        if self.verbose is True:
            print('\n----------------------------------')
            print('[INFO] {} Memory usage of data is {:.5f} MB.'.format(
                str(datetime.now())[:-4], memory_usage_after_reduced))
            print('[INFO] Decreased by {:.4f}%.'.format(
                100 * (memory_usage_before_transformed - \
                    memory_usage_after_reduced) \
                        / memory_usage_before_transformed))

        return self.data_table


class PurgedGroupTimeSeriesSplit:
    '''针对带有Group id（组id）数据的时间序列交叉验证集合生成类。

    生成针对带有Group id的数据的时序交叉验证集。其中训练与验证的
    Group之间可以指定group_gap，用来隔离时间上的关系。这种情况下
    group_id通常是时间id，例如天或者小时。

    @Parameters:
    ----------
        n_splits: {int-like}, default=5
            切分的集合数目。
        max_train_group_size: {int-like}, default=+inf
            训练集单个组的最大样本数据限制。
        group_gap: {int-like}, default=None
            依据group_id切分组时，训练组与测试组的id的gap数目。
        max_test_group_size: {int-like}, default=+inf
            测试集单个组的最大样本数据限制。

    @References:
    ----------
    [1] https://www.kaggle.com/gogo827jz/jane-street-ffill-xgboost-purgedtimeseriescv
    '''
    def __init__(self, n_splits=5, max_train_group_size=np.inf,
                 max_test_group_size=np.inf, group_gap=None, verbose=False):
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size
        self.max_test_group_size = max_test_group_size
        self.group_gap = group_gap
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        '''生成训练组与测试组的id索引，返回组索引的生成器。

        @Parameters:
        ----------
            X: {array-like} {n_samples, n_features}
                训练数据，输入形状为{n_samples, n_features}。
            y: {array-like} {n_samples, }
                标签数据，形状为{n_samples, }。
            groups: {array-like} {n_samples, }
                用来依据组来划分训练集与测试集的组id，必须为连续的，有序的组id。

        @Yields:
        ----------
            train_idx: ndarray
                依据group_id切分的训练组id。
            test_idx: ndarray
                依据group_id切分的测试组id。
        '''
        if X.shape[0] != groups.shape[0]:
            raise ValueError('The input shape mismatch!')

        # 构建基础参数
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size

        n_samples, n_splits, group_gap = len(X), self.n_splits, self.group_gap
        n_folds = n_splits - 1

        # 确定group_dict，用于存储每个组的样本index
        group_dict = {}
        unique_group_id, _ = np.unique(
            groups, return_index=True)

        # 扫描整个数据id list，构建group_dcit，{group_id: 属于该group的样本的idx}
        n_groups = len(unique_group_id)

        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]

        if n_folds > n_groups:
            raise ValueError(
                ('Cannot have number of folds={0} greater than'
                 ' the number of groups={1}').format(n_folds, n_groups))

        # test_group_size: 每个fold预留的test group的大小
        group_test_size = min(n_groups // n_splits, max_test_group_size)
        group_test_starts = range(n_groups - n_folds * group_test_size,
                                  n_groups, group_test_size)


        for group_test_start in group_test_starts:
            train_idx, gap_idx, test_idx = [], [], []

            # 计算train的group的起始位置
            group_train_start = max(0, group_test_start - \
                                       group_gap - max_train_group_size)

            for train_group_id in range(group_train_start,
                                        group_test_start - group_gap):
                raw_id = unique_group_id[train_group_id]
                if raw_id in group_dict:
                    train_idx.extend(group_dict[raw_id])

            for gap_id in range(group_test_start - group_gap,
                                group_test_start):
                raw_id = unique_group_id[gap_id]
                if raw_id in group_dict:
                    gap_idx.extend(group_dict[raw_id])

            for test_group_id in range(group_test_start,
                                       group_test_start + group_test_size):
                raw_id = unique_group_id[test_group_id]
                if raw_id in group_dict:
                    test_idx.extend(group_dict[raw_id])

            yield np.array(train_idx), np.array(gap_idx), np.array(test_idx)


class LiteModel:
    '''将模型转换为Tensorflow Lite模型，提升推理速度。目前仅支持Keras模型转换。

    @Attributes:
    ----------
    interpreter: {Tensorflow lite transformed object}
        利用tf.lite.interpreter转换后的Keras模型。

    @References:
    ----------
    [1] https://medium.com/@micwurm/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98
    '''

    @classmethod
    def from_file(cls, model_path):
        '''类方法。用于model_path下的模型，一般为*.h5模型。'''
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        '''类方法。用于直接转换keras模型。不用实例化类可直接调用该方法，返回
        被转换为tf.lite形式的Keras模型。

        @Attributes:
        ----------
        kmodel: {tf.keras model}
            待转换的Keras模型。

        @Returens:
        ----------
        经过转换的Keras模型。
        '''
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        '''为经过tf.lite.interpreter转换的模型构建构造输入输出的关键参数。

        TODO(zhuoyin94@163.com):
        ----------
        [1] 可添加关键字，指定converter选择采用INT8量化还是混合精度量化。
        [2] 可添加关键字，指定converter选择量化的方式：低延迟还是高推理速度？
        '''
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()

        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det['index']
        self.output_index = output_det['index']
        self.input_shape = input_det['shape']
        self.output_shape = output_det['shape']
        self.input_dtype = input_det['dtype']
        self.output_dtype = output_det['dtype']

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        ''' Like predict(), but only for a single record. The input data can be a Python list. '''
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]


# @njit
def njit_f1(y_true, y_pred_proba, threshold):
    '''计算F1分数，使用@njit进行加速'''
    y_true = y_true[:, 1]
    y_pred_proba = y_pred_proba[:, 1]
    y_pred_label = np.where(y_pred_proba > threshold, 1, 0)

    # https://www.itread01.com/content/1544007604.html
    tp = np.sum(np.logical_and(np.equal(y_true, 1),
                               np.equal(y_pred_label, 1)))
    fp = np.sum(np.logical_and(np.equal(y_true, 0),
                               np.equal(y_pred_label, 1)))
    # tn = np.sum(np.logical_and(np.equal(y_true, 1),
    #                            np.equal(y_pred_label, 0)))
    fn = np.sum(np.logical_and(np.equal(y_true, 1),
                               np.equal(y_pred_label, 0)))

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
