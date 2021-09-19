#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202107220155
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(training_lstm_v1.py)采用序列形式，注入Meta-info。
'''

import gc
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, Activation, Add,
                                     BatchNormalization, Bidirectional, Dense,
                                     Dot, Dropout, Embedding,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Input, Lambda,
                                     LayerNormalization, SpatialDropout1D,
                                     concatenate, multiply, subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from dingtalk_remote_monitor import RemoteMonitorDingTalk, send_msg_to_dingtalk
from utils import GensimCallback, LoadSave, njit_f1

GLOBAL_RANDOM_SEED = 1995
# np.random.seed(GLOBAL_RANDOM_SEED)
# tf.random.set_seed(GLOBAL_RANDOM_SEED)
warnings.filterwarnings('ignore')

TASK_NAME = 'iflytek_commodity_recommendation_2021'
GPU_ID = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制Tensorflow只使用GPU ID编号的GPU
        tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU')

        # 限制Tensorflow不占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)
###############################################################################

def build_embedding_matrix(word2idx=None, word2embedding=None,
                           max_vocab_size=300, embedding_size=128,
                           oov_token=None, verbose=False):
    '''利用idx2embedding，组合重新编码过的word2idx。

    @Parameters:
    ----------
    word2idx: {dict-like}
        将词语映射到index的字典。键为词语，值为词语对应的index。
    word2embedding: {array-like or dict-like}
        可按照Index被索引的对象，idx2embedding对应了词语的向量，
        通常是gensim的模型对象。
    embedding_size: {int-like}
        embedding向量的维度。
    max_vocab_size: {int-like}
        词表大小，index大于max_vocab_size的词被认为是OOV词。
    oov_token: {str-like}
        未登录词的Token表示。
    verbose: {bool-like}
        是否打印tqdm信息。

    @Return:
    ----------
    embedding_mat: {array-like}
        可根据index在embedding_mat的行上进行索引，获取词向量

    @References:
    ----------
    [1] https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold
    [2] https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
    '''
    if word2idx is None or word2embedding is None:
        raise ValueError('Invalid Input Parameters !')
    embedding_mat = np.zeros((max_vocab_size+1, embedding_size))

    for word, idx in tqdm(word2idx.items(), disable=not verbose):
        if idx > max_vocab_size:
            continue

        if word in word2embedding:
            embedding_vec = word2embedding[word]
        else:
            embedding_vec = np.array([1] * embedding_size)

        embedding_mat[idx] = embedding_vec
    return embedding_mat


def build_embedding_sequence(train_corpus=None, test_corpus=None,
                             max_vocab_size=1024,
                             max_sequence_length=128,
                             word2embedding=None,
                             oov_token='UNK'):
    '''利用训练与测试语料，基于embedding_model构建用于神经网络的embedding矩阵。

    @Parameters:
    ----------
    train_corpus: {list-like}
        包含训练样本的文本序列。每一个元素为一个list，每一个list为训练集的一条句子。
    test_corpus: {list-like}
        包含测试样本的文本序列。每一个元素为一个list，每一个list为测试集的一条句子。
    max_vocab_size: {int-like}
        仅仅编码词频最大的前max_vocab_size个词汇。
    max_sequence_length: {int-like}
        将每一个句子padding到max_sequence_length长度。
    word2embedding: {indexable object}
        可索引的对象，键为词，值为embedding向量。
    oov_token: {str-like}
        语料中的oov_token。

    @Returen:
    ----------
    train_corpus_encoded: {list-like}
        经过编码与补长之后的训练集语料数据。
    test_corpus_encoded: {list-like}
        经过编码与补长之后的测试集语料数据。
    embedding_meta: {dict-like}
        包含embedding_mat的基础信息的字典。
    '''
    try:
        embedding_size = word2embedding['feat_dim']
    except KeyError:
        embedding_size = word2embedding.layer1_size

    # 拼接train与test语料数据，获取总语料
    # --------------------------------
    total_corpus = train_corpus + test_corpus

    # 序列化编码语料数据
    # --------------------------------
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(total_corpus)

    word2idx = tokenizer.word_index
    train_corpus_encoded = tokenizer.texts_to_sequences(train_corpus)
    test_corpus_encoded = tokenizer.texts_to_sequences(test_corpus)

    # 补长训练与测试数据，默认以0进行填补
    train_corpus_encoded = pad_sequences(
        train_corpus_encoded, maxlen=max_sequence_length)
    test_corpus_encoded = pad_sequences(
        test_corpus_encoded, maxlen=max_sequence_length)

    # 构造预训练的embedding matrix
    # --------------------------------
    embedding_mat = build_embedding_matrix(
        word2idx=word2idx,
        word2embedding=word2embedding,
        max_vocab_size=max_vocab_size,
        embedding_size=embedding_size,
        oov_token=oov_token)

    embedding_meta = {}
    embedding_meta['embedding_size'] = embedding_mat.shape[1]
    embedding_meta['max_len'] = max_sequence_length
    embedding_meta['max_vocab'] = max_vocab_size
    embedding_meta['embedding_mat'] = embedding_mat
    embedding_meta['tokenizer'] = tokenizer

    return train_corpus_encoded, test_corpus_encoded, embedding_meta


def tf_f1_score(y_true, y_pred):
    return tf.py_function(njit_f1, (y_true, y_pred, 0.5), tf.double)


class ScaledDotProductAttention():
    """Scaled-Dot-Product注意力机制，计算给定q, k, v之间的注意力权重。
    @Parameters:
    ----------
    attn_dropout: {float-like}
        注意力的dropout值.
    q, k, v: {tensor-like}
        Query, Key and Value 张量.
        q shape --- (batch_size, len_q, hidden_dim_q)
        k shape --- (batch_size, len_q, hidden_dim_q)
        v shape --- (batch_size, len_q, hidden_dim_k)
    @Return:
    ----------
    注意力权重与依据权重加权求和的Values向量。
    """
    def __init__(self, atten_dropout_rate=0.1):
        self.dropout = tf.keras.layers.Dropout(atten_dropout_rate)

    def __call__(self, q, k, v, mask):
        """q, k, v 代表了论文[2]中的Query, Key, Value"""
        sqrt_d_k = tf.sqrt(tf.cast(k.shape[-1], dtype="float32"))

        # The dot-product of Query vectors and Key vectors
        # q: (batch_size, len_q, d_q)
        # k: (batch_size, len_q, d_q == d_k)
        # atten: (batch_size, len_q, len_q)
        atten = tf.matmul(q, k, transpose_b=True)
        atten = atten / sqrt_d_k
        # atten = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / sqrt_d_k)([q, k])
        if mask is not None:
            mmask = tf.keras.layers.Lambda(lambda x:(-1e+9) * (1.-K.cast(x, "float32")))(mask)
            atten = tf.keras.layers.Add()([atten, mmask])

        # Normal the attention weights
        atten = tf.keras.layers.Activation("softmax")(atten)
        atten = self.dropout(atten)

        # Weighted sum of Value vectors
        # v: (batch_size, len_q, d_k)
        # atten: (batch_size, len_q, len_q)
        outputs = tf.keras.layers.Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten, v])
        return outputs, atten


def build_model(verbose=False, is_compile=True, **kwargs):
    '''构造行为序列分类的LSTM模型。'''

    # 基础参数
    # --------------------------------
    embedding_meta_targid = kwargs.pop('embedding_meta_targid', None)
    embedding_meta_targid_dist = kwargs.pop('embedding_meta_targid_dist', None)
    embedding_meta_static = kwargs.pop('embedding_meta_static', None)
    n_dense_feats = kwargs.pop('n_dense_feats', None)
    model_lr = kwargs.pop('model_lr', 0.0001)

    # 构造模型
    # --------------------------------
    # 构建输入层
    # ***********
    layer_input_seq = Input(shape=(embedding_meta_targid['max_len'], ))
    layer_input_dense_feats = Input(shape=(n_dense_feats, ))

    layer_input_province = Input(
        shape=(1, ), name='layer_province_input', dtype='int32'
    )
    layer_input_city = Input(
        shape=(1, ), name='layer_city_input', dtype='int32'
    )
    layer_input_model = Input(
        shape=(1, ), name='layer_model_input', dtype='int32'
    )
    layer_input_make = Input(
        shape=(1, ), name='layer_make_input', dtype='int32'
    )

    # Dense feature transformation
    layer_dense_feats = BatchNormalization()(layer_input_dense_feats)
    layer_dense_feats = Dense(128, activation='relu')(layer_dense_feats)
    layer_dense_feats = Dropout(0.3)(layer_dense_feats)

    # Shared pre-training embedding layer
    # ***********
    layer_shared_embedding = Embedding(
        embedding_meta_targid['max_vocab']+1,
        embedding_meta_targid['embedding_size'],
        input_length=embedding_meta_targid['max_len'],
        weights=[embedding_meta_targid['embedding_mat']],
        name='layer_shared_embedding',
        trainable=False)
    layer_shared_embedding_dist = Embedding(
        embedding_meta_targid_dist['max_vocab']+1,
        embedding_meta_targid_dist['embedding_size'],
        input_length=embedding_meta_targid_dist['max_len'],
        weights=[embedding_meta_targid_dist['embedding_mat']],
        name='layer_shared_embedding_dist',
        trainable=False)
    layer_embedding_province = Embedding(
        input_dim=embedding_meta_static['n_province'], output_dim=32, input_length=1, 
    )(layer_input_province)
    layer_embedding_city = Embedding(
        input_dim=embedding_meta_static['n_city'], output_dim=32, input_length=1, 
    )(layer_input_city)
    layer_embedding_model = Embedding(
        input_dim=embedding_meta_static['n_model'], output_dim=32, input_length=1, 
    )(layer_input_model)
    layer_embedding_make = Embedding(
        input_dim=embedding_meta_static['n_make'], output_dim=32, input_length=1, 
    )(layer_input_make)
    layer_static_embedding = concatenate(
        [layer_embedding_province, layer_embedding_city,
         layer_embedding_model, layer_embedding_make]
    )
    layer_static_embedding = tf.keras.layers.Flatten()(layer_static_embedding)

    # embedding encoding layer
    layer_encoding_embedding = layer_shared_embedding(layer_input_seq)
    layer_encoding_embedding_dist = layer_shared_embedding_dist(layer_input_seq)

    layer_encoding_embedding = concatenate(
        [layer_encoding_embedding,
         layer_encoding_embedding_dist])

    # LSTM encoding layer x
    # ***********
    x = Bidirectional(LSTM(
        256, return_sequences=True))(layer_encoding_embedding)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(0.4)(x)

    x = Bidirectional(LSTM(
        256, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(0.4)(x)

    x, x_atten = ScaledDotProductAttention(0.2)(x, x, x, mask=None)
    # Residual connection
    # ***********
    layer_encoding_concat = concatenate(
        [layer_encoding_embedding, x])

    # 组合，构建分类层
    layer_feat_concat = concatenate(
        [GlobalAveragePooling1D()(layer_encoding_concat),
         GlobalMaxPooling1D()(layer_encoding_concat),
         layer_dense_feats,
         layer_static_embedding])

    layer_total_feat = BatchNormalization()(layer_feat_concat)
    layer_total_feat = Dropout(0.5)(layer_total_feat)
    layer_total_feat = Dense(128, activation='relu')(layer_total_feat)

    layer_total_feat = BatchNormalization()(layer_total_feat)
    layer_total_feat = Dropout(0.5)(layer_total_feat)

    layer_output = Dense(2, activation='softmax')(layer_total_feat)

    # 编译模型
    # --------------------------------
    model = Model(
        [layer_input_seq, layer_input_dense_feats, layer_input_province,
         layer_input_city, layer_input_model, layer_input_make], layer_output)

    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0),
            optimizer=Adam(model_lr),
            metrics=[tf.keras.metrics.AUC(num_thresholds=1500), tf_f1_score])

    return model


if __name__ == '__main__':
    # 全局化的参数
    # ---------------------
    MAX_VOCAB_SIZE = 110000
    MAX_SENTENCE_LENGTH = 250

    N_FOLDS = 5
    MODEL_LR = 0.0005
    N_EPOCHS = 128
    BATCH_SIZE = 512
    EARLY_STOP_ROUNDS = 4
    IS_SEND_TO_DINGTALK = False
    MODEL_NAME = 'lstm_rtx3090'

    IS_TRAIN_FROM_CKPT = False
    CKPT_DIR = './ckpt/'
    CKPT_FOLD_NAME = '{}_GPU_{}_{}'.format(TASK_NAME, GPU_ID, MODEL_NAME)

    # 载入Token数据与Pre-trained Embedding数据
    # ---------------------
    file_processor = LoadSave(dir_name='./cached_data/')

    # 载入语料与原始数据
    # ***********
    total_df = file_processor.load_data(
        file_name='total_df.pkl')
    total_feats = file_processor.load_data(
        file_name='total_sp_stat_mat.pkl')
    total_targid_list = file_processor.load_data(
        file_name='total_targid_list.pkl')
    total_timestamp_list = file_processor.load_data(
        file_name='total_timestamp_list.pkl')

    # 载入预训练模型
    # ***********
    cbow_model = file_processor.load_data(
        dir_name='./pretraining_models/',
        file_name='cbow_model.pkl')
    sg_model = file_processor.load_data(
        dir_name='./pretraining_models/',
        file_name='skip_gram_model.pkl')
    total_dist_model = file_processor.load_data(
        dir_name='./cached_data/',
        file_name='total_dist_dict.pkl')

    vocab2vec = {}
    vocab_list = list(cbow_model.wv.key_to_index.keys())
    for word in vocab_list:
        vocab2vec[word] = np.concatenate(
            [cbow_model.wv.get_vector(word),
            sg_model.wv.get_vector(word)]
        )

        if 'feat_dim' not in vocab2vec:
            vocab2vec['feat_dim'] = len(vocab2vec[word])

    distvocab2vec = total_dist_model
    distvocab2vec['feat_dim'] = len(
        distvocab2vec[list(distvocab2vec.keys())[0]])

    # 切分训练与测试数据
    # ***********
    train_df = total_df[total_df['label'].notnull()]
    test_df = total_df[total_df['label'].isnull()]
    train_idx, test_idx = list(train_df.index), list(test_df.index)

    for i in range(len(total_targid_list)):
        total_targid_list[i] = [str(item) for item in total_targid_list[i]]
        total_targid_list[i] = ' '.join(total_targid_list[i])
    train_corpus = [total_targid_list[i] for i in train_idx]
    test_corpus = [total_targid_list[i] for i in test_idx]

    # Dense特征的训练与测试
    train_feats = total_feats[train_idx].todense()
    test_feats = total_feats[test_idx].todense()

    # Meta information特征的训练与测试
    dist_meta = {}
    dist_meta['n_province'] = int(total_df['province'].nunique())
    dist_meta['n_city'] = int(total_df['city'].nunique())
    dist_meta['n_model'] = int(total_df['model'].nunique())
    dist_meta['n_make'] = int(total_df['make'].nunique())

    train_feats_province = total_df['province'].values[train_idx]
    train_feats_city = total_df['city'].values[train_idx]
    train_feats_model = total_df['model'].values[train_idx]
    train_feats_make = total_df['make'].values[train_idx]

    test_feats_province = total_df['province'].values[test_idx]
    test_feats_city = total_df['city'].values[test_idx]
    test_feats_model = total_df['model'].values[test_idx]
    test_feats_make = total_df['make'].values[test_idx]

    # 构造pre-training embedding matrix与编码语料
    # ---------------------
    _, _, embedding_meta_dict = \
        build_embedding_sequence(
            train_corpus, test_corpus,
            max_vocab_size=MAX_VOCAB_SIZE,
            max_sequence_length=MAX_SENTENCE_LENGTH,
            word2embedding=vocab2vec
        )

    train_corpus_encoded, test_corpus_encoded, embedding_meta_dist_dict = \
        build_embedding_sequence(
            train_corpus, test_corpus,
            max_vocab_size=MAX_VOCAB_SIZE,
            max_sequence_length=MAX_SENTENCE_LENGTH,
            word2embedding=distvocab2vec
        )

    # 准备训练数据与训练模型
    # ---------------------
    n_train_samples = len(train_corpus)
    n_test_samples = len(test_corpus)

    folds = KFold(n_splits=N_FOLDS, shuffle=True,
                  random_state=GLOBAL_RANDOM_SEED)

    send_msg_to_dingtalk('\n++++++++++++++++++++++++++++', IS_SEND_TO_DINGTALK)
    INFO_TEXT = '[BEGIN][{}] {} #Training: {}, #Testing: {}'.format(
        MODEL_NAME, str(datetime.now())[:-7], n_train_samples, n_test_samples)
    send_msg_to_dingtalk(info_text=INFO_TEXT, is_send_msg=IS_SEND_TO_DINGTALK)

    # 验证时的各种分数，以及各类callback
    # --------------------------------
    valid_scores = np.zeros((N_FOLDS, 3))
    valid_pred_proba = np.zeros((n_train_samples, 2))
    test_pred_proba = []

    train_target = total_df['label'][total_df['label'].notnull()].values
    train_target_oht = to_categorical(train_target)

    early_stop = EarlyStopping(
        monitor='val_tf_f1_score', mode='max',
        verbose=1, patience=EARLY_STOP_ROUNDS,
        restore_best_weights=True)
    remote_monitor = RemoteMonitorDingTalk(
        is_send_msg=IS_SEND_TO_DINGTALK, model_name=MODEL_NAME)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_tf_f1_score',
        factor=0.7,
        patience=3,
        min_lr=0.000003)

    # Training the NN Classifier
    # --------------------------------
    send_msg_to_dingtalk('\n[INFO][{}] {} Start training...'.format(
        MODEL_NAME, str(datetime.now())[:-7]),
        is_send_msg=IS_SEND_TO_DINGTALK)
    print('==================================')
    for fold, (tra_id, val_id) in enumerate(
            folds.split(train_corpus_encoded, train_target_oht)):
        # 销毁所有内存中的图结构，便于多fold验证
        K.clear_session()
        gc.collect()

        # 划分训练与验证数据
        d_train, d_valid = train_corpus_encoded[tra_id], train_corpus_encoded[val_id]
        d_train_feats, d_valid_feats = train_feats[tra_id], train_feats[val_id]

        d_train_province, d_valid_province = train_feats_province[tra_id], train_feats_province[val_id]
        d_train_city, d_valid_city = train_feats_city[tra_id], train_feats_city[val_id]
        d_train_model, d_valid_model = train_feats_model[tra_id], train_feats_model[val_id]
        d_train_make, d_valid_make = train_feats_make[tra_id], train_feats_make[val_id]

        t_train, t_valid = train_target_oht[tra_id], train_target_oht[val_id]

        # 构造与编译模型
        # ***********
        model = build_model(
            embedding_meta_targid=embedding_meta_dict,
            embedding_meta_targid_dist=embedding_meta_dist_dict,
            embedding_meta_static=dist_meta,
            n_dense_feats=d_train_feats.shape[1],
            model_lr=MODEL_LR)

        # 完善ckpt保存机制
        # ***********
        # 如果模型名的ckpt文件夹不存在，创建该文件夹
        ckpt_fold_name_tmp = CKPT_FOLD_NAME + '_fold_{}'.format(fold)

        if ckpt_fold_name_tmp not in os.listdir(CKPT_DIR):
            os.mkdir(CKPT_DIR + ckpt_fold_name_tmp)

        # 如果指定ckpt weights文件名，则从ckpt位s置开始训练
        ckpt_file_name_list = os.listdir(CKPT_DIR + ckpt_fold_name_tmp)

        if IS_TRAIN_FROM_CKPT:
            if len(ckpt_file_name_list) != 0:
                latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR + ckpt_fold_name_tmp)
                model.load_weights(latest_ckpt)
        else:
            # https://www.geeksforgeeks.org/python-os-remove-method/
            try:
                for file_name in ckpt_file_name_list:
                    os.remove(os.path.join(CKPT_DIR + ckpt_fold_name_tmp, file_name))
            except OSError:
                print('File {} can not be deleted !'.format(file_name))

        ckpt_saver = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    CKPT_DIR + ckpt_fold_name_tmp,
                    MODEL_NAME + '_epoch_{epoch:02d}_valacc_{val_auc:.3f}.ckpt'),
                monitor='val_tf_f1_score',
                mode='max',
                save_weights_only=True,
                save_best_only=True),

        # fitting模型
        # ***********
        model_train_input = [
            d_train, d_train_feats, d_train_province,
            d_train_city, d_train_model, d_train_make
        ]
        model_valid_input = [
            d_valid, d_valid_feats, d_valid_province,
            d_valid_city, d_valid_model, d_valid_make
        ]
        model_test_input = [
            test_corpus_encoded, test_feats,
            test_feats_province, test_feats_city,
            test_feats_model, test_feats_make
        ]

        history = model.fit(
            x=model_train_input, y=t_train,
            batch_size=BATCH_SIZE,
            epochs=N_EPOCHS,
            shuffle=False,
            use_multiprocessing=True,
            validation_data=(model_valid_input, t_valid),
            callbacks=[early_stop, remote_monitor, reduce_lr, ckpt_saver])

        # 构造验证与测试数据预测的结果
        # ***********
        valid_pred_proba_tmp = model.predict(x=model_valid_input)
        valid_pred_label_tmp = np.argmax(
            valid_pred_proba_tmp, axis=1).reshape((-1, 1))

        test_pred_proba_tmp = model.predict(x=model_test_input)

        # 保存验证与测试的预测概率
        valid_pred_proba[val_id] = valid_pred_proba_tmp
        test_pred_proba.append(test_pred_proba_tmp)

        valid_scores[fold, 0] = fold
        valid_scores[fold, 1] = roc_auc_score(
            t_valid[:, 1].reshape(-1, 1),
            valid_pred_proba_tmp[:, 1].reshape(-1, 1),
            average='macro')
        valid_scores[fold, 2] = f1_score(
            t_valid[:, 1].reshape((-1, 1)), valid_pred_label_tmp, average='macro')

        INFO_TEXT = '[INFO][{}] {} folds {}[{}], valid roc_auc: {:.5f}, f1: {:.5f}'.format(
            MODEL_NAME, str(datetime.now())[:-7],
            fold+1, N_FOLDS,
            valid_scores[fold, 1],
            valid_scores[fold, 2])
        send_msg_to_dingtalk(INFO_TEXT, is_send_msg=IS_SEND_TO_DINGTALK)

    train_target = train_target.reshape((-1, 1))
    total_roc_auc = roc_auc_score(
        train_target,
        valid_pred_proba[:, 1].reshape(-1, 1))

    valid_pred_label_tmp = np.argmax(
        valid_pred_proba, axis=1).reshape((-1, 1))
    total_f1 = f1_score(
        train_target,
        valid_pred_label_tmp.reshape(-1, 1))

    INFO_TEXT = '[INFO][{}] {} total oof roc_auc: {:.5f}, mean: {:.5f}'.format(
        MODEL_NAME, str(datetime.now())[:-7],
        total_roc_auc, np.mean(valid_scores[:, 1]))
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=IS_SEND_TO_DINGTALK)

    INFO_TEXT = '[INFO][{}] {} total oof f1: {:.5f}, mean: {:.5f}'.format(
        MODEL_NAME, str(datetime.now())[:-7],
        total_f1, np.mean(valid_scores[:, 2]))
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=IS_SEND_TO_DINGTALK)

    print('==================================')
    INFO_TEXT = '[INFO][{}] {} End training...'.format(
        MODEL_NAME, str(datetime.now())[:-7])
    send_msg_to_dingtalk(INFO_TEXT, is_send_msg=IS_SEND_TO_DINGTALK)

    val_scores_df = pd.DataFrame(
        valid_scores, columns=['fold', 'valid_roc_auc', 'valid_f1'])
    y_pred_proba = np.mean(test_pred_proba, axis=0)
    y_pred_df = pd.DataFrame(
        y_pred_proba,
        columns=['y_pred_{}'.format(i) for i in range(y_pred_proba.shape[1])])
    oof_pred_df = pd.DataFrame(
        valid_pred_proba,
        columns=['oof_pred_{}'.format(i) for i in range(valid_pred_proba.shape[1])])

    # 保存预测的结果
    # -------------------------
    sub_file_name = '{}_{}_nfolds_{}_valrocauc_{}_valf1_{}'.format(
        len(os.listdir('./submissions/')) + 1,
        MODEL_NAME,
        N_FOLDS,
        str(np.round(val_scores_df['valid_roc_auc'].mean(), 5))[2:],
        str(np.round(val_scores_df['valid_f1'].mean(), 5))[2:])

    test_sub_df = test_df.copy()[['pid', 'label']]
    test_sub_df.rename(
        {'pid': 'user_id', 'label': 'category_id'}, axis=1, inplace=True)
    test_sub_df['category_id'] = np.argmax(
        y_pred_proba, axis=1).reshape((-1, 1))
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
