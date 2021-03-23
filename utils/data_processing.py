import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk

import math
import glob

from utils.set_env import set_env, get_aa_vec, make_paths

NCH = 6
PAD_LENGTH = 20
EP_PAD_LENGTH = 10
set_types = ['test', 'val', 'train']


def make_dup_over(df):
    df_out = None
    vc = df['antigen.epitope'].value_counts()
    n = vc[0]
    for i in range(vc.shape[0]):
        if df_out is None:
            df_out = df[df['antigen.epitope'] == vc.index[i]]
        else:
            df_out = df_out.append(df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=True))
    return df_out

def make_dup_under(df):
    df_out = None
    vc = df['antigen.epitope'].value_counts()
    n = vc[-1]
    for i in range(vc.shape[0]):
        if df_out is None:
            df_out = df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=False)
        else:
            df_out = df_out.append(df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=False))
    return df_out

def make_dup_geo(df):
    df_out = None
    vc = df['antigen.epitope'].value_counts()
    n0 = vc[0]
    n1 = vc[-1]
    n = int(np.sqrt(n0*n1))
    for i in range(vc.shape[0]):
        df_add = df[df['antigen.epitope'] == vc.index[i]]
        if n > df_add.shape[0]:
            replace = True
        else:
            replace = False
        if df_out is None:
            df_out = df_add.sample(n=n, replace=replace)
        else:
            df_out = df_out.append(df_add.sample(n=n, replace=replace))
    return df_out

def split_train_neg(df, train_ratio, val_thresh, df_all, neg_ratio, dup_train=False, dup_test=False, sample=None):
    ep_vc = df['antigen.epitope'].value_counts()
    ep_vc_filt = ep_vc[ep_vc >= val_thresh]
    ep_list = list(ep_vc_filt.index)
    df = df[df['antigen.epitope'].isin(ep_list)]
    df_train = df.groupby('antigen.epitope'
                         ).apply(pd.DataFrame.sample, frac=train_ratio
                         ).reset_index(level='antigen.epitope', drop=True)
    df_test = df.drop(df_train.index).sample(frac=1)
    train_ep_vc = df_train['antigen.epitope'].value_counts()
    train_ep_order = list(train_ep_vc.index)
    test_ep_vc = df_test['antigen.epitope'].value_counts()
    test_ep_vc = test_ep_vc[train_ep_order]
    
    df_train_new = df_train[['cdr3', 'antigen.epitope']]
    df_test_new = df_test[['cdr3', 'antigen.epitope']]
    
    if sample:
        if dup_train:
            if sample == 'under':
                df_train_new = make_dup_under(df_train_new)
            if sample == 'over':
                df_train_new = make_dup_over(df_train_new)
            if sample == 'geo':
                df_train_new = make_dup_geo(df_train_new)
        if dup_test:
            if sample == 'under':
                df_test_new = make_dup_under(df_test_new)
            if sample == 'over':
                df_test_undup = df_test_new.copy()
                df_test_new = make_dup_over(df_test_new)
            if sample == 'geo':
                df_test_new = make_dup_geo(df_test_new)
    
    train_sample = int(df_train_new.shape[0]*neg_ratio)
    test_sample = int(df_test_new.shape[0]*neg_ratio)
    test_undup_sample = int(df_test_undup.shape[0]*neg_ratio)
    
    df_train_new = df_train_new.append(df_all[['cdr3', 'antigen.epitope']].sample(train_sample))
    df_test_new = df_test_new.append(df_all[['cdr3', 'antigen.epitope']].sample(test_sample))
    df_test_undup = df_test_undup.append(df_all[['cdr3', 'antigen.epitope']].sample(test_undup_sample))
        
    split_dict = {'df_train': df_train_new,
                  'df_test': df_test_new,
                  'df_test_dedup': df_test_undup,
                  'vc_train': train_ep_vc,
                  'vc_test': test_ep_vc}
    return split_dict


    
def pad_seq(s, length):
    return s + ' '*(length-len(s))

def encode_seq(s, aa_vec):
    s_enc = np.empty((len(s), NCH), dtype=np.float32)
    for i, c in enumerate(s):
        s_enc[i] = aa_vec[c]
    return s_enc

def encode_seq_array(arr, aa_vec, pad=True, pad_length=PAD_LENGTH):
    if pad:
        arr = arr.map(lambda x: pad_seq(x, pad_length))
    enc_arr = arr.map(lambda x: encode_seq(x, aa_vec))
    enc_tensor = np.empty((len(arr), pad_length, NCH))
    for i, mat in enumerate(enc_arr):
        enc_tensor[i] = mat
    return enc_tensor

def encode_mcmc(s, aa_vec, pad_length):
    s = pad_seq(s, pad_length)
    s_vec = encode_seq(s, aa_vec)
    s_vec = np.expand_dims(s_vec, 0)
    return s_vec

def encode_seq_array_map(arr, aa_vec, pad=True, pad_length=PAD_LENGTH):
    if pad:
        arr = np.array(map(lambda x: pad_seq(x, pad_length), arr))
    enc_arr = np.array(map(lambda x: encode_seq(x, aa_vec), arr))
    enc_tensor = np.empty((len(arr), pad_length, NCH))
    for i, mat in enumerate(enc_arr):
        enc_tensor[i] = mat
    return enc_tensor

def encode_ep_class(arr, ep_vc):
    arr_class = arr.map(list(ep_vc.index).index)
    return np.array(arr_class)

def encode_ep_class_neg(arr, ep_vc):
    arr_class = arr.map(lambda x: list(ep_vc.index).index(x) if x in ep_vc.index else -1)
    return np.array(arr_class)

def make_data_dict_neg(split_dict, aa_vec, tcr_pad_len, ep_pad_len, from_archive=None):
    data_dict = split_dict
    for set_type in ['train', 'test', 'train_dedup', 'test_dedup']:
        label = 'df_' + set_type
        if label not in data_dict:
            continue
        data_dict['X_{}'.format(set_type)] = encode_seq_array(data_dict['df_{}'.format(set_type)]['cdr3'], aa_vec, True, tcr_pad_len)
        data_dict['y_enc_{}'.format(set_type)] = encode_seq_array(data_dict['df_{}'.format(set_type)]['antigen.epitope'], aa_vec, True, ep_pad_len)
        #print(label)
        data_dict['y_cls_{}'.format(set_type)] = encode_ep_class_neg(data_dict['df_{}'.format(set_type)]['antigen.epitope'], split_dict['vc_train'])
        #data_dict['y_hot_{}'.format(set_type)] = keras.utils.to_categorical(data_dict['y_cls_{}'.format(set_type)], len(data_dict['vc_train']))
    
    return data_dict

aa_vec = get_aa_vec()
aa_letters = list(aa_vec.keys())[:-1]

def proper_tcr(s):
    if type(s) == str:
        return s[0] == 'C' and s[-1] == 'F'
    else:
        return False
    
def proper_tcr_v2(s):
    if pd.isna(s):
        return False
    if type(s) == str:
        if s[0] == 'C' and (s[-1] == 'F' or s[-2:] == 'YV'):
            for letter in s:
                if letter not in ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']:
                    return False
            return True
        else:
            return False
    else:
        return False

def proper_epitope(s):
    if pd.isna(s) or type(s) != str:
        return False
    for letter in s:
        if letter not in ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']:
            return False
    return True
