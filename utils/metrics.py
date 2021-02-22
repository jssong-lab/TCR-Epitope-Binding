import os
import keras

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from keras import backend as K
from keras import metrics
from keras.models import Sequential, model_from_json
from keras.layers import Convolution1D, Lambda
from keras.optimizers import Adam
from keras.layers import Input, MaxPool1D, Dropout, LSTM, Bidirectional
from keras.models import Model
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import UpSampling1D
from keras.layers import BatchNormalization
from keras.layers import Layer


"""
def viz_conf_mat(network_dict, X_train, y_train, X_test, y_test knn):
    test_embed = network_dict['TCR_Embed'].predict(data_dict['X_test'])
    train_embed = network_dict['TCR_Embed'].predict(data_dict['X_train'])
    knn_res = knn.fit(train_embed, data_dict['y_cls_train'])
    conf_mat = confusion_matrix(knn_res, data_dict['y_cls_test'])
    f, ax = plt.subplots()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    sns.heatmap(out, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    
    plt.xticks([0.5,1.5,2.5], ['a','b','c'])
    plt.show()"""
    
def viz_conf_mat(conf_mat, vc, title, acc_only=False, figsize=(17,8)):
    acc = np.trace(conf_mat)/np.sum(conf_mat)
    if acc_only:
        return acc
    normalized = np.divide(conf_mat.transpose(), np.sum(conf_mat, axis=1)).transpose()*100
    f, ax = plt.subplots(1, 2, figsize=figsize)
    sns.set(font_scale=2)
    sns.heatmap(conf_mat, annot=True, fmt=".1f", linewidths=.5, ax=ax[0], annot_kws={'size':14})
    ax[0].set_title(title)
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')
    tick_labels = list(vc.index)
    xticks = np.arange(len(tick_labels)) + 0.5
    yticks = xticks
    ax[0].set_xticks(xticks, tick_labels)
    ax[0].set_xticklabels(tick_labels, rotation=90)
    ax[0].set_yticks(yticks, tick_labels)
    ax[0].set_yticklabels(tick_labels, rotation=0)
    sns.heatmap(normalized, annot=True, fmt=".1f", linewidths=.5, ax=ax[1], annot_kws={'size':14})
    ax[1].set_title('Normalized ' + title + ' | Acc: {}'.format(str(acc)[:5]))
    ax[1].set_xlabel('Predicted Label', fontsize=16)
    tick_labels = list(vc.index)
    xticks = np.arange(len(tick_labels)) + 0.5
    yticks = xticks
    ax[1].set_xticks(xticks, tick_labels)
    ax[1].set_xticklabels(tick_labels, rotation=90)
    ax[1].set_yticks(yticks, tick_labels)
    ax[1].set_yticklabels(tick_labels, rotation=0)
    f.tight_layout()
    plt.show()
    return acc

def viz_knn(network_dict, train_embed, y_cls_train, test_embed, y_cls_test, knn):
    knn.fit(train_embed, y_test)
    res = knn.predict(test_embed)
    out = confusion_matrix(res, data_dict['y_cls_test']).transpose()