import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_accumulation(vc, title, xlabel, ylabel, xlog=False, index_thresh=True):
    plt.figure(figsize=(10,8))
    if index_thresh:
        unique_list = list(vc.index)
        unique_list.sort()
    else:
        unique_list = vc.unique()
    total = 0
    y = []
    for val in unique_list:
        if index_thresh:
            total += vc[val]
        else:
            total += (vc == val).sum()
        y.append(total)
    y = np.array(y)
    plt.plot(unique_list, y)
    if xlog:
        plt.xscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(which='both')
    plt.show()
    
def plot_hist(arr, title, xlabel, ylabel):
    plt.figure(figsize=(10,8))
    arr.hist()
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()

def profile_df(df):
    
    cdr3_len_vc = df['cdr3_length'].value_counts()
    ep_len_vc = df['epitope_length'].value_counts()
    ep_vc = df['antigen.epitope'].value_counts()
    
    print('\nCDR3 Length Value Counts\n')
    print(cdr3_len_vc)
    
    plot_hist(df['cdr3_length'],
             'CDR3 Length Histogram',
             'CDR3 Length',
             'CDR3 Count')
    plot_accumulation(cdr3_len_vc,
                     '# of CDR3s With Length Below Threshold',
                     'Length Threshold',
                     'CDR3 Count')
    
    print('\nEpitope Length Value Counts\n')
    print(ep_len_vc)
    
    plot_hist(df['epitope_length'],
             'CDR3 Epitope Length Histogram',
             'Epitope Length',
             'CDR3 Count')
    plot_accumulation(ep_len_vc,
                     '# of CDR3s With Epitope Length Below Threshold',
                     'Epitope Length Threshold',
                     'CDR3 Count')
    
    df_ep = df.drop_duplicates(subset=['antigen.epitope'])
    plot_hist(df_ep['epitope_length'],
             'Epitope Length Histogram',
             'Epitope Length',
             'Epitope Count')
    ep_len_count_vc = df_ep['epitope_length'].value_counts()
    plot_accumulation(ep_len_count_vc,
                     '# of Epitopes With Length Below Threshold',
                     'Epitope Length Threshold',
                     'Epitope Count')
    
    
    print('\nEpitope TCR Count\n')
    print(ep_vc)
    
    plot_accumulation(ep_vc,
                     '# Epitopes with TCR Count above Threshold',
                     'TCR Count Threshold',
                     'Epitope Count',
                     xlog=True,
                     index_thresh=False)