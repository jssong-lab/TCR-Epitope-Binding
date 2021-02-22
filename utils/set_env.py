import os

import pandas as pd
import pickle as pk

from os.path import join

def make_paths():
    paths = {}
    paths['PROJ'] = '/home/groups/song/songlab2/alanluu2/TCR2Vec'
    paths['DATA'] = os.path.join(paths['PROJ'], 'data')
    paths['BIND'] = os.path.join(paths['DATA'], 'bind_data')
    paths['WORK'] = os.path.join(paths['PROJ'], 'workspace')
    paths['UTIL'] = os.path.join(paths['WORK'], 'utils')
    paths['AAVEC'] = os.path.join(paths['DATA'], 'aa_vec')
    paths['TCR'] = os.path.join(paths['DATA'], 'tcr')
    paths['TCRPROC'] = os.path.join(paths['TCR'], 'processed')
    paths['TCRENC'] = os.path.join(paths['TCR'], 'encoded')
    paths['RAW'] = os.path.join(paths['DATA'], 'raw')
    return paths

def get_aa_vec():
    paths = make_paths()
    aa_vec = pk.load(open(join(paths['AAVEC'], 'atchley.pk'), 'rb'))
    return aa_vec

def set_env():
    paths = make_paths()
    df = pd.read_csv(join(paths['BIND'], 'vdjdb.txt'), delimiter='\t')
    aa_vec = pk.load(open(join(paths['AAVEC'], 'atchley.pk'), 'rb'))
    df['cdr3_length'] = df['cdr3'].apply(len)
    df['epitope_length'] = df['antigen.epitope'].apply(len)
    df = df[df['gene'] == 'TRB']
    #df = df[df['mhc.class' == 'MHCI']]
    df = df.drop_duplicates(subset=['cdr3', 'antigen.epitope'])
    return df, paths, aa_vec