import pandas as pd

def make_dup_oversample(df):
    df_out = None
    vc = df['antigen.epitope'].value_counts()
    n = vc[0]
    for i in range(vc.shape[0]):
        if df_out is None:
            df_out = df[df['antigen.epitope'] == vc.index[i]]
        else:
            df_out = df_out.append(df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=True))
    return df_out

def make_dup_undersample(df):
    df_out = None
    vc = df['antigen.epitope'].value_counts()
    n = vc[-1]
    for i in range(vc.shape[0]):
        if df_out is None:
            df_out = df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=False)
        else:
            df_out = df_out.append(df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=False))
    return df_out

def make_dup_geometric_sample(df):
    df_out = None
    vc = df['antigen.epitope'].value_counts()
    n = vc[-1]
    for i in range(vc.shape[0]):
        if df_out is None:
            df_out = df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=False)
        else:
            df_out = df_out.append(df[df['antigen.epitope'] == vc.index[i]].sample(n, replace=False))
    return df_out