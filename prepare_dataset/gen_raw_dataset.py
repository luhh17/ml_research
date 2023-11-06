
# coding: utf-8

# In[6]:

# Import Packages
import os
import gc
from re import I
from syslog import LOG_SYSLOG
from threading import Timer
import time as timer
import random
import math
from turtle import forward
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader


def step1_preprocess():
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    data_path = '/mnt/HDD16TB/output_20220809'
    raw_folder = 'signal_raw'

    # Part 1: Fill missing and infinite values
    filled_folder = 'signal_filled'
    os.chdir(data_path)
    if not os.path.exists(filled_folder):
        os.mkdir(filled_folder)
    files = sorted([f for f in os.listdir(raw_folder) if f.endswith('.feather')])

    print('Fill missing values and infinite values')
    for file in tqdm(files):
        df = pd.read_feather(os.path.join(raw_folder, file))
        df = df[df['stkcd'].str.startswith(('00', '60'))]
        df.set_index(['date', 'stkcd'], inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        xs_mean = df.groupby('date').mean()
        df = df.fillna(xs_mean).astype(np.float32)
        df.reset_index(inplace=True)
        df.to_feather(os.path.join(filled_folder, file))
        del df, xs_mean

    # Part 2: Winsorize
    winsor_folder = 'signal_winsor'
    os.chdir(data_path)
    if not os.path.exists(winsor_folder):
        os.mkdir(winsor_folder)
    files = sorted([f for f in os.listdir(filled_folder) if f.endswith('.feather')])

    def winsorize(df, col=None, p=0.01, inplace=False):
        'Winsorize variables at a certain quantile'
        if col is None:
            col = df.columns
        lower = df[col].quantile(p, interpolation='higher')
        upper = df[col].quantile(1 - p, interpolation='lower')
        if inplace:
            df[col] = df[col].clip(lower, upper, axis=1)
        else:
            df_w = df.copy()
            df_w[col] = df_w[col].clip(lower, upper, axis=1)
            return df_w

    print('Winsorize')
    for file in tqdm(files):
        df = pd.read_feather(os.path.join(filled_folder, file))
        df.set_index(['date', 'stkcd'], inplace=True)
        df = df.groupby('date').apply(winsorize, p=0.05).astype(np.float32)
        df.reset_index(inplace=True)
        df.to_feather(os.path.join(winsor_folder, file))
        del df

    # Part 3: Normalize
    norm_folder = 'signal_norm'
    os.chdir(data_path)
    if not os.path.exists(norm_folder):
        os.mkdir(norm_folder)
    files = sorted([f for f in os.listdir(winsor_folder) if f.endswith('.feather')])

    print('Normalize')
    for file in tqdm(files):
        df = pd.read_feather(os.path.join(winsor_folder, file))
        df.set_index(['date', 'stkcd'], inplace=True)
        xs_mean = df.groupby('date').mean()
        xs_std = df.groupby('date').std()
        df = ((df - xs_mean) / xs_std).astype(np.float32)
        df.reset_index(inplace=True)
        df.to_feather(os.path.join(norm_folder, file))
        del df, xs_mean, xs_std

    # Part 4: Rank and normalize
    rank_folder = 'signal_rank'
    os.chdir(data_path)
    if not os.path.exists(rank_folder):
        os.mkdir(rank_folder)
    files = sorted([f for f in os.listdir(raw_folder) if f.endswith('.feather')])

    print('Rank and normalize')
    for file in tqdm(files):
        df = pd.read_feather(os.path.join(raw_folder, file))
        df = df[df['stkcd'].str.startswith(('00', '60'))]
        df.set_index(['date', 'stkcd'], inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.groupby('date').rank()
        xs_mean = df.groupby('date').mean()
        xs_std = df.groupby('date').std()
        df = (df - xs_mean) / xs_std
        xs_mean[xs_mean.notna()] = 0
        df = df.fillna(xs_mean).astype(np.float32)
        df.reset_index(inplace=True)
        df.to_feather(os.path.join(rank_folder, file))
        del df, xs_mean, xs_std


def step2_combine_feather():
    import os
    import pandas as pd
    from tqdm import tqdm

    output_data_path = '/mnt/SSD4TB_1/data_output'
    folder_list = ['data_raw', 'data_filled', 'data_winsor', 'data_norm', 'data_rank']

    # Output HDF5 files
    os.chdir(output_data_path)
    print('Output HDF5 files')
    for folder in tqdm(folder_list):
        files = sorted([f for f in os.listdir(folder) if f.endswith('.feather')])
        dfs = [pd.read_feather(f'{folder}/{file}') for file in files]
        combined_df = pd.concat(dfs, axis=0)
        del dfs
        combined_df = combined_df[combined_df['stkcd'].str.startswith(('00', '60'))]
        combined_df.sort_values(['date', 'stkcd'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        # Save an index file
        combined_df[['date', 'stkcd']].to_pickle('index.pkl')
        combined_df.to_hdf(f'{folder}.h5', key='data')
        del combined_df

    # Delete feather files
    os.chdir(output_data_path)
    print('Delete feather files')
    for folder in tqdm(folder_list):
        for file in os.listdir(folder):
            os.remove(f'{folder}/{file}')
        os.rmdir(folder)


def step3_filter_stocks():

    # Set Device
    # -1: CPU, 0~3: GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True
    ##########################################################

    data_path = '/mnt/HDD16TB/data_output'

    mkt_path = '/mnt/HDD16TB/Wind_data'
    file_list = ['data_raw.h5', 'data_filled.h5', 'data_winsor.h5', 'data_norm.h5', 'data_rank.h5']
    file_num = 2

    data_name = file_list[file_num].split('.')[0].split('_')[1]
    data = pd.read_hdf(data_path + '/' + file_list[file_num])

    index_file = 'index.pkl'
    index_df = pd.read_pickle(os.path.join(data_path, index_file))

    mkt_path = '/mnt/HDD16TB/Wind_data'
    mkt_file = 'AIndexEODPrices.pkl'
    mkt_df = pd.read_pickle(os.path.join(mkt_path, mkt_file))

    mkt_df['date'] = pd.to_datetime(mkt_df['date'], format='%Y%m%d')
    mkt_df['date'] = mkt_df['date'].shift(1)
    mkt_df['lg_open'] = mkt_df['open'].shift(-1)
    mkt_df['mktret'] = mkt_df['lg_open']/mkt_df['open'] - 1

    mkt_merge = pd.merge(data, mkt_df[['date','mktret']], on=['date'], how='left')
    mkt_merge['exret_open'] = mkt_merge['ret_open'] - mkt_merge['mktret']
    mkt_merge['exret_vwap'] = mkt_merge['ret_vwap'] - mkt_merge['mktret']

    del data

    #mkt_df[['date','mktret']].to_hdf(data_path+'/'+'data_mkt.h5', key='data')
    mkt_merge = mkt_merge[mkt_merge['stkcd']<800000].reset_index(drop=True)
    # mkt_merge.to_hdf(data_path+'/'+f'data_exmkt_nontb_{data_name}.h5', key='data')

    # No extreme return
    # mkt_merge = pd.read_hdf(data_path+'/'+f'data_exmkt_nontb_{data_name}.h5')
    mkt_merge = mkt_merge[np.abs(mkt_merge['ret_open'])<0.25].reset_index(drop=True)
    # mkt_merge.to_hdf(data_path+'/'+f'data_exmkt_nontb_{data_name}.h5', key='data')
    mkt_merge[['date','stkcd']].to_pickle(data_path+'/'+f'index_exmkt_nontb_{data_name}.pkl')

    # Top 2000
    # mkt_merge = pd.read_hdf(data_path+'/'+f'data_exmkt_nontb_{data_name}.h5')
    mkt_merge['mktcap_r'] = mkt_merge.groupby('date')['C.1.1'].rank(ascending=False, method='min')
    mkt_merge = mkt_merge[mkt_merge['mktcap_r'] <= 2000].reset_index(drop=True)
    del mkt_merge['mktcap_r']
    mkt_merge.to_hdf(data_path+'/'+f'data_exmkt_nontb_top2000_{data_name}.h5', key='data')
    mkt_merge[['date','stkcd']].to_pickle(data_path+'/'+f'index_exmkt_nontb_top2000_{data_name}.pkl')

    # Top 1000
    mkt_merge = pd.read_hdf(data_path+'/'+f'data_exmkt_nontb_{data_name}.h5')
    mkt_merge['mktcap_r'] = mkt_merge.groupby('date')['C.1.1'].rank(ascending=False, method='min')
    mkt_merge = mkt_merge[mkt_merge['mktcap_r']<=1000].reset_index(drop=True)
    del mkt_merge['mktcap_r']
    mkt_merge.to_hdf(data_path+'/'+f'data_exmkt_nontb_top1000_{data_name}.h5', key='data')
    mkt_merge[['date','stkcd']].to_pickle(data_path+'/'+f'index_exmkt_nontb_top1000_{data_name}.pkl')

    # Read
    data_path = '/mnt/SSD4TB_1/CH_data'
    data_exmkt_nontb_rank = pd.read_hdf(data_path+'/'+f'data_exmkt_nontb_top2000_{data_name}.h5')



    # count = data_exmkt_nontb_rank.groupby('date').transform(lambda X: len(X))
    #
    # CharsVars_avail = data_exmkt_nontb_rank.columns[4:-1].tolist()
    #
    # count = data_exmkt_nontb_rank.groupby('date')[CharsVars_avail[:5]].transform(lambda X: len(X)).mean()
    # count = pd.DataFrame(count)
    # count.to_csv(data_path + '/count.csv')
    #
    # mktret = pd.read_hdf(data_path+'/'+'data_mkt.h5')
