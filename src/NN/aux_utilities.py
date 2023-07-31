#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:31:17 2023

@author: jed169

Various utilities 

"""

import os
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from settings import datapath

def save_strlist_txt(path, lis):
    with open(path+'.txt','w') as fp:
        for item in lis:
            fp.write(f'{item}\n')
    fp.close()
    
def create_dir_ifnexists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def raise_naninf(df):
    '''
    Functionality is self-explanatory
    '''
    checknaninf = df.isin([np.inf, -np.inf]).sum().sum() or df.isnull().sum().sum()
    if checknaninf != 0:
        raise ValueError('Dataframe has nans or infs!')
    else:
        print('All is good. Dataframe has no nans or infs.')
        

def enum_dates_assets(df, date_idx = 'date', asset_idx = 'permno', save_path=f"{datapath}/current_train_val_test", job_number = 15):
    '''
    Function used for preprocessing of dataset before training.
    
    Dates will be enumerated starting from 1 on. Excess returns will have the label ret_e.
    Note that dates are always contemporaneous to returns. 
    Returns: 
    - a dataframe where first col is t_index, second col is date, third col is asset_id and fourth col is contemporaneous return.
        The rest of columns stay put.
        This will be used in the TrainValGAN class below. 
        The dates are ranked in increasing order (with repetitions), the t_index and asset_id also in increasing order.
    - dataframe with nr dates per asset
    - dataframe with nr assets per date
    '''
    for x in ['eret','e_ret']:
        if x in df.columns:
            df.rename(columns = {x:'ret_e'},inplace = True)
    if asset_idx == 'asset_id': 
        asset_idx = 'asset_idx'
        df.rename(columns ={'asset_id':asset_idx},inplace = True)
    if date_idx == 'date_id': 
        date_idx = 'date_idx'
        df.rename(columns ={'date_id':date_idx},inplace = True)
    
    df = df.sort_values(by = [date_idx, asset_idx])
    
    unique_dates = df[date_idx].unique()
    unique_assets = df[asset_idx].unique()
    enum_dates = pd.DataFrame(enumerate(unique_dates,1), columns = ['date_id',date_idx])
    enum_assets = pd.DataFrame(enumerate(unique_assets,1), columns = ['asset_id',asset_idx])
    
    df = enum_dates.merge(df, how = 'inner', on = [date_idx]).merge(enum_assets, how = 'inner', on = asset_idx)
    df_othercols = [col for col in df.columns if col not in ['date_id','asset_id','ret_e', asset_idx, date_idx]]
    df = df[['date_id','asset_id','ret_e']+df_othercols].sort_values(by = ['date_id','asset_id'])
    df = df.reset_index(drop=True).reset_index().rename(columns = {'index':'t_index'})
    
    if save_path is not None:
        enum_dates.to_excel(f"{save_path}/enum_dates_{job_number}.xlsx")
        enum_assets.to_excel(f"{save_path}/enum_assets_{job_number}.xlsx")
        
    dates_per_asset = df[['date_id','asset_id']].drop_duplicates().groupby('asset_id').size().reset_index().sort_values('asset_id')
    dates_per_asset.rename(columns = {0:'date_counts'},inplace = True)
    assets_per_date = df[['date_id','asset_id']].drop_duplicates().groupby('date_id').size().reset_index().sort_values('date_id')
    assets_per_date.rename(columns = {0:'asset_counts'}, inplace = True)

    return df, dates_per_asset, assets_per_date


def get_train_val_data(X,y, train_dates, val_dates, date_idx = 'date', reset_index = True):
    
    '''
    given set of dates train_dates, val_dates, date identifier, and X,y with a multiindex containing
    dates (or other identifier), this function creates
    the respective train and validation datasets.
    
    Input:
        X, (dataframe) of predictors
        y, (dataframe) of outcomes
        train_dates, (pd.series or dataframe) of validation dates only (no other columns)
        val_dates (pd.series or dataframe) of validation dates only (no other columns),
        date_idx, (str) identifier for merging variable
        
        
    Output:
        4 dataframes X_train, y_train, X_val, y_val with multi-index reset, if original dataframes 
        have multiindices
    '''
    X_train = X.loc[X.index.isin(train_dates,level = date_idx)].sort_index()
    y_train = y.loc[y.index.isin(train_dates,level = date_idx)].sort_index()
    X_val = X.loc[X.index.isin(val_dates,level = date_idx)].sort_index()
    y_val = y.loc[y.index.isin(val_dates,level = date_idx)].sort_index()
    
    if reset_index:
        return X_train.reset_index(), y_train.reset_index(), X_val.reset_index(), y_val.reset_index()
    else:
        return X_train, y_train, X_val, y_val

def df_lags_builder(df, seq_len):
    
    '''
    Takes a dataframe that has date in its index, and produces seq_len lags of it. Note: seq_len > 0 here. 
    
    Returns: a dataframe with lagged values from past to present (read left to right), that ends with the values from df 
    (i.e. ends contemporaneous values)
    '''
    df_1 = df.copy()
    df_1 = df_1.sort_index()
    df_final = df.copy()
    for i in range(1, seq_len+1):
        df_aux = df_1.shift(i)
        df_aux.rename(columns = {col:col+f'_lag_{i}' for col in df_1.columns}, inplace = True)
        df_final = pd.concat([df_aux, df_final],axis = 1)
   
    return df_final.dropna()

def l1_penalty(params,lam):
    '''
    Auxiliary function for train criterion adding a L1-noise to trained parameters.
    '''
    
    if isinstance(lam, str):
        lam = float(lam)
    
    l1loss = nn.L1Loss(reduction = 'sum')
    l1_norm = 0
    # loop is needed because zeros_like does not like generators
    for param in params:
        l1_norm += l1loss(param, target = torch.zeros_like(param))    
        
    return lam*l1_norm

class DatasetSimple(Dataset):
    
    '''
    Typical dataset class for pytorch dataloader purposes. 
    
    Additionally, it covers the case where we only need x by creating fictitious y. This is not memory efficient.
    x, y should come in as np.arrays with with specifier .astype(np.float32)
    '''
    
    def __init__(self, x,y=None):
        
        self.length = x.shape[0] 
        self.nr_features = x.shape[1]
        self.x = torch.from_numpy(x)
        if y is not None:
            self.y = torch.from_numpy(y)
            self.y_len = self.y.shape[1]
        else:
            self.y = torch.zeros(size = (self.x.shape[0],))
            self.y_len = 1

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return self.length
    
    def __nr_features__(self):
        return self.nr_features

def pred_metrics_NN(features, outcomes, model, weights=None, performance_measures=None, 
                    xscaler = None, target_dim = 1, yscaler = None, device = 'cuda'):
    
    '''
    Produces performance measures and predictions for a trained neural network 
    '''
    
    x, actual = features.copy(), outcomes.copy().reshape(-1,target_dim)
    
    if xscaler is not None:
        x = xscaler.transform(features).astype(np.float32)
        x = torch.from_numpy(x)
    else:
        if(isinstance(x,pd.DataFrame)):
            x = features.to_numpy().astype(np.float32)
        x = torch.from_numpy(x)
    
    x = x.to(device)
    model = model.to(device)
    model.eval()
    
    if weights is not None:
        model.load_state_dict(weights)
        
    with torch.set_grad_enabled(False):
        y_pred = model.forward(x).detach().cpu().numpy()
        if target_dim == 1: y_pred = y_pred.reshape(-1,1)
        if yscaler is not None:    
            predictions = yscaler.inverse_transform(y_pred)
            if target_dim == 1: predictions = predictions.reshape(-1,1)
        else:
            predictions = y_pred
            if target_dim == 1: predictions = predictions.reshape(-1,1)
            
    performance = [f(actual, predictions) for f in performance_measures]
    
    return performance, predictions

def performance_df(dic):
    '''
    Gathers performance dataframes into a single dictionary of dataframes
    '''
    perf_df = {}
    for loss_name, df in dic.items():
        df = pd.DataFrame.from_dict(df, orient = 'index',columns = [loss_name]).reset_index()
        df[['phase','epoch']] = pd.DataFrame(df.iloc[:,0].to_list(),index = df.index)
        df = df.iloc[:,1:]
        perf_df[loss_name] = df
    
    return perf_df
        