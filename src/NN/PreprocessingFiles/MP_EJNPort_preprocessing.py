#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:57:29 2022

@author: jed169
"""

import sys, os
import json, pickle
import pandas as pd
import numpy as np
sys.path.insert(0, '/ix/jduraj/Bonds/Code/NN')

from aux_utilities import raise_naninf, create_dir_ifnexists, df_lags_builder, enum_dates_assets

project_name = 'MP_EJNPort' 
job_number = 5 

dirpath='/ix/jduraj/'
os.chdir(dirpath)
datapath=f'{dirpath}Bonds/Data'
### Savepath
savepath=f'{dirpath}Bonds/Data/current_train_val_test/{project_name}'
create_dir_ifnexists(savepath)

config_dict = {
    "multiprocessing": False,
    'train_val_split': 0.4,
    'validate': True,
    'optimizer_name': 'Adam', 
    'early_stopping': [8,0], 
    'momentum': 0.9,
    'leverage_constraint_type': 'L2',
    'leverage':1,
    'exogenous_test_assets': False, 
    'train_criterion_name': 'mispricing_loss',
    'val_criterion_name': 'sharpe_ratio',
    'xscaling': True,
    'yscaling': True,
    'view_losses':True,
    'save_model': True,
    'get_betanet_labels': True,
    'verbose':1
    }

asset_id = 'asset_id'
date_idx = 'date'

# returns of Nozawa portfolios; returns are contemporaneous
returns = pd.read_pickle(f'{datapath}/final_dataset/dat_nozawa_int_eret_long.pkl')
returns.sort_index(level = ['date', 'asset_id'], inplace = True)
returns_dates = returns.reset_index()[['date']].drop_duplicates().reset_index()

# funda_data for bond portfolios (lagged one month to returns)
funda_data = pd.read_pickle(f"{datapath}/final_dataset/dat_bondpfs_fundamentals.pkl")
funda_data['date'] = pd.to_datetime(funda_data['date'])
funda_data['asset_id'] = funda_data['asset_id'].astype('str')
funda_dim = funda_data.shape[1]-2 # -2 for date and asset_id

print('funda_dim.date.min is ', funda_data.date.min())

# aggregate_bond_features; this file has features all lagged by one month
bond_agg_features = pd.read_pickle(f'{datapath}/final_dataset/dat_nozawa_int_eret_long_features.pkl')
# throw out eret, PCs and CP08
bond_agg_features.drop(columns = [f'PC{i}' for i in range(1,31)]+['ret_e', 'asset_id'], inplace = True)
bond_agg_features = bond_agg_features.iloc[:,:-40].drop_duplicates()
# we have 9 forward spreads + 50 meds here = 59 features

print('bond_agg_features.date.min is ', bond_agg_features.date.min())

macro_raw = pd.read_pickle(f'{datapath}/final_dataset/dat_macrodata.pkl')
macro_raw.drop(columns = ['year','month'], inplace = True)
# add forward spreads. These are not lagged.
# add medchars. There are 41 of them. These are lagged by a month, so need to shift them up to have on same level as macro_raw so far.
medchars=pd.read_pickle(f'{datapath}/final_dataset/dat_equityreturn_medchar.pkl')

medchars=medchars.set_index('date').shift(-1).reset_index().dropna() 
macro_raw=macro_raw.merge(medchars, how='inner', on='date')
macro_raw = macro_raw.drop_duplicates()

#lag macro by a date since they're contemporaneous to dates
df = df_lags_builder(macro_raw.set_index('date').sort_index(), 1)
df = df.iloc[:,:macro_raw.shape[1]-1]
df = df.reset_index()
macro_raw = df.copy()
del df

# merge with bond_agg_features
macro_raw = bond_agg_features.merge(macro_raw, how = 'inner', on='date').sort_values(by='date')

# macro factors will be produced from 233 time series = 60 bond_agg_features + 132 macro + 41 medchars from stocks
macro_dim = macro_raw.shape[1]-1

final_dataset = returns.reset_index().merge(funda_data, how = 'inner', on = ['date', 'asset_id']).merge(macro_raw, how = 'inner', on = 'date').sort_values(by = ['date','asset_id'])
raise_naninf(final_dataset)

# final dates
data_dates = final_dataset['date'].drop_duplicates().sort_values().astype('datetime64[ns]').reset_index(drop = True)
print(f'There are {len(data_dates)} many months of data ') # 449 months of data
print(f'Date range of all dataset is {final_dataset.date.min()} to {final_dataset.date.max()}.')

# preprocessing for MP networks
data, dates_per_asset, assets_per_date = enum_dates_assets(final_dataset, date_idx = date_idx, asset_idx = asset_id, 
                                            save_path=savepath, job_number = job_number)
raise_naninf(data)
'''extremely important to have dates and t_index aligned here'''
T = data.date_id.nunique()
returns = data.iloc[:,:4].sort_values('t_index')
X = data.drop(columns=['ret_e','asset_id']).sort_values('t_index')

trainval_test_split = int(T*0.7)
X.loc[X['date_id']<=trainval_test_split].to_pickle(f'{savepath}/X_train_val_{job_number}.pkl')
X.loc[X['date_id']>trainval_test_split].to_pickle(f'{savepath}/X_test_{job_number}.pkl')
returns.to_pickle(f'{savepath}/asset_returns_{job_number}.pkl')

dropout = 0.01 
batchnormalization = True

num_features = X.shape[1]-2 # -2 for t_index, date_id
config_dict['nr_features'] = num_features

ffn_params1 = {'num_features' : num_features, 
             'hidden_units' : [64,32,16,8],
             'output_size' : 1,
             'activation' : 'SiLU', 
             'batchnormalization' : batchnormalization, 
             'dropout' : dropout}

ffn_params2 = {'num_features' : num_features, 
             'hidden_units' : [64,64,32,32,16,16,8,8],
             'output_size' : 1,
             'activation' : 'SiLU', 
             'batchnormalization' : batchnormalization, 
             'dropout' : dropout}

ffn_params3 = {'num_features' : num_features, 
             'hidden_units' : [64,64,64,32,32,32,16,16,16,8,8,8,4],
             'output_size' : 1,
             'activation' : 'SiLU', 
             'batchnormalization' : batchnormalization, 
             'dropout' : dropout}

enet_params = {'num_features' : num_features, 
             'hidden_units' : [1],
             'output_size' : 1,
             'activation' : 'Identity', 
             'batchnormalization' : False, 
             'dropout' : None,
             'bias':False}

architecture_params = {'Model 1':['FFNStandard',ffn_params1], 
                       'Model 2': ['FFNStandard',ffn_params2],
                       'Model 3': ['FFNStandard',ffn_params3],
                       'Model 4': ['FFNStandard',enet_params]
                       }

archidefs = open(f'{savepath}/archidefs_{job_number}.pkl', 'wb')
pickle.dump(architecture_params, archidefs)
archidefs.close()

with open(f"{savepath}/config_file_{job_number}.json", "w") as outfile:
    json.dump(config_dict, outfile)
outfile.close()
