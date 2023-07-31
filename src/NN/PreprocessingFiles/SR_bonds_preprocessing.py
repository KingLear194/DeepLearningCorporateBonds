#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:31:40 2022

@author: jed169
"""

import sys, os
import json, pickle

import pandas as pd
import numpy as np
sys.path.insert(0, '/ix/jduraj/Bonds/Code/NN')
from aux_utilities import raise_naninf, create_dir_ifnexists, df_lags_builder, enum_dates_assets

# define project name
project_name = 'SR_bonds'
job_number = 21

### datapath
dirpath='/ix/jduraj/'
os.chdir(dirpath)
datapath=f'{dirpath}Bonds/Data'
### Savepath
savepath=f'{dirpath}Bonds/Data/current_train_val_test/{project_name}'
create_dir_ifnexists(savepath)
# the dictionary to be saved as a config json file for gan
config_dict = {
    "multiprocessing": False,
    'train_val_split': 0.4,
    'validate': True,
    'optimizer_name': 'Adam', 
    'early_stopping': [8,0], 
    'momentum': 0.9,
    'leverage_constraint_type': 'L2',
    'leverage':0.1,
    'train_criterion_name': 'minus_mean_std_utility',
    'val_criterion_name': 'sharpe_ratio',
    'risk_aversion':0.01,
    'xscaling': True,
    'yscaling': True,
    'view_losses':True,
    'save_model': True,
    'get_betanet_labels': True,
    'verbose':1}

asset_id = 'cusip'
date_idx = 'date'

data = pd.read_pickle(f'{datapath}/final_dataset/dat_indreturn_macro_fundamentals_nomissing.pkl')
data.drop(columns = ['year', 'month', 'issuer_cusip', 'permno'], inplace = True)

macro_raw = pd.read_pickle(f'{datapath}/final_dataset/dat_macrodata.pkl')
macro_raw.drop(columns = ['year','month'], inplace = True)
# add forward spreads. These are not lagged.
fwdspreads=pd.read_pickle(f'{datapath}/final_dataset/dat_forwardspreads.pkl')
macro_raw = macro_raw.merge(fwdspreads,how='inner',on= 'date')
# add recursive CP8 factor
cp8=pd.read_pickle(f'{datapath}/final_dataset/dat_recursivecp08.pkl')
macro_raw=macro_raw.merge(cp8,how='inner',on= 'date')    
# add medchars. There are 41 of them. These are lagged by a month, so need to shift them up to have on same level as macro_raw so far.
medchars=pd.read_pickle(f'{datapath}/final_dataset/dat_equityreturn_medchar.pkl')
medchars=medchars.set_index('date').shift(-1).reset_index().dropna() # one needs to be careful how to apply shift here!
macro_raw=macro_raw.merge(medchars, how='inner', on='date')
# in total we have 132(macro+goyal welch data) + 1 CP08 factor + 9 fwdspreads + 42 medchars = 192 factors.
macro_raw = macro_raw.drop_duplicates()

#lag macro by a date since they're contemporaneous to dates
df = df_lags_builder(macro_raw.set_index('date').sort_index(), 1)
df = df.iloc[:,:macro_raw.shape[1]-1]
df = df.reset_index()
macro_raw = df.copy()
del df

macro_raw = macro_raw.sort_values(by='date')
# macro factors will be produced from 183 time series = 9 fwdspreads + 1 cp08+ 132 macro + 41 medchar 
macro_dim = macro_raw.shape[1]-1
# funda_dim= 82
funda_dim = data.drop(columns = ['ret_e']).shape[1]-2 # -2 due to date and asset_id

# equalize dates of macro and of chars/returns
chars_dates = data[['date']].drop_duplicates()
macro_raw = chars_dates.merge(macro_raw, how = 'inner', on  = 'date')
data = macro_raw[['date']].merge(data, how = 'inner', on = 'date')
# final dates
data_dates = data['date'].drop_duplicates().sort_values().astype('datetime64[ns]').reset_index(drop = True)
print(f'There are {len(data_dates)} many months of data ') # 449 months of data
final_dataset = data.merge(macro_raw, how = 'inner', on = 'date')
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
