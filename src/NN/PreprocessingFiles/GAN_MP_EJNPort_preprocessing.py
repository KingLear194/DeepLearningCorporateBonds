#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 14:17:59 2022

@author: jed169
"""

import sys, os
import json, pickle
import pandas as pd
sys.path.insert(0, '/ix/jduraj/Bonds/Code/NN')
from aux_utilities import raise_naninf, create_dir_ifnexists, df_lags_builder, enum_dates_assets

project_name = 'GAN_MP_EJNPort'
job_number = 6

dirpath='/ix/jduraj/'
os.chdir(dirpath)
#datapath = os.path.expanduser("~/Desktop/Bonds/Data/")
datapath=f'{dirpath}Bonds/Data'
### codepath
#codepath = os.path.expanduser("~/Desktop/Bonds/Code/")
codepath = f'{dirpath}Bonds/Code'
### Savepath
savepath=f'{dirpath}Bonds/Data/current_train_val_test/{project_name}'
create_dir_ifnexists(savepath)

# the dictionary to be saved as a config json file for gan
# note: the values from here can be overrun through command line
config_dict = {
    "multiprocessing": False,
    'train_val_split': 0.4,
    'validate': True,
    'sub_epochs': (4,4,4),
    'optimizer_name': 'Adam', # other option is RMSprop
    'early_stopping': [8,0], 
    'sdf_momentum': 0.9,
    'moments_momentum': 0.9,
    'train_criterion_name': 'mispricing_loss',
    'val_criterion_name': 'sharpe_ratio',
    'xscaling': True,
    'yscaling': True,
    'unconditional_loss_weight': 1.0,
    'output_predictions': True,
    'view_losses':True,
    'save_model': True,
    'get_betanet_labels': True,
    'verbose':1
    }

asset_id = 'asset_id'
date_idx = 'date'

# data contains excess returns as well as 41 equity factors, industry dummies (10 industries), and 9. Note that the bond and equity
# characteristics have been split according to above and below median. Hence, here we have in total 2*50+10=110 bond-level features

# returns of EJN portfolios; returns are contemporaneous
returns = pd.read_pickle(f'{datapath}/final_dataset/dat_nozawa_int_eret_long.pkl')
returns.sort_index(level = ['date', 'asset_id'], inplace = True)
returns_dates = returns.reset_index()[['date']].drop_duplicates().reset_index()

# funda_data for bond portfolios (lagged one month to returns)
'''
funda_data restricts dataset to start in 1991-01-01
'''
funda_data = pd.read_csv(f"{datapath}/final_dataset/dat_bondpfs_fundamentals.csv").drop(columns = ['Unnamed: 0'])
funda_data['date'] = pd.to_datetime(funda_data['date'])
funda_data['asset_id'] = funda_data['asset_id'].astype('str')
funda_dim = funda_data.shape[1]-2 # -2 for date and asset_id

# aggregate_bond_features; this file has features all lagged by one month
bond_agg_features = pd.read_pickle(f'{datapath}/final_dataset/dat_nozawa_int_eret_long_features.pkl')
# throw out eret, PCs
bond_agg_features.drop(columns = [f'PC{i}' for i in range(1,31)]+['ret_e', 'asset_id'], inplace = True)
bond_agg_features = bond_agg_features.iloc[:,:-40].drop_duplicates()
cols = bond_agg_features.columns[1:]
bond_agg_features.rename(columns = {col:'bond_agg_'+col for col in cols},inplace=True)
# we have 9 forward spreads + 50 meds here = 59 features

macro_raw = pd.read_pickle(f'{datapath}/final_dataset/dat_macrodata.pkl')
macro_raw.drop(columns = ['year','month'], inplace = True)
# add forward spreads. These are not lagged.
# add medchars. There are 41 of them. These are lagged by a month, so need to shift them up to have on same level as macro_raw so far.
medchars=pd.read_pickle(f'{datapath}/final_dataset/dat_equityreturn_medchar.pkl')
medchars=medchars.set_index('date').shift(-1).reset_index().dropna() # one needs to be careful how to apply shift here!
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

# define seq_len and create macro-features
seq_len = 12
# see documentation of df_lags_builder in NNTrainingUtilities
macro_raw = df_lags_builder(macro_raw.set_index('date'), seq_len-1).reset_index()
final_dataset = returns.reset_index().merge(funda_data, how = 'inner', on = ['date', 'asset_id']).merge(macro_raw, how = 'inner', on = 'date').sort_values(by = ['date','asset_id'])
raise_naninf(final_dataset)
# equalize dates of macro and of chars/returns
# final dates
data_dates = final_dataset['date'].drop_duplicates().sort_values().astype('datetime64[ns]').reset_index(drop = True)
print(f'There are {len(data_dates)} many months of data ') # 449 months of data
print(f'Date range of all dataset is {final_dataset.date.min()} to {final_dataset.date.max()}.')

# preprocessing for MP networks
data, _ , _ = enum_dates_assets(final_dataset, date_idx = date_idx, asset_idx = asset_id, 
                                            save_path=savepath, job_number = job_number)
raise_naninf(data)

'''extremely important to have dates and t_index aligned here'''
T = data.date_id.nunique()

returns = data.iloc[:,:4].sort_values('t_index')
X = data.drop(columns=['ret_e','asset_id']).sort_values('t_index')#.iloc[:,:2810]  # added due to current issues with the number of columns being produced

trainval_test_split = int(T*0.7) #305
X.loc[X['date_id']<=trainval_test_split].to_pickle(f'{savepath}/X_train_val_{job_number}.pkl')
X.loc[X['date_id']>trainval_test_split].to_pickle(f'{savepath}/X_test_{job_number}.pkl')
returns.to_pickle(f'{savepath}/asset_returns_{job_number}.pkl')


dropout = 0.01 ## default is None
batchnormalization = True

macro_input_length = macro_dim*seq_len
macro_hidden_dim = 6
propagate_state_sdf = propagate_state_moments = False
test_assets_dim = 6

num_features = X.shape[1]-2 # -2 for t_index, date_id
config_dict['nr_features'] = num_features

ffnmerge_archi_sdf={}

ffnmerge_archi_sdf[0] = [[macro_hidden_dim, funda_dim],
                      [[1],[1]],
                      [macro_hidden_dim, funda_dim],
                      [64,32,16,8]] 

ffnmerge_archi_sdf[1] = [[macro_hidden_dim, funda_dim],
                      [[1],[32,16]],
                      [macro_hidden_dim, 4],
                      [64,64,32,32,16,8]] 


ffnmerge_params_sdf = {}

for i in ffnmerge_archi_sdf.keys():
    ffnmerge_params_sdf[i] = {'num_features_pre_merge':ffnmerge_archi_sdf[i][0], 
                 'hidden_units_pre_merge':ffnmerge_archi_sdf[i][1],
                 'pre_merge_output_size': ffnmerge_archi_sdf[i][2],
                 'hidden_units_at_merge':ffnmerge_archi_sdf[i][3],
                 'final_output_size':1, # we want a portfolio weight
                 'activation':'SiLU', 
                 'batchnormalization':batchnormalization, 
                 'dropout': dropout}

ffnmerge_archi_moments = [[macro_hidden_dim, funda_dim],
                      [[1],[1]],
                      [macro_hidden_dim, funda_dim],
                      [32,16,8]] 
ffnmerge_params_moments = {'num_features_pre_merge':ffnmerge_archi_moments[0], 
             'hidden_units_pre_merge':ffnmerge_archi_moments[1],
             'pre_merge_output_size': ffnmerge_archi_moments[2],
             'hidden_units_at_merge':ffnmerge_archi_moments[3],
             'final_output_size':test_assets_dim, # we want a vector of test assets
             'activation':'SiLU', 
             'batchnormalization':batchnormalization, 
             'dropout': dropout}

rnn_params = {'num_features':macro_dim,
                'seq_len':seq_len,
                'hidden_dim':macro_hidden_dim,
                'num_layers':1,
                'dropout':None}


model_args_sdf = {}
for i in ffnmerge_archi_sdf.keys():
    model_args_sdf[i] = {'T':T,'ffnmerge_params':ffnmerge_params_sdf[i], 
                       'rnn_params':rnn_params, 
                       'nr_indiv_features':funda_dim, 
                       'macro_input_length':macro_input_length, 
                       'propagate_state':propagate_state_sdf}

model_args_moments = {'T':T,'ffnmerge_params':ffnmerge_params_moments, 
                      'rnn_params':rnn_params, 
                      'nr_indiv_features':funda_dim, 
                      'macro_input_length':macro_input_length, 
                      'propagate_state':propagate_state_moments}

architecture_params = {f'Model {i+1}': ['GAN', [model_args_sdf[i], model_args_moments]] for i in ffnmerge_archi_sdf.keys()}

archidefs = open(f'{savepath}/archidefs_{job_number}.pkl', 'wb')
pickle.dump(architecture_params, archidefs)
archidefs.close()

with open(f"{savepath}/config_file_{job_number}.json", "w") as outfile:
    json.dump(config_dict, outfile)
outfile.close()


