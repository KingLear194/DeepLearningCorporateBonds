#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 22:28:37 2022

@author: jed169
"""

import sys, os
import json, pickle
import pandas as pd
sys.path.insert(0, '/ix/jduraj/Bonds/Code/NN')

from aux_utilities import raise_naninf, create_dir_ifnexists, df_lags_builder, enum_dates_assets

project_name = 'GAN_MP_bonds'
job_number = 4

dirpath='/ix/jduraj/'
os.chdir(dirpath)
datapath=f'{dirpath}Bonds/Data'
### Savepath
savepath=f'{dirpath}Bonds/Data/current_train_val_test/{project_name}'
create_dir_ifnexists(savepath)

# the dictionary to be saved as a config json file for gan
# note: the values from here can be overrun through command line
config_dict = {
    "multiprocessing": False,
    'train_val_split': 0.6,
    'validate': True,
    'sub_epochs': (4,4,4),
    'optimizer_name': 'Adam', # other option is RMSprop
    'early_stopping': [8,0], 
    'sdf_momentum': 0.9,
    'moments_momentum': 0.9,
    'leverage':5,
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

asset_id = 'cusip'
date_idx = 'date'

data = pd.read_pickle(f'{datapath}/final_dataset/dat_indreturn_macro_fundamentals_nomissing.pkl')
data.drop(columns = ['year', 'month', 'issuer_cusip'], inplace = True)

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

# define seq_len and create macro-features
seq_len = 12
# see documentation of df_lags_builder in NNTrainingUtilities
macro_features = df_lags_builder(macro_raw.set_index('date'), seq_len-1).reset_index()
print('date in macro_features is: ')
print(macro_features.date.max())
# equalize dates of macro and of chars/returns
chars_dates = data[['date']].drop_duplicates()
macro_raw = chars_dates.merge(macro_raw, how = 'inner', on  = 'date')
data = macro_raw[['date']].merge(data, how = 'inner', on = 'date')
print('date in data is: ')
print(data.date.max())
# final dates
data_dates = data['date'].drop_duplicates().sort_values().astype('datetime64[ns]').reset_index(drop = True)
print(f'There are {len(data_dates)} many months of data ') # 449 months of data
final_dataset = data.merge(macro_features, how = 'inner', on = 'date')
print(f'Date range of all dataset is {final_dataset.date.min()} to {final_dataset.date.max()}.')
del data
# preprocessing for MP networks
final_dataset, dates_per_asset, assets_per_date = enum_dates_assets(final_dataset, date_idx = date_idx, asset_idx = asset_id, 
                                            save_path=savepath, job_number = job_number)

print("passed enum_dates_assets")

'''extremely important to have dates and t_index aligned here'''
T = final_dataset.date_id.nunique()
returns = final_dataset.iloc[:,:4].sort_values('t_index')
X = final_dataset.drop(columns=['ret_e','asset_id']).sort_values('t_index')
del final_dataset

raise_naninf(X)
print("just before train_val_test_split")


trainval_test_split = int(T*0.7)
X.loc[X['date_id']<=trainval_test_split].to_pickle(f'{savepath}/X_train_val_{job_number}.pkl') 
X.loc[X['date_id']>trainval_test_split].to_pickle(f'{savepath}/X_test_{job_number}.pkl')
returns.to_pickle(f'{savepath}/asset_returns_{job_number}.pkl')

dropout = 0.01 
batchnormalization = True

macro_input_length = macro_dim*seq_len
macro_hidden_dim = 6
propagate_state_sdf = True
propagate_state_moments = True
test_assets_dim = 6

num_features = X.shape[1]-2 # -2 for t_index, date_id
config_dict['nr_features'] = num_features

ffnmerge_archi_sdf1 = [[macro_hidden_dim, funda_dim],
                      [[1],[1]],
                      [macro_hidden_dim, funda_dim],
                      [64,64,32,32,16,8]] 
ffnmerge_params_sdf1 = {'num_features_pre_merge':ffnmerge_archi_sdf1[0], 
             'hidden_units_pre_merge':ffnmerge_archi_sdf1[1],
             'pre_merge_output_size': ffnmerge_archi_sdf1[2],
             'hidden_units_at_merge':ffnmerge_archi_sdf1[3],
             'final_output_size':1, # we want a portfolio weight
             'activation':'SiLU', 
             'batchnormalization':batchnormalization, 
             'dropout': dropout}

ffnmerge_archi_sdf2 = [[macro_hidden_dim, funda_dim],
                      [[1],[32,16]],
                      [macro_hidden_dim, 4],
                      [64,64,32,32,16,8]] 
ffnmerge_params_sdf2 = {'num_features_pre_merge':ffnmerge_archi_sdf2[0], 
             'hidden_units_pre_merge':ffnmerge_archi_sdf2[1],
             'pre_merge_output_size': ffnmerge_archi_sdf2[2],
             'hidden_units_at_merge':ffnmerge_archi_sdf2[3],
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

rnn_params1 = {'num_features':macro_dim,
                'seq_len':seq_len,
                'hidden_dim':macro_hidden_dim,
                'num_layers':1,
                'dropout':None}
rnn_params2 = {'num_features':macro_dim,
                'seq_len':seq_len,
                'hidden_dim':macro_hidden_dim,
                'num_layers':2,
                'dropout':None}

model_args_sdf1 = {'T':T,'ffnmerge_params':ffnmerge_params_sdf1, 'rnn_params':rnn_params1, 'nr_indiv_features':funda_dim, 'macro_input_length':macro_input_length, 'propagate_state':propagate_state_sdf}
model_args_moments1 = {'T':T,'ffnmerge_params':ffnmerge_params_moments, 'rnn_params':rnn_params1, 'nr_indiv_features':funda_dim, 'macro_input_length':macro_input_length, 'propagate_state':propagate_state_moments}
model_args_sdf2 = {'T':T,'ffnmerge_params':ffnmerge_params_sdf2, 'rnn_params':rnn_params2, 'nr_indiv_features':funda_dim, 'macro_input_length':macro_input_length, 'propagate_state':propagate_state_sdf}
model_args_moments2 = {'T':T,'ffnmerge_params':ffnmerge_params_moments, 'rnn_params':rnn_params2, 'nr_indiv_features':funda_dim, 'macro_input_length':macro_input_length, 'propagate_state':propagate_state_moments}

model_args1 = ['GAN', [model_args_sdf1, model_args_moments1]] 
model_args2 = ['GAN', [model_args_sdf2, model_args_moments2]] 
architecture_params = {'Model 1': model_args1, 'Model 2':model_args2}

archidefs = open(f'{savepath}/archidefs_{job_number}.pkl', 'wb')
pickle.dump(architecture_params, archidefs)
archidefs.close()

# in the preprocessing file, besides definition of config_dict, do to dump it
with open(f"{savepath}/config_file_{job_number}.json", "w") as outfile:
    json.dump(config_dict, outfile)
outfile.close()
print("Done!")