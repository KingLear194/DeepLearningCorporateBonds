#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:30:27 2023

@author: jed169

BetaNet Bonds preprocessing

"""

import json, pickle, glob
import pandas as pd

origin_project = 'SR_bonds' 
betanet_jobnumber = 21
project_name = 'BetaNet_'
savepath = f"/ix/jduraj/Bonds/Data/current_train_val_test/{project_name}"

trainval_test_split = 305 if 'NozawaPort' in origin_project else 392
y = pd.read_pickle(glob.glob(f"{savepath}/*_betanet_labels_*.pkl")[0])[['t_index','date_id','betas_mean']].rename(columns = {'betas_mean':'beta_label'}).sort_values(by=['t_index','date_id']).drop(columns = ['t_index'])
print(y.shape)

ytrainval = y.loc[y['date_id']<=trainval_test_split].set_index(['date_id'])
ytrainval.to_pickle(f'{savepath}/y_train_val_{betanet_jobnumber}.pkl')
print(ytrainval.shape)
ytest = y.loc[y['date_id']>trainval_test_split].set_index(['date_id'])
print(ytest.shape)
ytest.to_pickle(f'{savepath}/y_test_{betanet_jobnumber}.pkl')

pd.read_pickle(glob.glob(f"{savepath}/asset_returns*")[0]).to_pickle(f"{savepath}/almost_reg.pkl")

X_train_val = pd.read_pickle(f"{savepath}/X_train_val_{betanet_jobnumber}.pkl")
print(X_train_val.shape)
X_test = pd.read_pickle(f"{savepath}/X_test_{betanet_jobnumber}.pkl")
print(X_test.shape)

X = pd.concat([X_train_val, X_test], axis=0)
X_train_val = X.loc[X['date_id']<=trainval_test_split]
X_test = X.loc[X['date_id']>trainval_test_split]

print(X_train_val.shape)
print(X_test.shape)

num_features = X_test.shape[1]
X_train_val.sort_values(by=['t_index','date_id']).drop(columns = ['t_index']).set_index('date_id').to_pickle(f"{savepath}/X_train_val_{betanet_jobnumber}.pkl")
X_test.sort_values(by=['t_index','date_id']).drop(columns = ['t_index']).set_index('date_id').to_pickle(f"{savepath}/X_test_{betanet_jobnumber}.pkl")
num_features-=2

config_dict = {'device':'cuda',
    'validate': True,
    'optimizer_name': 'Adam', 
    'early_stopping': [8,0], 
    'momentum': 0.9,
    'nr_features': num_features, 
    'train_criterion_name': 'MSE',
    'val_criterion_name': 'MSE',
    'xscaling': True,
    'yscaling': True,
    'view_losses':True,
    'save_model': True,
    'verbose':1}

dataparams = {'fixed': 0.4, 'date_idx':'date_id'} # 0.4 of train_val is val

#architecture defs
dropout = 0.01 ## default is None
batchnormalization = True

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

architecture_params = {'Model 1':[dataparams, 'FFNStandard',ffn_params1],
                       'Model 2':[dataparams, 'FFNStandard',ffn_params2], 
                       'Model 3':[dataparams, 'FFNStandard',ffn_params3], 
                       'Model 4':[dataparams, 'FFNStandard',enet_params]}

archidefs = open(f'{savepath}/archidefs_{betanet_jobnumber}.pkl', 'wb')
pickle.dump(architecture_params, archidefs)
archidefs.close()

with open(f"{savepath}/config_file_{betanet_jobnumber}.json", "w") as outfile:
    json.dump(config_dict, outfile)
outfile.close()


