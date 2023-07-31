#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:09:17 2023

@author: jed169

Code for alpha calculation

"""

from settings import savepath

import argparse, glob, pandas as pd, numpy as np, pickle
from aux_utilities import raise_naninf
from performancemeasures import split_rets
import statsmodels.api as sm
from scipy.stats import t

parser = argparse.ArgumentParser(description  = 'Arguments for alpha calculation')
parser.add_argument('-proj', '--project_name', type = str, default = 'SR_bonds', help = 'project_name')
parser.add_argument('-bench', '--benchmark', type = str, default = 'NozawaPort', help = 'benchmark for factors')
parser.add_argument('-trainend', '--trainend', type=int, default=235, help='date_id of last train date')
parser.add_argument('-valend', '--valend', type=int, default=392, help='date_id of last validation date')
parser.add_argument('-save', '--save', type=bool, default=True, help='save results')
parser.add_argument('-verbose', '--verbose', type=bool, default=True, help='whether to print alpha info')
parser.add_argument('-res_folder', '--results_folder', type=str, default='Results', help='name of folder where results are saved inside the project folder')

def alpha_calc(reg_data, trainend, valend, date_idx = 'date_id', verbose = True):
    '''
    First column of reg_data needs to be date_idx, second column the pf_return (return_label)
    The rest of the columns are the portfolios we are regressing against (benchmark factors)
    '''
    
    print("reg_data shape is ", reg_data.shape)
    
    reg_train, reg_val, reg_test = split_rets(rets = reg_data, trainend=trainend, return_label=None,
                                valend = valend, date_idx=date_idx)
    
    reg_train = reg_train.drop(columns = [date_idx])
    reg_test = reg_test.drop(columns = [date_idx])
    reg_val = reg_val.drop(columns = [date_idx])
        
    reg_dict = dict(zip(['train','val','test'],[reg_train, reg_val, reg_test]))
    reg_results = {}
    
    for phase, df in reg_dict.items():
        reg_results[phase] = {}
        X = df.iloc[:,1:].to_numpy()
        X = sm.add_constant(X)
        reg = sm.OLS(df.iloc[:,0], X)
        reg = reg.fit()
        reg_results[phase]['R2'] = reg.rsquared
        
        if verbose: print(f'\nalpha for phase {phase}: {reg.params[0]}')
        reg_results[phase]['alpha'] = reg.params[0]
        reg_results[phase]['factor_coeff'] = reg.params[1:]
        reg_results[phase]['t-values'] = reg.tvalues
        reg_results[phase]['p-values'] = t.sf(np.abs(reg.tvalues), X.shape[0]-X.shape[1])*2
        if verbose: print("t-value for alpha: \n", reg_results[phase]['t-values'][0], 
              '\np-value for alpha:\n', reg_results[phase]['p-values'][0],'\n\n')
        
        reg_results[phase]['residuals'] = df.iloc[:,0] - reg.predict()
        if verbose: print(reg.summary())
        
    return reg_results
    
def main():
    args = parser.parse_args()

    datapath = f'{savepath}/{args.project_name}/{args.results_folder}'
    rets = pd.read_pickle(glob.glob(f"{datapath}/*final_returns_*_mean_ensembled.pkl")[0])
    enum_dates = pd.read_excel(glob.glob(f"{datapath}/enum_dates*.xlsx")[0])
    rets = rets.merge(enum_dates, how = 'inner', on = 'date_id')[['date','date_id','return_mean']].rename(columns = {'return_mean':'sdf_return'})

    if args.benchmark=='NozawaPort':
        factors = pd.read_pickle("/ix/jduraj/Bonds/Data/final_dataset/dat_nozawa_int_eret_long.pkl").reset_index().pivot(index = 'date', columns = 'asset_id', values = 'ret_e').dropna()
        factor_cols = list(factors.columns)
    elif args.benchmark=='FF3':
        factors = pd.read_pickle("/ix/jduraj/Bonds/Data/int/factorsforperformanceevaluation.pkl").iloc[:,:4]
        factor_cols = list(factors.columns)[1:]
    elif args.benchmark=='FF5':
        factors = pd.read_pickle("/ix/jduraj/Bonds/Data/int/factorsforperformanceevaluation.pkl").iloc[:,:6]
        factor_cols = list(factors.columns)[1:]        
    elif args.benchmark=='SPY':
        factors = pd.read_pickle("/ix/jduraj/Bonds/Data/int/factorsforperformanceevaluation.pkl").iloc[:,[0,-1]]
        factor_cols = list(factors.columns)[1:]    
        
    rets = rets.merge(factors, how = 'inner', on = 'date')[['date_id','sdf_return',*factor_cols]].copy()
    raise_naninf(rets)
    print("\n")
    alpha_results = alpha_calc(rets, valend = args.valend, trainend = args.trainend, verbose = args.verbose)
    if args.save:
        with open(f"{datapath}/{args.project_name}_alpha_{args.benchmark}_results.pkl", 'wb') as file:
            pickle.dump(alpha_results, file)
        file.close()

if __name__ == '__main__':
    main()