#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:43:39 2022

@author: jed169

Script to ensemble all models, including Beta, SDF with both Sharpe max approach and Mispricing Loss Approach, and 
GAN
"""
import os

from settings import savepath

import pandas as pd
import numpy as np
from datetime import date, datetime
today = date.today().strftime("%b-%d-%Y")
now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

import argparse
from performancemeasures import analyze_residuals, analyze_returns

parser = argparse.ArgumentParser(description  = 'Arguments for ensembling models trained with the mispricing loss or linearized Sharpe, or BetaNet')

parser.add_argument('-rs', '--random_seeds', nargs="+", type = int, default = 19, help = 'random seeds')
parser.add_argument('-proj','--project_name', type = str, default = 'mp_sim', help = 'project name for SDF estimation')
parser.add_argument('-jobnr','--jobnumber', type = int, default = 0, help = 'GAN job number to fetch return data')
parser.add_argument('-modelnr', '--modelnumber', default = 1, type = int, help = 'modelnumber for training and val')
parser.add_argument('-b', '--batch_size', default = 'full', metavar = 'bs', help = 'total batch_size of all GPUs on the current node. Note: if want to give full batch then need to set this to a number above the data-length.')
parser.add_argument('-lr', '--learning_rate', default = 0.01, type = float, metavar = 'LR', help = 'initial learning rate')
parser.add_argument('-l2reg', '--l2regparam', default = 'none', metavar = 'L2Reg', help = 'weight for L2 regularization')
parser.add_argument('-l1reg', '--l1regparam', default = 'none', metavar = 'L1Reg', help = 'weight for L1 regularization')

parser.add_argument('-lr_sdf', '--learning_rate_sdf', default = 0.01, type = float, metavar = 'LR_sdf', help = 'learning rate for sdf network')
parser.add_argument('-lr_mom', '--learning_rate_mom', default = 0.01, type = float, metavar = 'LR_mom', help = 'learning rate for moments network')
parser.add_argument('-l1reg_sdf', '--l1regparam_sdf', default = 'none', metavar = 'L1Reg_sdf', help = 'weight for L1 regularization for sdf network')
parser.add_argument('-l1reg_mom', '--l1regparam_mom', default = 'none', metavar = 'L1Reg_mom', help = 'weight for L1 regularization for moments network')

# ensembling method and valuation metrics
parser.add_argument('-ensmethod', '--ensemble_method', type=str, default='mean', help='ensembling method')
parser.add_argument('-betanet_jobnr', '--betanet_jobnumber', type=int, default=-1, help='job number to calculate residuals from beta regression on SDF, and analyze them')
parser.add_argument('-betanet_modelnr', '--betanet_modelnumber', type=int, default=0, help='model number to calculate residuals from beta regression on SDF, and analyze them')
parser.add_argument('-val_size', '--val_size', type=float, default=0.4, help='validation size from train_val dataset')
parser.add_argument('-valend', '--valend', type=int, default=450, help='date_id of last validation date to analyze residuals and returns')

def produce_pathtail(p_args):
    if p_args.project_name[:3]=='GAN':
        p_args.path_tail = f"jnr_{p_args.jobnumber}_bs_{p_args.batch_size}_lrs_{(p_args.learning_rate_sdf,p_args.learning_rate_mom)}_l2reg_{p_args.l2regparam}_l1regs_{(p_args.l1regparam_sdf,p_args.l1regparam_mom)}"
    else:
        p_args.path_tail = f"jnr_{p_args.jobnumber}_bs_{p_args.batch_size}_lr_{p_args.learning_rate}_l2reg_{p_args.l2regparam}_l1reg_{p_args.l1regparam}"

def main():

    args = parser.parse_args()
    
    try:
        fun=getattr(np, args.ensemble_method)
    except AttributeError:
        print('Need to give an aggregation function recognized by Numpy!')

    if args.betanet_jobnumber==-1:
        '''
            Ensembling a SDF estimation project
        '''
        produce_pathtail(args)
        
        os.chdir(f'{savepath}/{args.project_name}/')
        sdf_returns = pd.read_pickle(f"{args.project_name}_Model_{args.modelnumber}_final_returns_{args.path_tail}_rs_{args.random_seeds[0]}.pkl").set_index("date_id")
        sdf_returns.columns = sdf_returns.columns + f"_{args.random_seeds[0]}"
        sdf_weights = pd.read_pickle(f"{args.project_name}_Model_{args.modelnumber}_final_weights_{args.path_tail}_rs_{args.random_seeds[0]}.pkl").set_index(["t_index","date_id"])
        sdf_weights.columns = sdf_weights.columns + f"_{args.random_seeds[0]}"
        
        if not args.project_name.startswith('SR'):
        
            betas = pd.read_pickle(f"{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}_rs_{args.random_seeds[0]}.pkl").set_index(["t_index","date_id"])
            betas.columns = [f'betas_{args.random_seeds[0]}']
        
        for random_state in args.random_seeds[1:]:
            df = pd.read_pickle(f"{args.project_name}_Model_{args.modelnumber}_final_returns_{args.path_tail}_rs_{random_state}.pkl").set_index("date_id")
            df.columns = df.columns + f"_{random_state}"
            sdf_returns = pd.concat([sdf_returns,df],axis = 1)
            df = pd.read_pickle(f"{args.project_name}_Model_{args.modelnumber}_final_weights_{args.path_tail}_rs_{random_state}.pkl").set_index(["t_index","date_id"])
            df.columns = df.columns + f"_{random_state}"
            sdf_weights = pd.concat([sdf_weights,df],axis = 1)
            if not args.project_name.startswith('SR'):
                df = pd.read_pickle(f"{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}_rs_{random_state}.pkl").set_index(["t_index","date_id"])
                df.columns = df.columns + f"betas_{random_state}"
                betas = pd.concat([betas,df],axis = 1)
        
        sdf_returns[f'return_{args.ensemble_method}'] = sdf_returns.apply(fun, axis=1)
        sdf_weights[f'weights_{args.ensemble_method}'] = sdf_weights.apply(fun, axis=1)
        if not args.project_name.startswith('SR'):
            betas[f'betas_{args.ensemble_method}'] = betas.apply(fun, axis=1)
        
        sdf_returns.reset_index().to_pickle(f"{args.project_name}_Model_{args.modelnumber}_final_returns_{args.path_tail}_{args.ensemble_method}_ensembled.pkl")
        sdf_weights.reset_index().to_pickle(f"{args.project_name}_Model_{args.modelnumber}_final_weights_{args.path_tail}_{args.ensemble_method}_ensembled.pkl")
        sdf_returns.reset_index().to_excel(f"{args.project_name}_Model_{args.modelnumber}_final_returns_{args.path_tail}_{args.ensemble_method}_ensembled.xlsx")
        sdf_weights.reset_index().to_excel(f"{args.project_name}_Model_{args.modelnumber}_final_weights_{args.path_tail}_{args.ensemble_method}_ensembled.xlsx")
        if not args.project_name.startswith('SR'):
            betas.reset_index().to_pickle(f"{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}_{args.ensemble_method}_ensembled.pkl")
            betas.reset_index().to_excel(f"{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}_{args.ensemble_method}_ensembled.xlsx")
        
        sdf_returns.reset_index(inplace = True)
        rets = sdf_returns.loc[sdf_returns['date_id']>args.valend, f'return_{args.ensemble_method}']
        results = analyze_returns(returns = rets)
        results.to_pickle(f"{args.project_name}_Model_{args.modelnumber}_returns_performance_{args.path_tail}_{args.ensemble_method}_ensembled.pkl")
        results.to_excel(f"{args.project_name}_Model_{args.modelnumber}_returns_performance_{args.path_tail}_{args.ensemble_method}_ensembled.xlsx")
        
    else:
        '''
            Ensembling a BetaNet project
        '''
        os.chdir(f'/ix/jduraj/Bonds/Data/current_train_val_test/BetaNet_{args.project_name}/')
        betas = {}
        args.betanet_project_name = 'BetaNet_'+args.project_name
        args.betanet_pathtail = f"jnr_{args.betanet_jobnumber}_bs_{args.batch_size}_lr_{args.learning_rate}_l2reg_{args.l2regparam}_l1reg_{args.l1regparam}"
        for i, random_state in enumerate(args.random_seeds):
            
            betas[f"betas_rs_{random_state}"] = np.load(f"{args.betanet_project_name}_Model_{args.modelnumber}_AllPredictions_{args.betanet_pathtail}_rs_{random_state}.npy")
        
        betas = pd.DataFrame(np.concatenate(list(betas.values()), axis = 1), columns = list(betas.keys()))#, columns = ['betas_rs_'+ str(rs) for rs in args.random_seeds]
        betas[f'betas_{args.ensemble_method}'] = betas.apply(fun, axis = 1)
        reg = pd.read_pickle("almost_reg.pkl")
        
        reg['beta'] = betas[f'betas_{args.ensemble_method}']
        
        reg.to_pickle(f"raw_betareg_data_{args.ensemble_method}.pkl")
        args.trainend = args.valend - int(args.valend*args.val_size)
        reg_train = reg[reg['date_id']<=args.trainend]
        reg_val = reg.loc[(reg['date_id']<=args.valend) & (reg['date_id']>args.trainend), :]
        reg_test = reg[reg['date_id']>args.valend]
        ev, xsr2, residuals = {},{},{}
        ev['train'], xsr2['train'], residuals['train'] = analyze_residuals(reg_train)
        ev['val'], xsr2['val'], residuals['val'] = analyze_residuals(reg_val)
        ev['test'], xsr2['test'], residuals['test'] = analyze_residuals(reg_test)
        dat = np.concatenate([np.array(list(ev.values())).reshape(-1,1), np.array(list(xsr2.values())).reshape(-1,1)], axis = 1)
        dat = pd.DataFrame(dat, index = list(ev.keys()), columns = ['ev','xsr2'])
        dat.to_pickle(f"ev_xsr2_{args.ensemble_method}.pkl")
        dat.to_excel(f"ev_xsr2_{args.ensemble_method}.xlsx")
        
if __name__ == '__main__':
    main()
    
    

