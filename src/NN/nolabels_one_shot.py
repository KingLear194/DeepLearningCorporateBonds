#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:30:10 2021

@author: jed169

A script for a one-shot Training for the MP and SR models. 
Called 'nolabels' because there is no prediction here, just minimizing a statistical loss given data.

"""

import sys

from settings import datapath

import pandas as pd
import numpy as np
import json, pickle
import time
import torch

import TrainUtils 
from labels_one_shot import init_optimizer, init_model

import argparse

start_time = time.time()

'''
parse arguments from terminal
'''
parser = argparse.ArgumentParser(description  = 'Arguments for pytorch one-shot Asset-Pricing (no-labels) script')

parser.add_argument('-proj','--project_name', type = str, default = 'SR_sim', help = 'Project name')
parser.add_argument('-jobnr','--jobnumber', type = int, default = 1, help = 'job number when submitting multiple jobs to the cluster')
parser.add_argument('-printc' , '--printc', type =bool, default = False, help = 'print on console, otherwise save on txt file')
parser.add_argument('-time_start' , '--time_start', type=str, default = "no_timelog", help = 'get start time')

'''args for training and validation'''

# hyperparameters we optimize over in the validation phase
parser.add_argument('-rs', '--random_seed', type = int, default = 19, help = 'random seed for all packages')
parser.add_argument('-modelnr', '--modelnumber', default = 1, type = int, help = 'modelnumber for training and val')
parser.add_argument('-b', '--batch_size', default = 32, metavar = 'bs', help = 'batch_size')
parser.add_argument('-lr', '--learning_rate', default = 0.01, type = float, metavar = 'LR', help = 'initial learning rate')
parser.add_argument('-l2reg', '--l2regparam', default = 'none', metavar = 'L2Reg', help = 'weight for L2 regularization')
parser.add_argument('-l1reg', '--l1regparam', default = 'none', metavar = 'L1Reg', help = 'weight for L1 regularization')

# hyperparameters and other modeling choices we do not optimize over in the validation phase
parser.add_argument('-epochs', '--num_epochs', default = 10, type = int, metavar = 'nr Epochs', help = 'Number of epochs to train')

# other
parser.add_argument('-config', '--config_file_path', default = 'none', type = str, help = 'config file name to load')


def init_train_val_instance(train_criterion_name, args):
    if train_criterion_name == 'mispricing_loss':
        return TrainUtils.TrainValMPLoss(**args) 
    elif train_criterion_name == 'minus_mean_std_utility':
        return TrainUtils.TrainValSR(**args)

def main():
    
    args = parser.parse_args()
    
    # load the config parameters
    if args.config_file_path != 'none':
        with open(args.config_file_path, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
            
    args.savepath=f'{datapath}/current_train_val_test/{args.project_name}'
    ### Plotpath
    args.pltpath = args.savepath
    
    args.path_tail = f"jnr_{args.jobnumber}_bs_{args.batch_size}_lr_{args.learning_rate}_l2reg_{args.l2regparam}_l1reg_{args.l1regparam}_rs_{args.random_seed}"
    if not args.printc: 
        sys.stdout = open(f"{args.savepath}/{args.project_name}_log_{args.time_start}.out", 'a')
    
    # retrieve architecture (a dict with key 'Model <modelnumber>')
    archidefs = open(f"{args.savepath}/archidefs_{args.jobnumber}.pkl",'rb')
    architectures = pickle.load(archidefs)
    archidefs.close()
    args.model_args = architectures[f"Model {args.modelnumber}"] # this will be a list with name and list of two dicts that will contain the archi_params for the two networks
    del architectures    
    # retrieve full file of returns and also test_assets if they are there
    args.returns = pd.read_pickle(f"{args.savepath}/asset_returns_{args.jobnumber}.pkl")
    if (args.train_criterion_name == 'mispricing_loss') and args.exogenous_test_assets:
        # test assets need to have columns date_id, then rest of row are the test assets
        args.test_assets = pd.read_pickle(f"{args.savepath}/exogenous_test_assets_{args.jobnumber}.pkl")
    else:
        args.test_assets = None
   
    # turn l2regparam into float
    if args.l2regparam != 'none': 
        args.l2regparam = float(args.l2regparam)
    else:
        args.l2regparam = 0.0
    # turn l1regparam into float
    if args.l1regparam != 'none': 
        args.l1regparam = float(args.l1regparam) 
    else:
        args.l1regparam = 0.0
        
    args.best_mispricing_loss = np.Inf
    args.best_sharpe_ratio = -np.Inf
    
    # we train on one model only
    args.world_size = 1
    
    main_worker(0, args)
    
    if not args.printc: 
        sys.stdout.close()
    

def main_worker(gpu, args):
    
    start = time.time()
    
    torch.cuda.device(gpu)
    TrainUtils.random_seeding(args.random_seed, use_cuda=True)
    
    
    network = init_model(*args.model_args, world_size = args.world_size)
    torch.cuda.set_device(gpu)
    network.cuda(gpu)
    
    # initialize optimizer
    optimizer = init_optimizer(network.parameters(), args)
    
    print(f'\nLoaded criterion and optimizer.\nlearning Rate is {args.learning_rate}, weight decay is {args.l2regparam} and L1-loss parameter is {args.l1regparam}.\n')
    print("==================================================================================================================================")
    print(f'\n\nLoaded for global training.\nTraining Model {args.modelnumber} with params {args.path_tail}.\n')

    # Data loading 
    init_data = TrainUtils.InitDataloader(args, labels = False)
    
    # initialize train_val stuff
    trainval_args_dict = {'idxdates':init_data.idxdates, 
                          'network':network, 
                          'optimizer':optimizer, 
                          'world_size':args.world_size, 
                          'returns':args.returns, 
                          'leverage_constraint_type':args.leverage_constraint_type,
                          'leverage':args.leverage,
                          'nr_features':args.nr_features, 
                          'test_loader': init_data.test_loader,
                          'train_loader':init_data.train_loader,
                          'val_loader':init_data.val_loader,
                          'val_criterion':'sharpe_ratio', 
                          'train_criterion':args.train_criterion_name,
                          'early_stopping':args.early_stopping,
                          'l1regparam':args.l1regparam,
                          'verbose' : args.verbose}
    
    if args.train_criterion_name == 'mispricing_loss' and args.test_assets!=None: 
        trainval_args_dict['test_assets'] =args.test_assets
    elif args.train_criterion_name == 'minus_mean_std_utility':
        trainval_args_dict['risk_aversion']=args.risk_aversion
        
    train_val = init_train_val_instance(args.train_criterion_name, trainval_args_dict)    
    
    print("Loaded data and the estimation utilities.")
        
    for epoch in range(1,(args.num_epochs+1)):
        
        # train for one epoch
        train_val.train_one_ep(epoch)  
        # check for early stopping
        if args.validate:
            train_val.validate_one_ep(epoch)
            if args.early_stopping[0] < np.Inf:
                if train_val.stop_early:
                    break
    
    if args.validate:
        
        # test set performance 
        train_val.testset_performance()
        #prepare results for output/saving
        train_val.gather_performance()
             
        if args.save_model: 
            model_pth = f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_Network_{args.path_tail}.pt"
            torch.save(train_val.network, model_pth)
        
        # save final features
        dataframes = {}
        if args.get_betanet_labels:
            betanet_labels = TrainUtils.betanet_labels(jobnumber = args.jobnumber, pf_returns_df = train_val.final_returns, savepath = args.savepath)
            betanet_labels.to_pickle(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}.pkl")
            betanet_labels.to_excel(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}.xlsx")
        
        dataframes['final_returns'] = train_val.final_returns
        dataframes['final_weights'] = pd.DataFrame(train_val.final_weights.detach().cpu().numpy(), columns = ['weights'])
        dataframes['final_weights'] = args.returns[['t_index','date_id']].merge(dataframes['final_weights'].reset_index().rename(columns = {'index':'t_index'}),
                                                                    on = 't_index', how = 'inner')
        dataframes['final_features'] = args.returns.merge(dataframes['final_weights'], how='inner', 
                                            on = ['t_index','date_id']).merge(dataframes['final_returns'], how = 'inner', on = 'date_id')
                                                                    
        if args.get_betanet_labels:
            dataframes['final_features'] = dataframes['final_features'].merge(betanet_labels, how = 'inner', on = ['t_index','date_id'])
        # save performance dataframes
        perf_dict = train_val.performance_dict.copy() # probably unnecessary
        for name, df in perf_dict.items():
            cols = list(set(df.columns)-set(['phase','epoch']))
            cols.sort()
            df = df[['phase','epoch']+cols]
            perf_dict[name] = df
            
        dataframes.update(perf_dict)
        
        for key in dataframes:
            pth = f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_{key}_{args.path_tail}"
            dataframes[key].to_pickle(f"{pth}.pkl")
            try:
                dataframes[key].to_excel(f"{pth}.xlsx")
            except ValueError as error:
                print(error)
        
        # Save pics of performance measures
        if args.view_losses:
            for key, df in train_val.performance_dict.items():
                TrainUtils.plot_performance(df, listgrpbycols = ['phase'], value = key, pltpath=args.savepath,
                                title = f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_plot_{key}_{args.path_tail}")
                
    end = time.time()
    time_elapsed = end-start
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'\nTraining/validation/testing for Model {args.modelnumber} with {args.path_tail}\n in gpu {gpu} complete in {hours} hours, {minutes} minutes and {round(seconds,1)} seconds.\n')
    print("==================================================================================================================================")
    print("==================================================================================================================================")
            
if __name__ == '__main__':
    main()
