#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:30:10 2021

@author: jed169

A script for a one-shot Training of the GAN model

"""

import sys
from settings import datapath

import pandas as pd
import numpy as np
import json, pickle
import time
import torch
import TrainUtils 
from labels_one_shot import init_optimizer
from aux_utilities import save_strlist_txt
import argparse

start_time = time.time()

#### parse arguments from terminal
'''
Add flag to save predictions, add flag for job number which is consistent with flag you'll add to files produced from the preprocessing file (a version of ddp_train_val_prep.py)'
'''
parser = argparse.ArgumentParser(description  = 'Arguments for pytorch one-shot Asset-Pricing GAN script')

parser.add_argument('-proj','--project_name', type = str, default = 'SDF_GAN', help = 'Project name')
parser.add_argument('-jobnr','--jobnumber', type = int, default = 1, help = 'job number when submitting multiple jobs to the cluster')

# DDP arguments
parser.add_argument('-printc' , '--printc', type =bool, default = False, help = 'print on console, otherwise save on .out file')
parser.add_argument('-time_start' , '--time_start', type=str, default = "no_timelog", help = 'get start time')

'''args for training and validation'''

# hyperparameters we optimize over in the validation phase
parser.add_argument('-rs', '--random_seed', type = int, default = 19, help = 'random seed for all packages')
parser.add_argument('-modelnr', '--modelnumber', default = 1, type = int, help = 'modelnumber for training and val')
parser.add_argument('-b', '--batch_size', default = 32, metavar = 'bs', help = 'total batch_size of all GPUs on the current node. Note: if want to give full batch then need to set this to a number above the data-length.')
parser.add_argument('-lr_sdf', '--learning_rate_sdf', default = 0.01, type = float, metavar = 'LR_sdf', help = 'learning rate for sdf network')
parser.add_argument('-lr_mom', '--learning_rate_mom', default = 0.01, type = float, metavar = 'LR_mom', help = 'learning rate for moments network')
parser.add_argument('-l2reg', '--l2regparam', default = 'none', metavar = 'L2Reg', help = 'weight for L2 regularization')
parser.add_argument('-l1reg_sdf', '--l1regparam_sdf', default = 'none', metavar = 'L1Reg_sdf', help = 'weight for L1 regularization for sdf network')
parser.add_argument('-l1reg_mom', '--l1regparam_mom', default = 'none', metavar = 'L1Reg_mom', help = 'weight for L1 regularization for moments network')

# hyperparameters and other modeling choices we do not optimize over in the validation phase
parser.add_argument('-epochs', '--num_epochs', default = 10, type = int, metavar = 'Nr of global Epochs', help = 'Number of global epochs to train')
parser.add_argument('-subepochs', '--sub_epochs', default = (2,2,2), type = tuple, metavar = 'subepochs', help = 'Number of subepochs to train unconditional and conditional loss')
parser.add_argument('-optimizer', '--optimizer_name', default = 'Adam', type = str, metavar = 'optimizer', help = 'Optimizer name')
parser.add_argument('-momentum', '--momentum', default = 0.9, type = float, metavar = 'momentum', help = 'momentum')
parser.add_argument('-train_criterion', '--train_criterion_name', default = 'mispricing_loss', type = str, metavar = 'criterion', help = 'criterion name for training')
parser.add_argument('-val_criterion', '--val_criterion_name', default = 'sharpe_ratio', type = str, metavar = 'val criterion', help = 'val criterion name; may be different from train criterion')
parser.add_argument('-xscaling', '--xscaling', default = True, type = bool, nargs = "?", const=True, metavar = 'xscaling', help = 'bool for xscaling')
parser.add_argument('-yscaling', '--yscaling', default = True, type = bool, nargs = "?", const=True, metavar = 'yscaling', help = 'bool for yscaling')

# other
parser.add_argument('-config', '--config_file_path', default = 'none', type = str, help = 'config file name to load')
parser.add_argument('-unc_loss_weight', '--unconditional_loss_weight', default = 1, type = float, metavar = 'unc_loss_weight', help = 'unconditional loss weight (vs. conditional loss weight)')
parser.add_argument('-verbose', '--verbose', default = 1, type = float, metavar = 'verbose', help = 'verbose level')
parser.add_argument('-outpreds', '--output_predictions', default = True, type = bool, nargs = "?", const=True, metavar = 'predictions', help = 'output predictions in separate pickle file')
parser.add_argument('-savemodel', '--save_model', default = True, type = bool, nargs = "?", const=True,metavar = 'save model', help = 'bool for save model')
parser.add_argument('-viewlosses', '--view_losses', default = True, type = bool, nargs = "?", const=True, metavar = 'view losses', help = 'bool for view losses')

# initialize global variables

def main():
    
    args = parser.parse_args()
    # load the config parameters
    if args.config_file_path != 'none':
        with open(args.config_file_path, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
            
    args.savepath=f'{datapath}/current_train_val_test/{args.project_name}'
    args.pltpath = args.savepath
    
    if not args.printc: 
        sys.stdout = open(f"{args.savepath}/{args.project_name}_log_{args.time_start}.out", 'a')
    
    # retrieve architecture (a dict with key 'Model <modelnumber>')
    archidefs = open(f"{args.savepath}/archidefs_{args.jobnumber}.pkl",'rb')
    architectures = pickle.load(archidefs)
    archidefs.close()
    args.model_args = architectures[f"Model {args.modelnumber}"] # this will be a list with name and list of two dicts that will contain the archi_params for the two networks
    del architectures
    args.nr_features = []
    args.nr_features.append(args.model_args[1][0]['rnn_params']['hidden_dim'])
    args.nr_features.append(args.model_args[1][1]['ffnmerge_params']['final_output_size'])

    # retrieve full file of returns
    args.returns = pd.read_pickle(f"{args.savepath}/asset_returns_{args.jobnumber}.pkl")
    # turn l2regparam into float
    if args.l2regparam != 'none': 
        args.l2regparam = float(args.l2regparam)
    else:
        args.l2regparam = 0.0
    # turn l1regparams into float
    args.l1regparam = []
    for val in [args.l1regparam_sdf, args.l1regparam_mom]:
        if  val != 'none':
            args.l1regparam.append(float(val))
        else:
            args.l1regparam.append(0.0)

    sdf_optimizer_params = argparse.ArgumentParser(description = "Args for sdf optimizer")
    moments_optimizer_params = argparse.ArgumentParser(description = "Args for moment optimizer")
    
    sdf_optimizer_params.optimizer_name = args.optimizer_name
    sdf_optimizer_params.learning_rate = args.learning_rate_sdf
    sdf_optimizer_params.l2regparam = args.l2regparam
    sdf_optimizer_params.momentum = args.sdf_momentum
    
    moments_optimizer_params.optimizer_name = args.optimizer_name
    moments_optimizer_params.learning_rate = args.learning_rate_mom
    moments_optimizer_params.l2regparam = args.l2regparam
    moments_optimizer_params.momentum = args.moments_momentum
    
    args.best_mispricing_loss = np.Inf
    args.best_sharpe_ratio = -np.Inf
    
    args.path_tail = f"jnr_{args.jobnumber}_bs_{args.batch_size}_lrs_{(args.learning_rate_sdf,args.learning_rate_mom)}_l2reg_{args.l2regparam}_l1regs_{(args.l1regparam_sdf,args.l1regparam_mom)}_rs_{args.random_seed}"
    
    args.world_size = 1
    main_worker(0, args, sdf_optimizer_params, moments_optimizer_params)
    
    if not args.printc: 
        sys.stdout.close()

def main_worker(gpu, args, sdf_optimizer_params, moments_optimizer_params):
    
    start = time.time()
    errors = []
    
    torch.cuda.device(gpu)
    TrainUtils.random_seeding(args.random_seed, use_cuda=True)

    # initialize model
    # model_args will be a list of the form [arch_name,arch_spec]
    # arch_name being a str and arch_spec being a two-element list of dictionaries
    # first element of arch_spec are the params for the sdf network in dict form, second are the params for the moments network in dict form
    network = TrainUtils.init_gan_model(*args.model_args, world_size=args.world_size)
    for model in network:
        model.cuda(gpu)      
    # initialize optimizers
    optimizer = []
    optimizer.append(init_optimizer(network[0].parameters(), sdf_optimizer_params))
    optimizer.append(init_optimizer(network[1].parameters(), moments_optimizer_params))

    print("==================================================================================================================================")
    print(f'\n\nLoaded for global training.\nTraining Model {args.modelnumber} with params {args.path_tail}.\n')

    # Data loading 
    init_data = TrainUtils.InitDataloader(args, labels = False)
    
    # initialize train_val instance
    train_val = TrainUtils.TrainValGAN(
        idxdates = init_data.idxdates,
        network = network,
        optimizer = optimizer,
        subepochs = args.sub_epochs,
        world_size = args.world_size,
        returns = args.returns,
        nr_features = args.nr_features,
        train_loader = init_data.train_loader,
        val_loader = init_data.val_loader,
        test_loader = init_data.test_loader, 
        val_criterion = 'sharpe_ratio',
        early_stopping = args.early_stopping,
        l1regparam= args.l1regparam,
        unc_loss_weight = 1.0,
        sdf_macro_states = None,
        moments_macro_states = None,
        test_assets = None,
        sdf_weights = None,
        verbose = args.verbose)
    
    print("Loaded data and the estimation utilities.")
        
    for epoch in range(1,(args.num_epochs+1)):
        
        # train for one epoch
        train_val.train_one_ep(epoch)  
                
        if args.validate:
            train_val.validate_one_ep(epoch)
            if args.early_stopping[0] < np.Inf:
                if train_val.stop_early:
                    break
            
    if args.validate:
        
        # test set performance 
        train_val.testset_performance()
        train_val.gather_performance()
       
        if args.get_betanet_labels:
            betanet_labels = TrainUtils.betanet_labels(jobnumber = args.jobnumber, pf_returns_df = train_val.final_returns, savepath = args.savepath)
            betanet_labels.to_pickle(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}.pkl")
            betanet_labels.to_excel(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_betanet_labels_{args.path_tail}.xlsx")
            
        # save best/last models
        names = ['SDF','Moments']
        for i in range(2):
            if args.save_model: 
                if train_val.best_model_weights != None:    
                    train_val.network[i].load_state_dict(train_val.best_model_weights[i])
                model_pth = f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_{names[i]}_Network_{args.path_tail}.pt"
                torch.save(train_val.network[i], model_pth)
                
        # save macro states, sdf_weights, and test assets of the best/last model
        dataframes = {}
        dataframes['final_macro_states_longformat'] = pd.DataFrame(train_val.final_macro_states.detach().cpu().numpy(), 
                                          columns = [f"{name}_ms_{i}" for name in names for i in range(1,args.nr_features[0]+1)])
        dataframes['final_macro_states_longformat'] = args.returns[['t_index','date_id']].merge(dataframes['final_macro_states_longformat'].reset_index().rename(columns = {'index':'t_index'}),
                                                                    on = 't_index', how = 'inner')
        dataframes['final_macro_states'] = dataframes['final_macro_states_longformat'].drop(columns = ['t_index']).drop_duplicates()
        
        if not (network[0].propagate_state or network[1].propagate_state):
            dataframes['final_macro_states'] = dataframes['final_macro_states'].groupby(['date_id']).mean().reset_index()
        
        dataframes['final_returns'] = train_val.final_returns
        dataframes['final_weights'] = pd.DataFrame(train_val.final_weights.detach().cpu().numpy(), columns = ['weights'])
        dataframes['final_weights'] = args.returns[['t_index','date_id']].merge(dataframes['final_weights'].reset_index().rename(columns = {'index':'t_index'}),
                                                                    on = 't_index', how = 'inner')
        dataframes['final_test_assets'] = pd.DataFrame(train_val.final_test_assets.detach().cpu().numpy(),
                                         columns = ['test_asset_'+str(i) for i in range(1,args.nr_features[1]+1)])
        dataframes['final_test_assets'] = args.returns[['t_index','date_id']].merge(dataframes['final_test_assets'].reset_index().rename(columns = {'index':'t_index'}),
                                                                    on = 't_index', how = 'inner')
        
        dataframes['final_features'] = args.returns.merge(dataframes['final_macro_states_longformat'], how='inner', 
                                  on = ['t_index','date_id']).merge(dataframes['final_weights'], how='inner', 
                                  on = ['t_index','date_id']).merge(dataframes['final_test_assets'],
                                  how='inner', on = ['t_index','date_id']).merge(dataframes['final_returns'], how = 'inner', on = 'date_id')
        if args.get_betanet_labels:
            dataframes['final_features'] = dataframes['final_features'].merge(betanet_labels, how = 'inner', on = ['t_index','date_id'])
        # save performance dataframes
        train_losses = train_val.performance_dict['unc_sdf_train_data_loss'].merge(train_val.performance_dict['cond_sdf_train_data_loss']
                                                                , how = 'inner', on = ['phase','epoch']).merge(train_val.performance_dict['mom_train_data_loss'],
                                                                how = 'inner', on = ['phase','epoch'])
        perf_dict = {'train_losses': train_losses, 'sharpe_ratio': train_val.performance_dict['sharpe_ratio'], 'val_test_data_loss':train_val.performance_dict['val_test_data_loss']}
        
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
                errors.append((args.path_tail,error))
        
        if len(errors)>0:
            save_strlist_txt(f"{args.savepath}/Errors_{args.project_name}_Model_{args.modelnumber}_{args.path_tail}", errors)
        
        # Save pics of performance measures
        if args.view_losses:
            for key, df in train_val.performance_dict.items():
                TrainUtils.plot_performance(df, listgrpbycols = ['phase'], value = key, pltpath = args.savepath,
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
