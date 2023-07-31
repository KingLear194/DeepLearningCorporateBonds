#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:30:10 2021

@author: jed169

A script for training for time-series prediction.
Multiprocessing features: tested for data parallelism on one node of GPUs.

"""

import os
import sys

from settings import datapath

import pandas as pd
import numpy as np
import pickle 
import json
import time
import copy
from sklearn.preprocessing import StandardScaler

from datetime import date, datetime
today = date.today().strftime("%b-%d-%Y")
now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist


from TrainUtils import random_seeding
import performancemeasures as perfmes
import TStraintest
import NNmodels
from aux_utilities import create_dir_ifnexists, get_train_val_data, DatasetSimple, pred_metrics_NN, l1_penalty
import argparse


'''
#######################################
Read off from terminal of the node and prepare the arguments for mp.spawn and later dist.init_process_group
#######################################
'''
parser = argparse.ArgumentParser(description  = 'Arguments for pytorch multi-processing one-shot script')

parser.add_argument('-device', '--device', type=str, default = 'cuda', help = 'device to train on')
parser.add_argument('-rs', '--random_seed', type = int, default = 19, help = 'random seed for all packages')
parser.add_argument('-proj','--project_name', type = str, default = 'FFNRNNMerge_trial_prediction=True', help = 'Project name')
parser.add_argument('-jobnr','--jobnumber', type = int, default = 1, help = 'job number when submitting multiple jobs to the cluster')

# DDP arguments
parser.add_argument('-n', '--nrnodes', default = 1, type = int, metavar='N', help = 'number of nodes to be used')
parser.add_argument('-g', '--nrgpus', default = 1, type = int, help = 'number of gpus per node')
parser.add_argument('-rank', '--noderank', default = 0, type = int, help = 'rank of current node among all nodes, 0 is the master node')
parser.add_argument('-masteraddr' , '--masteraddress', type = str, help = 'IP address of the master node')
parser.add_argument('-masterport' , '--freeport', type = str, help = 'free port of the master node')

'''args for training and validation'''

# hyperparameters we optimize over in the validation phase
parser.add_argument('-modelnr', '--modelnumber', default = 1, type = int, help = 'modelnumber for training and val')
parser.add_argument('-b', '--batch_size', default = 32, metavar = 'bs', help = 'total batch_size of all GPUs on the current node. Note: if want to give full batch then need to set this to a number above the data-length.')
parser.add_argument('-lr', '--learning_rate', default = 0.01, type = float, metavar = 'LR', help = 'initial learning rate')
parser.add_argument('-l2reg', '--l2regparam', default = 'none', metavar = 'L2Reg', help = 'weight for L2 regularization')
parser.add_argument('-l1reg', '--l1regparam', default = 'none', metavar = 'L1Reg', help = 'weight for L1 regularization')

# hyperparameters and other modeling choices we do not optimize over in the validation phase
parser.add_argument('-validate', '--validate', type = bool, nargs = "?", const=True, default=True, metavar = 'validate', help = 'bool for validate')
parser.add_argument('-epochs', '--num_epochs', default = 10, type = int, metavar = 'NrEpochs', help = 'Number of epochs to train')
parser.add_argument('-optimizer', '--optimizer_name', default = 'Adam', type = str, metavar = 'optimizer', help = 'Optimizer name')
parser.add_argument('-criterion', '--train_criterion_name', default = 'MSE', type = str, metavar = 'criterion', help = 'criterion name')
parser.add_argument('-val_criterion', '--val_criterion_name', default = 'MSE', type = str, metavar = 'val criterion', help = 'val criterion name; may be different from train criterion')

parser.add_argument('-val_metrics', '--performance_measures_names', default = ['mean_squared_error', 'rmse', 'minus_r2'], type = list, metavar = 'performance measures', help = 'performance measures for validation')
parser.add_argument('-xscaling', '--xscaling', default = True, type = bool, nargs = "?", const=True, metavar = 'xscaling', help = 'bool for xscaling')
parser.add_argument('-yscaling', '--yscaling', default = True, type = bool, nargs = "?", const=True, metavar = 'yscaling', help = 'bool for yscaling')

# other
parser.add_argument('-verbose', '--verbose', default = 1, type = float, metavar = 'verbose', help = 'verbose level')
parser.add_argument('-outpreds', '--output_predictions', default = True, type = bool, nargs = "?", const=True, metavar = 'predictions', help = 'output predictions in separate pickle file')
parser.add_argument('-savemodel', '--save_model', default = True, type = bool, nargs = "?", const=True,metavar = 'save model', help = 'bool for save model')
parser.add_argument('-viewlosses', '--view_losses', default = True, type = bool, nargs = "?", const=True, metavar = 'view losses', help = 'bool for view losses')
parser.add_argument('-printc' , '--printc', type =bool, default = True, help = 'print on console, otherwise save on .out file')
parser.add_argument('-time_start' , '--time_start', type=str, default = "no_timelog", help = 'get start time')
parser.add_argument('-config', '--config_file_path', default = 'none', type = str, help = 'config file name to load')

# initialize global variables
start_time = time.time()
date_idx = 'date'
loss_hist = {} 

def main():
    
    args = parser.parse_args()    
    # load the config parameters
    if args.config_file_path != 'none':
        with open(args.config_file_path, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    args.savepath=f'{datapath}/current_train_val_test/{args.project_name}'
    create_dir_ifnexists(args.savepath)
    
    if not args.printc: 
        sys.stdout = open(f"{args.savepath}/{args.project_name}_log_{args.time_start}.out", 'a')
    
    # retrieve architecture
    archidefs = open(f"{args.savepath}/archidefs_{args.jobnumber}.pkl",'rb')
    architectures = pickle.load(archidefs)
    archidefs.close()
    current_arch = architectures[f"Model {args.modelnumber}"]
    del architectures
    args.dataparams = current_arch[0]
    args.model_args = current_arch[1:]
    
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
        
    args.best_loss = np.Inf  
    args.path_tail = f"jnr_{args.jobnumber}_bs_{args.batch_size}_lr_{args.learning_rate}_l2reg_{args.l2regparam}_l1reg_{args.l1regparam}_rs_{args.random_seed}"
    
    if args.device=='cuda':
        args.world_size = args.nrgpus*args.nrnodes
    # initialize master gpu-node
        init_os(args)
        if args.world_size> 1: 
            mp.spawn(main_worker, nprocs = args.world_size, args = (args,)) 
        else:
            main_worker(0, args)
    else:
        args.world_size = 1
        main_worker(0, args)
        
    if not args.printc: 
        sys.stdout.close()

def main_worker(gpu, args):
    
    # initialize global vars and rank
    start = time.time()
    
    # rank of the process needs to be the global rank here of the gpu across all gpu-s
    if args.device=='cuda':
        rank = setup_dist_group(gpu, args) 
        #print("RANK IS ", rank)
        random_seeding(args.random_seed+rank, use_cuda=True)
    else: 
        rank=0
        args.current_proc_rank=0
    print(f'The current rank is {rank}.')
    
    args.best_weights = None
    global loss_hist
    args.gpu = gpu
    loss_hist[args.current_proc_rank] = []
    
    # initialize model
    model = init_model(*args.model_args, world_size = args.world_size)
    
    # wrap model with DDP
    if args.device=='cuda':
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        if args.world_size>1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [gpu], output_device=gpu)
        
    # initialize loss criterion
    train_criterion = init_criterion(args.train_criterion_name, args.l1regparam)
    val_criterion = init_criterion(args.val_criterion_name, 0)#.cuda(gpu)
    
    optimizer = init_optimizer(model.parameters(), args)
    
    print(f'\nLoaded criterion and optimizer.\nlearning Rate is {args.learning_rate}, weight decay is {args.l2regparam} and L1-loss parameter is {args.l1regparam}.\n')

    # Data loading code 
    init_data = InitDataloader(rank, args)
    train_sampler, train_loader, val_loader = init_data.train_sampler, init_data.train_loader, init_data.val_loader
    args.train_data_len, args.val_data_len = init_data.train_data_len, init_data.val_data_len
    
    # earlystopping
    if args.early_stopping[0] is not np.Inf: 
        earlystop_counter = 0
        early_stopping_active = True
    else:
        early_stopping_active = False
        
    for epoch in range(1,(args.num_epochs+1)):
        # set epoch for train sampler before launching the train_one_ep function
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print(f'Current epoch is {epoch}')
        
        # train for one epoch
        train_one_ep(train_loader, model, train_criterion, optimizer, args, epoch)
        
        if args.validate:
            current_val_loss = validate_one_ep(val_loader, model, val_criterion, args, epoch)
            '''
            Note: in difference to nolabels_one_shot.py and one_shot_gan.py, in the early stopping routine here, 
            the direction of improvement of the val_criterion is minimization. This is because we are typically using this with MSE loss
            '''
            improvement = (current_val_loss + args.early_stopping[1]) < args.best_loss
            
            if improvement:
                print(f'\nUpdating global best model weights based on validation of epoch {epoch} in gpu {gpu}.\n')
                args.best_weights = copy.deepcopy(model.state_dict())
                args.best_loss = current_val_loss
                earlystop_counter = 0
                
            elif early_stopping_active:
                earlystop_counter += 1
                if earlystop_counter>= args.early_stopping[0]:
                    print(f"\n----Early stopping in epoch {epoch} in gpu {gpu} with parameters:\npatience: {args.early_stopping[0]}\ndelta: {args.early_stopping[1]}")
                    break                
                
    # end the training with the best weights for the model across all gpus!
    if args.world_size > 1:
        dist.barrier() 
    if args.validate and args.best_weights != None: 
        model.load_state_dict(args.best_weights)
    
    if rank == 0 and args.save_model and args.best_weights != None: #don't save models with val loss=inf
        args.model_savepth = f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_{args.path_tail}.pt"
        torch.save(model, args.model_savepth)
    
    end = time.time()
    time_elapsed = end-start
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'\nSynchronized training in {args.device} {rank} complete in {minutes} minutes and {seconds} seconds.\n')
    if args.world_size > 1:
        dist.barrier()
    ## calculate performance of final model on validation and test set on rank 0
    if rank == 0 and args.validate and args.best_weights != None: # don't calculate losses for models with val loss = inf
        keeptrack_perf = []
        performance_measures = perf_measures_init(args)

        model = torch.load(args.model_savepth)
        # note: here we want to calculate data losses, so we give yscaler, but we don't need to give xscaler since x_val is already scaled above
        per_test, train_pred = pred_metrics_NN(model = model, features = init_data.data[0], outcomes = init_data.data[1], 
                    performance_measures = performance_measures, xscaler = init_data.xscaler, target_dim = init_data.target_dim, yscaler = init_data.yscaler, device = args.device)
        per_val, val_pred = pred_metrics_NN(model = model, features = init_data.data[2], outcomes = init_data.data[3], 
                    performance_measures = performance_measures, xscaler = init_data.xscaler, target_dim = init_data.target_dim, yscaler = init_data.yscaler, device = args.device)
        per_test, test_pred = pred_metrics_NN(model = model, features = init_data.data[4], outcomes = init_data.data[5], 
                    performance_measures = performance_measures, xscaler = init_data.xscaler, target_dim = init_data.target_dim, yscaler = init_data.yscaler, device = args.device)
        # note: test data are not scaled at the start, so need to give both xscaler and yscaler here
        if args.output_predictions:
            with open(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_TrainPredictions_{args.path_tail}.npy", 'wb') as ft:
                np.save(ft, train_pred)
            with open(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_ValPredictions_{args.path_tail}.npy", 'wb') as fv:
                np.save(fv, val_pred)
            with open(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_TestPredictions_{args.path_tail}.npy", 'wb') as f:
                np.save(f, test_pred)
            pred = np.concatenate([train_pred, val_pred, test_pred])
            with open(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_AllPredictions_{args.path_tail}.npy", 'wb') as f:
                np.save(f, pred)
        keeptrack_perf.append([f"Model {args.modelnumber}", init_data.data[3].shape, args.batch_size, args.learning_rate, args.l2regparam, args.l1regparam, args.best_loss, *per_val])
        keeptrack_perf.append([f"Model {args.modelnumber}",init_data.data[5].shape, args.batch_size, args.learning_rate, args.l2regparam, args.l1regparam, args.best_loss, *per_test])
        
        performance_results = pd.DataFrame(data = keeptrack_perf, columns = ['modelnumber','data shape','batch_size','learning_rate','l2regparam', 'l1regparam', 'best_val loss',
                                        *args.performance_measures_names], index = ['Validation', 'Test'])
        performance_results.to_pickle(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_ValTestResults_{args.path_tail}.pkl")
        performance_results.to_excel(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_ValTestResults_{args.path_tail}.xlsx")
        
        loss_hist[args.current_proc_rank].append(('test', 0, per_test[0]))
        loss_hist[args.current_proc_rank].append(('best_val', 0, args.best_loss))
        loss_hist = pd.DataFrame(loss_hist[args.current_proc_rank], columns = ['phase','epoch',f'{args.performance_measures_names[0]}'])
        loss_hist.to_pickle(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_{args.performance_measures_names[0]}_{args.path_tail}.pkl")
        loss_hist.to_excel(f"{args.savepath}/{args.project_name}_Model_{args.modelnumber}_{args.performance_measures_names[0]}_{args.path_tail}.xlsx")
        
        print("\n\n")

        
def train_one_ep(train_loader, model, criterion, optimizer, ddp_args, epoch):
    
    global loss_hist

    # switch to train mode
    model.train()
    running_loss = 0.0
    data_len = 0
    for x,y in train_loader:
        data_len += x.size(0)
        if ddp_args.device == 'cuda':
            x = x.cuda(ddp_args.gpu, non_blocking = True)
            y = y.cuda(ddp_args.gpu, non_blocking = True)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
        
            y_pred = model.forward(x)            
            loss = criterion(y = y,y_pred = y_pred,params = model.parameters(), lam = ddp_args.l1regparam) 
            dat_loss = loss - l1_penalty(params = model.parameters(), lam = ddp_args.l1regparam)
            data_loss = dat_loss.cpu().item()
            data_loss = loss.cpu().item()
            
            loss.backward()
            optimizer.step()
            
        running_loss += data_loss*x.size(0)
    
    epoch_loss = running_loss/data_len
    
    if ddp_args.verbose > 0 and epoch%int(1/ddp_args.verbose) == 0:
        print('{} Loss: {:.4f} in gpu {} in epoch {}\n'.format('Train', epoch_loss, ddp_args.current_proc_rank, epoch))
     
    loss_hist[ddp_args.current_proc_rank].append(('train', epoch, epoch_loss))
 
def validate_one_ep(val_loader, model, val_criterion, ddp_args, epoch):
    
    global loss_hist
    
    model.eval()
    running_val_loss = 0.0
    data_len = 0
    for x,y in val_loader:
        
        data_len+=x.size(0)
        if ddp_args.device == 'cuda':
            x = x.cuda(ddp_args.gpu, non_blocking = True)
            y = y.cuda(ddp_args.gpu, non_blocking = True)
                
        with torch.set_grad_enabled(False):
        
            y_pred = model.forward(x)
            loss = val_criterion(y = y,y_pred = y_pred,params = model.parameters(), lam = 0)    
    
        running_val_loss += loss.cpu().item()*x.size(0)
    epoch_val_loss = running_val_loss/data_len
    loss_hist[ddp_args.current_proc_rank].append(('val', epoch, epoch_val_loss))
    
    if ddp_args.verbose > 0 and epoch%int(1/ddp_args.verbose) == 0:
        print('{} Loss: {:.4f} in gpu {} in epoch {}\n'.format('Val', epoch_val_loss, ddp_args.current_proc_rank, epoch))
    
    return epoch_val_loss
    

def init_os(ddp_args):
    # set environ variables of master node
    os.environ['MASTER_ADDR'] = ddp_args.masteraddress
    os.environ['MASTER_PORT'] = ddp_args.freeport

def setup_dist_group(gpu, ddp_args):
    
    #calculate global rank, noderank is contained in ddp_args
    rank = ddp_args.noderank*ddp_args.nrgpus + gpu
    # distinit group
    # we are passing the address of the master node and its free port here via ddp_args already in the estimate_ddp
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size=ddp_args.world_size, rank = rank)
    ddp_args.current_proc_rank = rank
    
    return rank

def init_model(arch_name, arch_spec, world_size=1):
    
    model = None
    
    print(f'The architecture type is {arch_name}.')
        
    if arch_name == 'FFNStandard':
        arch_spec.update({'world_size':world_size})
        model = NNmodels.FFNStandardTorch(**arch_spec)
        
    elif arch_name == 'FFNMerge':
        arch_spec.update({'world_size':world_size})
        model = NNmodels.FFNMergeTorch(**arch_spec)
        
    elif arch_name == 'RecurrentStandard':
        model = NNmodels.RecurrentNNStandardTorch(**arch_spec)
    
    elif arch_name == 'Recurrent-ED':
        model = NNmodels.RecurrentEncoderDecoder(**arch_spec)
    
    elif arch_name == 'FFN-Recurrent-Merge':
        model = NNmodels.FFNRecurrentMerge(**arch_spec)
        
    else:
        print("Invalid architecture name, exiting...")
        return
    
    return model

def init_criterion(criterion_name, l1reg = 0):
    
    if criterion_name == 'MSE':
           
        def criter(y,y_pred,params, lam):
            # it must return mean of overall loss over batch
            mseloss = nn.MSELoss(reduction = 'none')
                
            crit = torch.mean(mseloss(y,y_pred)) + l1_penalty(params, lam)
                
            return crit
        cri = criter
    else: 
        raise ValueError("Loss criterion {0} not found.".format(criterion_name))
        
    return cri


def init_optimizer(params_to_update, args):
    
    print(f"We load the optimizer {args.optimizer_name}")
    
    if isinstance(args.l2regparam, str):
        wd = 0
    else:
        wd = args.l2regparam
    
    if args.optimizer_name == 'SGD':
        optimizer = optim.SGD(
            params_to_update, 
            lr = args.learning_rate,
            momentum = args.momentum,
            weight_decay = wd)
        
    elif args.optimizer_name == 'Adam':
        optimizer = optim.Adam(
            params_to_update, 
            lr = args.learning_rate,
            weight_decay = wd)  
    
    elif args.optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(
            params_to_update, 
            lr = args.learning_rate,
            momentum = args.momentum,
            weight_decay = wd)  
        
    else:
        raise ValueError("Optimizer {0} not found.".format(args.optimizer_name))
        return -1
        
    return optimizer

class InitDataloader:
    
    def __init__(self,
                    rank,
                    ddp_args):
        
        print("Loading current train- and validation data.")
        
        self.data, self.xscaler, self.yscaler = dataprep(ddp_args)
        
        self.batch_size = ddp_args.batch_size
        
        self.ddp_args = ddp_args
    
        self.train_data_len = self.data[0].shape[0]
        self.val_data_len = self.data[3].shape[0]
        self.target_dim = self.data[1].shape[1]
    
        print("Loading current train- and validation data.")
    
        if self.data is not None:
            self.train_dataset = DatasetSimple(self.data[0], self.data[1])
            self.validation_dataset = DatasetSimple(self.data[2],self.data[3])
        
        if isinstance(self.batch_size, str) and self.batch_size == 'full':
            self.batch_size = self.train_dataset.__len__()
        else:
            self.batch_size = int(self.batch_size)
        
        # since device_ids are set in the main worker, then we divide the batch_size among the nr of gpus we have 
        self.batch_size = int(self.batch_size/self.ddp_args.world_size)
        print(f'batch_size on one process is {self.batch_size}')
        
        # set up and wrap sampler 
        if ddp_args.world_size>1:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=self.ddp_args.world_size,
                                                                    rank = rank, shuffle = False, drop_last=False)
        else:
            self.train_sampler = None
        torch.manual_seed(ddp_args.random_seed)
    
        self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset, batch_size=self.batch_size, 
                                                        drop_last=True, shuffle = False, pin_memory=True, 
                                                        sampler = self.train_sampler) 
        self.val_loader = torch.utils.data.DataLoader(dataset = self.validation_dataset, batch_size = min(self.batch_size,1024), 
                                                      drop_last=True, shuffle = False, pin_memory=True)    
    

def dataprep(args, target_dim = 1, date_idx = 'date_id'):
    
    '''
    target_dim is dimension of labels, to be selected from dataframe of y_test and y_train_val below
    '''

    print('-'*40)
    print('-'*40)
    print(f'\nData prep for Model {args.modelnumber}.\n')
    
    # start date generator
    splitdata = TStraintest.TSTrainTestSplit(**args.dataparams)   
    
    print('\nCurrently loading training and validation data.\n')
    
    X_train_val = pd.read_pickle(f'{args.savepath}/X_train_val_{args.jobnumber}.pkl')
    y_train_val = pd.read_pickle(f'{args.savepath}/y_train_val_{args.jobnumber}.pkl')
    X_test = pd.read_pickle(f'{args.savepath}/X_test_{args.jobnumber}.pkl').to_numpy()
    y_test = pd.read_pickle(f'{args.savepath}/y_test_{args.jobnumber}.pkl').to_numpy()
    
    print(f'\nTotal batch size for train-val is {args.batch_size}.\n')
    print('-'*40)
    
    train_dates, val_dates = next(splitdata.split(y_train_val.reset_index())) 
    x_train, y_train, x_val, y_val = get_train_val_data(X_train_val, y_train_val, train_dates, val_dates, date_idx=date_idx)
    
    x_train = x_train.drop(columns = [date_idx]).to_numpy()
    x_val = x_val.drop(columns = [date_idx]).to_numpy()
    y_train = y_train.iloc[:,-target_dim].to_numpy()
    y_val = y_val.iloc[:,-target_dim].to_numpy()
        
    if args.xscaling == True:
        xscaler = StandardScaler()
        x_train = xscaler.fit_transform(x_train).astype(np.float32)
        x_val = xscaler.transform(x_val).astype(np.float32)            
    else:
        xscaler = None
        x_train = x_train.astype(np.float32)
        x_val = x_val.astype(np.float32)
    if args.yscaling == True:
        yscaler = StandardScaler()
        y_train = yscaler.fit_transform(y_train.reshape(-1,target_dim).astype(np.float32))
        y_val = yscaler.transform(y_val.reshape(-1,target_dim).astype(np.float32))
    else:
        yscaler = None
        y_train = y_train.reshape(-1,target_dim).astype(np.float32)
        y_val = y_val.reshape(-1,target_dim).astype(np.float32)
          
    data = [x_train, y_train, x_val, y_val, X_test, y_test]    

    return data, xscaler, yscaler

def perf_measures_init(ddp_args):
    
    performance_measures=[]
    for name in ddp_args.performance_measures_names:
        performance_measures.append(getattr(perfmes, name))
        
    return performance_measures

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    main()
