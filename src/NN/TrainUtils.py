"""
Created on Thu Jun 10 19:09:17 2021

@author: jed-pitt

TrainUtils 

"""
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import copy
from sklearn.preprocessing import StandardScaler
import torch

import performancemeasures as pm
import GANmodels
from TStraintest import TSTrainTestSplit
from aux_utilities import get_train_val_data, DatasetSimple, l1_penalty, performance_df

from settings import datapath

def random_seeding(seed_value, use_cuda):
    random.seed(seed_value) # random vars
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda:
        # gpu vars
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

'''Utils for the model data preprocessing, training and validation'''
def dataprep(fixed, jobnumber, savepath, date_idx = 'date_id', xscaling = True, yscaling = True, labels = False):
    
    '''
    fixed is the share of val in train_val
    NOTE: The data will be returned in numpy form. We drop the asset_id column at the start.
    X_train_val first columns shld be t_idx, date_id
    Output: data, scalers, nr of dates for train_val.
    '''

    print('-'*40)
    print('-'*40)
    
    # start date generator
    splitdata = TSTrainTestSplit(fixed = fixed, date_idx=date_idx)   
    X_train_val = pd.read_pickle(f'{savepath}/X_train_val_{jobnumber}.pkl').set_index(['t_index','date_id'])
    X_test = pd.read_pickle(f'{savepath}/X_test_{jobnumber}.pkl').set_index(['t_index','date_id'])
    
    if labels:
        y_train_val = pd.read_pickle(f'{savepath}/y_train_val_{jobnumber}.pkl').drop(columns = 'asset_id').set_index(['t_index','date_id'])
        y_test = pd.read_pickle(f'{savepath}/y_test_{jobnumber}.pkl').drop(columns = 'asset_id').set_index(['t_index','date_id']).to_numpy().astype(np.float32)
    else:
        y_train_val = pd.DataFrame(np.zeros(shape = (X_train_val.shape[0],1)),index = X_train_val.index)
        y_test = pd.DataFrame(np.zeros(shape = (X_test.shape[0],1)),index = X_test.index)
    print('-'*40)
    ## preprocess train-val data 
    train_dates, val_dates = next(splitdata.split(X_train_val.reset_index())) 
    x_train, y_train, x_val, y_val = get_train_val_data(X_train_val, y_train_val, train_dates, val_dates, date_idx=date_idx)
    idxdates_train = x_train.iloc[:,:2].to_numpy().astype(np.float32)
    idxdates_val = x_val.iloc[:,:2].to_numpy().astype(np.float32)
    idxdates_test = X_test.reset_index().iloc[:,:2].to_numpy().astype(np.float32)
        
    if xscaling == True:
        xscaler = StandardScaler()
        x_train = xscaler.fit_transform(x_train.iloc[:,2:]).astype(np.float32)
        x_val = xscaler.transform(x_val.iloc[:,2:]).astype(np.float32)
        x_train = np.concatenate((idxdates_train, x_train), axis = 1)
        x_val = np.concatenate((idxdates_val, x_val), axis = 1)
                
    else:
        xscaler = None
        x_train = x_train.to_numpy().astype(np.float32)
        x_val = x_val.to_numpy().astype(np.float32)
                
    if yscaling == True and labels:
        yscaler = StandardScaler()
        y_train = yscaler.fit_transform(y_train.to_numpy()).astype(np.float32)
        y_val = yscaler.transform(y_val.to_numpy()).astype(np.float32)
        y_train = np.concatenate((idxdates_train, y_train), axis = 1)
        y_val = np.concatenate((idxdates_val, y_val), axis = 1)
                
    else:
        yscaler = None
        y_train = None 
        y_val = None 
    
    X_test=X_test.reset_index().to_numpy().astype(np.float32)
    y_test=y_test.to_numpy().astype(np.float32)
    data = [x_train, y_train, x_val, y_val, X_test, y_test]
    idxdates = [idxdates_train, idxdates_val, idxdates_test]

    return data, idxdates, xscaler, yscaler


class InitDataloader:

    def __init__(self,
                ddp_args,
                labels = False):
            
        print("Loading current train- and validation data.")
        
        self.data, self.idxdates, self.xscaler, self.yscaler = dataprep(fixed = ddp_args.train_val_split, 
                                                savepath = f"{datapath}/current_train_val_test/{ddp_args.project_name}",
                                                jobnumber = ddp_args.jobnumber,
                                                xscaling=ddp_args.xscaling, yscaling=ddp_args.yscaling, labels=labels)
        
        self.batch_size = ddp_args.batch_size
        
        print("Loading current train- and validation data.")
    
        if self.data is not None:
            self.train_dataset = DatasetSimple(self.data[0], self.data[1])
            self.validation_dataset = DatasetSimple(self.data[2],self.data[3])
            
        self.test_dataset = DatasetSimple(self.data[4], self.data[5]) 
        
        if isinstance(self.batch_size, str) and self.batch_size == 'full':
            self.batch_size = self.train_dataset.__len__()
        else:
            self.batch_size = int(self.batch_size)
        
        # since device_ids are set in the main worker, then we divide the batch_size among the nr of gpus we have 
        self.batch_size = int(self.batch_size/ddp_args.world_size)
        print(f'\nTotal batch size for train-val is {ddp_args.batch_size}.\n')
        print(f'Batch_size on one process is {self.batch_size}.')
        
        torch.manual_seed(ddp_args.random_seed)
    
        self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset, batch_size=self.batch_size, 
                                                        drop_last=False, shuffle = False, pin_memory=True) #worker_init_fn=seed_worker, generator=g)
        self.val_loader = torch.utils.data.DataLoader(dataset = self.validation_dataset, batch_size = self.batch_size, 
                                                      drop_last=False, shuffle = False, pin_memory=True) 
        
        self.test_loader = torch.utils.data.DataLoader(dataset = self.test_dataset,  batch_size = self.batch_size,#batch_size = 1024, 
                                                      drop_last=False, shuffle = False, pin_memory=True)
    

def init_gan_model(arch_name, arch_spec, world_size = 1):
    
    print(f'The architecture type is {arch_name}.')
    arch_params = ['ffnmerge_params','rnn_params','nr_indiv_features','macro_input_length','propagate_state']
        
    if arch_name == 'GAN':
        sdf_network = GANmodels.GANNet(**{key:arch_spec[0][key] for key in arch_params}, 
                macro_states = torch.zeros(size=(arch_spec[0]['T']+1,arch_spec[0]['rnn_params']['hidden_dim'])).cuda(0),
                network_type='SDF',world_size=world_size)
        moments_network = GANmodels.GANNet(**{key:arch_spec[1][key] for key in arch_params}, 
                macro_states = torch.zeros(size=(arch_spec[1]['T']+1,arch_spec[1]['rnn_params']['hidden_dim'])).cuda(0),
                network_type='Moments', world_size=world_size)
    else:
        print("Invalid architecture name, exiting...")
        return
    
    return [sdf_network, moments_network]

def betanet_labels(jobnumber, savepath, pf_returns_df, pf_return_label = 'pf_return'):
    
    returns = pd.read_pickle(f"{savepath}/asset_returns_{jobnumber}.pkl")
    returns = returns.merge(pf_returns_df, how = 'inner', on = 'date_id')
    returns['RF'] = returns['ret_e']*returns[pf_return_label]
    
    return returns[['t_index','date_id','RF']]

def ensembled_betalabel():
    '''
        Temporary function
    '''
    rets = pd.read_pickle(glob.glob("asset_returns_*.pkl")[0])
    rets = rets.merge(pd.read_pickle(glob.glob("*final_returns_*_mean_ensembled.pkl")[0]), how = 'inner', on = 'date_id')
    rets['betas_mean'] = rets['ret_e']*rets['return_mean']
    
    return rets[['t_index','date_id','betas_mean']]

def plot_performance(df, listgrpbycols, value, title, pltpath):
    '''Note: this needs the call of self.gather_losses() first in TrainValGAN'''
    
    fig, ax = plt.subplots(figsize = (10,4))
    for key, grp in df.groupby(by = listgrpbycols):
        ax.plot(grp['epoch'],grp[value],label = key)
    ax.legend()
    plt.title(title)
    plt.savefig(f"{title}.tiff")
    plt.close()

def normalize_weights(weights, constraint_type = 'L2', L = 0.1, epsilon = 1e-4):
    '''
    We give here upper bounds for leverage, with options both for L1 and L2.
    '''
    if constraint_type=='L1':
        norm = weights.abs().sum()+torch.tensor(epsilon)
    elif constraint_type=='L2':
        norm = torch.sqrt((weights**2).sum()+torch.tensor(epsilon))
    else:
        raise ValueError("Either L2 or L1 needs to be activated!")
    weights = float(L)*weights/norm
    
    return weights


class TrainValBase:
    
    def __init__(self,
                    idxdates,
                    network,
                    optimizer,
                    world_size,
                    returns,
                    l1regparam,
                    leverage_constraint_type,
                    leverage,
                    early_stopping = [np.Inf,0],
                    test_loader=None,
                    train_loader = None,
                    val_loader = None,
                    val_criterion = 'sharpe_ratio', 
                    train_criterion = 'mispricing_loss',
                    test_assets = None,
                    weights = None,
                    verbose = 0):
        
        self.idxdates = idxdates
        self.network = network
        self.optimizer = optimizer
        self.world_size = world_size
        self.returns = returns
        self.test_loader = test_loader 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_criterion = val_criterion
        self.train_criterion = train_criterion
        self.early_stopping = early_stopping
        self.l1regparam = l1regparam
        self.leverage_constraint_type = leverage_constraint_type
        self.leverage = leverage
        self.test_assets = test_assets 
        self.weights = weights
        self.verbose = verbose
        # preparations for train_val_test
        self.other_data_loss_args = {}
        
    def preprocess_x(self,x):
        channels = list(torch.split(x, [1,1, x.shape[-1]-2], dim=1))
        for i in range(2):
            channels[i] = channels[i].flatten().to(torch.long)
            
        return channels
    
    def get_weights(self):
        return self.weights 
    
    def update_global_tensor(self, global_tensor, indices, new_values):
        if len(new_values.shape)==2 and len(global_tensor.shape)==1:
            new_values = new_values.flatten()
        global_tensor[indices] = new_values
    
    def get_returns(self, idx, date_id):
        return pm.calculate_pf_return(idx, date_id, self.weights, self.returns)[1]
    
    def get_final_returns(self):
        phases =[]
        final_returns= [np.concatenate((pm.calculate_pf_return(self.t_idx['train'], self.date_ids['train'], self.weights, self.returns)[0].flatten().detach().cpu().numpy().reshape(-1,1),
                self.get_returns(self.t_idx['train'], self.date_ids['train']).flatten().detach().cpu().numpy().reshape(-1,1)),axis = 1)]
        if self.val_loader is not None:
                phases.append('val')
        if self.test_loader is not None:
                phases.append('test')
        for phase in phases:
            final_returns.append(np.concatenate((pm.calculate_pf_return(self.t_idx[phase], self.date_ids[phase], self.weights, self.returns)[0].flatten().detach().cpu().numpy().reshape(-1,1),
                    self.get_returns(self.t_idx[phase], self.date_ids[phase]).flatten().detach().cpu().numpy().reshape(-1,1)),axis = 1))
        
        final_returns = pd.DataFrame(np.concatenate(final_returns, axis = 0), columns = ['date_id','pf_return'])
        if final_returns['date_id'].dtype != 'int64':
            final_returns['date_id'] = final_returns['date_id'].astype(int)
            
        return final_returns
    
    def get_sharpe_ratio(self,ret, phase):
        sr = self.sr(ret)
        if (sr<self.agnostic_sharpe_ratio[phase] or sr==np.Inf):
            print(f"Agnostic model does better in phase {phase} because Sharpe ratio={sr}!")
            sr = np.clip(sr,-np.abs(self.agnostic_sharpe_ratio[phase]),np.abs(self.agnostic_sharpe_ratio[phase]))
        return sr
        
    def prepare_trainvaltest(self):
        
        self.data_loss = {}
        self.sharpe_ratio = {}
        
        # best net performance during validation 
        self.best_sharpe_ratio = -np.Inf
        self.best_model_weights = None
        self.best_weights = None

        dataset_len = len(self.returns)
        
        self.initialize_loss()
        
        if self.weights is None:
            self.weights = torch.zeros(dataset_len).cuda(0)  
        if self.test_assets is None:
            self.test_assets = torch.ones(size = (dataset_len, 1)).cuda(0) 
        else:
            self.test_assets = self.returns[['t_index','date_id']].merge(self.test_assets, how = "inner", on = "date_id")
            self.test_assets = self.test_assets.iloc[:,2:].to_numpy().astype(np.float32)
            self.test_assets = torch.from_numpy(self.test_assets).flatten().cuda(0)  # shape (-1,nr_test_assets)
    
        # note: these are full-dataset vectors
        self.asset_ids = torch.from_numpy(self.returns.iloc[:,2].to_numpy().astype(np.float32)).to(torch.long).cuda(0)
        self.returns = self.returns.iloc[:,3].to_numpy().astype(np.float32)
        self.returns = torch.from_numpy(self.returns).flatten().cuda(0)
        
        # note, the data will come in a list of numpy format from InitDataloader
        fun = lambda x: torch.from_numpy(x).to(torch.long).flatten()
        
        self.t_idx, self.date_ids = {}, {}
        self.agnostic_sharpe_ratio = {}
        for i, phase in enumerate(['train','val','test']):
            self.t_idx[phase] = fun(self.idxdates[i][:,0])
            self.date_ids[phase] = fun(self.idxdates[i][:,1])
            self.agnostic_sharpe_ratio[phase] = self.sr(self.returns[self.t_idx[phase]])
            print(f"agnostic_sharpe_ratio[\"{phase}\"]: {self.agnostic_sharpe_ratio[phase]}")
        if self.early_stopping[0] < np.Inf: 
            print(f"Early stopping with patience {self.early_stopping[0]} and delta {self.early_stopping[1]} activated.")
            self.earlystop_counter = 0
            
    def initialize_loss(self): 
        
        assert (self.val_criterion == "sharpe_ratio")
        self.sr = getattr(pm, "sharpe_ratio")
        self.train_loss = getattr(pm,self.train_criterion)
    
    def train_one_ep(self,epoch):
    
        self.network.train()
            
        self.data_loss['train',epoch] = 0.0
        data_len = self.t_idx['train'].shape[0]
              
        for x, _ in self.train_loader:
            x = x.cuda(non_blocking = True)
            
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                channels = self.preprocess_x(x)
                new_weights = self.network.forward(channels[2])
                
                new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                
                data_loss = self.train_loss(idx = channels[0], date_id = channels[1], new_weights = new_weights, returns = self.returns, **self.other_data_loss_args, # this is risk_aversion in the case 
                                          new_test_assets = self.test_assets[channels[0]])
                loss = data_loss + l1_penalty(params = self.network.parameters(), lam = self.l1regparam)
                
                loss.backward()
                self.optimizer.step()

            self.data_loss['train',epoch] += data_loss.detach().cpu().numpy()*x.size(0)
            self.update_global_tensor(global_tensor = self.weights, indices = channels[0], 
                                      new_values = new_weights.detach())         
        self.data_loss['train', epoch] =  self.data_loss['train',epoch]/data_len
        self.sharpe_ratio['train',epoch] = self.get_sharpe_ratio(self.get_returns(self.t_idx['train'], 
                                                                self.date_ids['train']), phase = 'train')
        if self.verbose > 0 and epoch%int(1/self.verbose) == 0:
            print(f"\nFinished training epoch {epoch}")
            print(f"Data loss in training set: {self.data_loss['train', epoch]}")
            print(f"Sharpe ratio in training set: {self.sharpe_ratio['train', epoch]}\n")
        
    def validate_one_ep(self,epoch):
    
        self.network.eval()
            
        self.data_loss['val',epoch] = 0.0
        data_len = self.t_idx['val'].shape[0]
        
        for x, _ in self.val_loader:                  
            x = x.cuda(non_blocking = True) 
            
            with torch.set_grad_enabled(False):
                channels = self.preprocess_x(x)
                new_weights = self.network.forward(channels[2])
                new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                data_loss = self.train_loss(idx = channels[0], date_id = channels[1], new_weights = new_weights, 
                                        returns = self.returns, **self.other_data_loss_args, 
                                          new_test_assets = self.test_assets[channels[0]])
                
            self.data_loss['val',epoch] += data_loss.detach().cpu().numpy()*x.size(0)
            self.update_global_tensor(global_tensor = self.weights, indices = channels[0], 
                                      new_values = new_weights.detach())
               
        self.data_loss['val',epoch] = self.data_loss[('val',epoch)]/data_len
        self.sharpe_ratio['val', epoch] = self.get_sharpe_ratio(self.get_returns(self.t_idx['val'], self.date_ids['val']), phase = 'val')
        
        # early stopping routine
        self.stop_early = False
        if self.early_stopping[0] < np.Inf: 
            early_stopping_active = True
        else:
            early_stopping_active = False
            
        improvement = (self.sharpe_ratio['val', epoch] > self.best_sharpe_ratio + self.early_stopping[1]) and (self.data_loss['val',epoch] < 1000000) # don't save stupidly large losses, even with initialization
        
        if improvement:
            print(f'\nUpdating global best model weights based on validation of epoch {epoch}.\n')
            
            self.best_model_weights = copy.deepcopy(self.network.state_dict())# best weights for the models so far
            self.best_sharpe_ratio = self.sharpe_ratio[('val',epoch)]
            self.best_weights = self.weights
            self.earlystop_counter = 0
            
        elif early_stopping_active:
            self.earlystop_counter += 1
            if self.earlystop_counter>= self.early_stopping[0]:
                self.stop_early = True 
                print(f"\n----Early stopping in epoch {epoch} with parameters:\npatience: {self.early_stopping[0]}\ndelta: {self.early_stopping[1]}")
        print(f"\nFinished validation epoch {epoch}")
        
        if self.verbose > 0 and epoch%int(1/self.verbose) == 0:
            print(f"Data loss in validation set set: {self.data_loss['val', epoch]}")
            print(f"Sharpe ratio in validation set (current epoch): {self.sharpe_ratio['val', epoch]}\n")
            print(f"Best sharpe ratio so far in validation set: {self.best_sharpe_ratio}\n")
                        
    def testset_performance(self):
        
        if self.best_model_weights != None:    
           self.network.load_state_dict(self.best_model_weights)
        self.data_loss['test', 0] = 0.0
        
        self.network.eval()
        data_len = self.t_idx['test'].shape[0]
        for x, _ in self.test_loader:                  
            x = x.cuda(non_blocking = True) 
            
            with torch.set_grad_enabled(False):
                channels = self.preprocess_x(x)
                new_weights = self.network.forward(channels[2])
                new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                
                data_loss = self.train_loss(idx = channels[0], date_id = channels[1], new_weights = new_weights, returns = self.returns, 
                                            **self.other_data_loss_args, 
                                            new_test_assets = self.test_assets[channels[0]])
                
            self.data_loss['test',0] += data_loss.detach().cpu().numpy()*x.size(0)
            self.update_global_tensor(global_tensor = self.best_weights, indices = channels[0], 
                                      new_values = new_weights.detach())
               
        self.data_loss['test',0] = self.data_loss[('test',0)]/data_len
        self.sharpe_ratio['test',0] = self.get_sharpe_ratio(self.get_returns(self.t_idx['test'], self.date_ids['test']), phase = 'test')
  
        if self.verbose > 0:
            print("\nFinished testing.")
            print(f"SDF mispricing loss in test set: {self.data_loss['test',0]}")
            print(f"Sharpe ratio in test set: {self.sharpe_ratio['test', 0]}\n")
        
    def gather_performance(self):
        
        self.final_weights = self.weights
        self.final_returns = self.get_final_returns()
        self.sharpe_ratio['best_val',0] = self.best_sharpe_ratio
        df_dic = {'data_loss': self.data_loss,'sharpe_ratio': self.sharpe_ratio}
            
        self.performance_dict = performance_df(df_dic)    


class TrainValMPLoss(TrainValBase):
    
    def __init__(self,idxdates,
                    network,
                    optimizer,
                    world_size,
                    returns,
                    nr_features,
                    leverage_constraint_type,
                    leverage,
                    test_loader=None,
                    train_loader = None,
                    val_loader = None,
                    val_criterion = 'sharpe_ratio', 
                    train_criterion = 'mispricing_loss',
                    early_stopping = [np.Inf,0],
                    l1regparam = 0.0,
                    test_assets = None,
                    weights = None,
                    verbose = 0):
        
        super(TrainValMPLoss,self).__init__(idxdates=idxdates, network=network, optimizer=optimizer, world_size=world_size, returns = returns,
                                l1regparam=l1regparam, early_stopping=early_stopping, 
                                leverage_constraint_type = leverage_constraint_type, leverage = leverage,
                                test_loader = test_loader,
                                train_loader=train_loader, val_loader = val_loader, val_criterion = val_criterion,
                                train_criterion=train_criterion, weights = weights, verbose = verbose)
        
        # preparations for train_val_test
        self.prepare_trainvaltest()
        self.other_data_loss_args = {"moment_weighting":True, "loss_weight":1, "asset_ids":self.asset_ids}
            

class TrainValSR(TrainValBase):
    
    def __init__(self,
                    idxdates,
                    network,
                    optimizer,
                    world_size,
                    returns,
                    nr_features,
                    risk_aversion,
                    leverage_constraint_type,
                    leverage,
                    test_loader=None,
                    train_loader = None,
                    val_loader = None,
                    val_criterion = 'sharpe_ratio', 
                    train_criterion = 'sharpe_ratio',
                    early_stopping = [np.Inf,0],
                    l1regparam = 0.0,
                    weights = None,
                    verbose = 0):
        
        super(TrainValSR,self).__init__(idxdates=idxdates, network=network, optimizer=optimizer, world_size=world_size, 
                                returns = returns,test_assets=None,l1regparam=l1regparam, 
                                leverage_constraint_type = leverage_constraint_type, leverage = leverage,
                                early_stopping=early_stopping, test_loader = test_loader,
                                train_loader=train_loader, val_loader = val_loader, val_criterion = val_criterion,
                                train_criterion=train_criterion, weights = weights, verbose = verbose)
        
        self.risk_aversion = risk_aversion
        self.nr_features=nr_features
        self.other_data_loss_args = {"risk_aversion":self.risk_aversion}
        self.prepare_trainvaltest()
        
            
class TrainValGAN(TrainValBase):
    
    def __init__(self,idxdates,
                    network, # is a list with 2 elements in case of GAN
                    optimizer, # is a list with 2 elements in case of GAN
                    subepochs,
                    world_size,
                    returns,
                    leverage_constraint_type='L2',
                    leverage=1,
                    test_loader=None,
                    train_loader = None,
                    val_loader = None,
                    val_criterion = 'sharpe_ratio', 
                    train_criterion = 'mispricing_loss',
                    early_stopping = [np.Inf,0],
                    l1regparam = [0.0,0.0], 
                    nr_features = [6,4], # is now a list, with elements macro_states_dim, test_assets_dim
                    sdf_macro_states = None,
                    moments_macro_states = None,
                    test_assets = None,
                    weights = None,
                    verbose = 0,
                    **kwargs):
        
        super(TrainValGAN,self).__init__(idxdates=idxdates, network=network, optimizer=optimizer, world_size=world_size, returns = returns,
                                         l1regparam=l1regparam, early_stopping=early_stopping, 
                                         leverage_constraint_type = leverage_constraint_type, leverage = leverage,
                                         test_loader = test_loader, train_loader=train_loader, 
                                         val_loader = val_loader, val_criterion = val_criterion,
                                         train_criterion=train_criterion, weights = weights, verbose = verbose)
        
        self.subepochs = subepochs
        self.l1regparam_sdf = l1regparam[0]
        self.l1regparam_mom = l1regparam[1]
        self.unc_loss_weight = 1#unc_loss_weight
        self.sdf_macro_states = sdf_macro_states
        self.moments_macro_states = moments_macro_states
        self.macro_states_dim = nr_features[0]
        self.test_assets_dim = nr_features[1]
        self.sdf_macro_states = sdf_macro_states
        self.moments_macro_states = moments_macro_states
        self.test_assets = test_assets
        self.weights = weights
        self.verbose = verbose
                
        # preparations for train_val_test
        self.prepare_trainvaltest()
        
    def specific_train_loss_other_args(self): 
        self.other_data_loss_args = {"moment_weighting":True, "loss_weight":self.unc_loss_weight, "asset_ids":self.asset_ids}
        
    def prepare_trainvaltest(self):
        
        self.unconditional_sdf_loss = {}
        self.moments_loss = {}
        self.conditional_sdf_loss = {}
        self.sharpe_ratio = {}
        self.mp_val_test_loss = {}
        
        # best net performance during validation 
        self.best_sharpe_ratio = -np.Inf
        self.best_model_weights = None
        self.best_macro_states = None
        self.best_weights = None
        self.best_test_assets = None
        
        dataset_len = len(self.returns)
        if self.sdf_macro_states is None:
            self.sdf_macro_states = torch.zeros((dataset_len,self.macro_states_dim)).cuda()  

        if self.moments_macro_states is None:
            self.moments_macro_states = torch.zeros((dataset_len,self.macro_states_dim)).cuda()  
        if self.weights is None:
            self.weights = torch.zeros(dataset_len).cuda()  
        if self.test_assets is None:
            self.test_assets = torch.ones(size = (dataset_len, self.test_assets_dim)).cuda() 
            
        self.macro_states = [self.sdf_macro_states, self.moments_macro_states]
            
        self.asset_ids = torch.from_numpy(self.returns.iloc[:,2].to_numpy().astype(np.float32)).to(torch.long).cuda()
        self.returns = self.returns.iloc[:,3].to_numpy().astype(np.float32)
        self.returns = torch.from_numpy(self.returns).flatten().cuda()
        
        self.t_idx, self.date_ids = {}, {}
        self.agnostic_sharpe_ratio = {}
        
        self.specific_train_loss_other_args()
        self.initialize_loss()
        
        fun = lambda x: torch.from_numpy(x).to(torch.long).flatten()
        
        for i, phase in enumerate(['train','val','test']):
            self.t_idx[phase] = fun(self.idxdates[i][:,0])
            self.date_ids[phase] = fun(self.idxdates[i][:,1])
            self.agnostic_sharpe_ratio[phase] = self.sr(self.returns[self.t_idx[phase]])
            print(f"agnostic_sharpe_ratio[\"{phase}\"]: {self.agnostic_sharpe_ratio[phase]}")
        if self.early_stopping[0] < np.Inf: 
            print(f"Early stopping with patience {self.early_stopping[0]} and delta {self.early_stopping[1]} activated.")
            self.earlystop_counter = 0
        
    def train_one_ep(self,epoch):
    
        for network in self.network:
            network.train()
            
        self.unconditional_sdf_loss[('train',epoch)] = 0.0
        self.moments_loss[('train',epoch)] = 0.0
        self.conditional_sdf_loss[('train',epoch)] = 0.0
        data_len = self.t_idx['train'].shape[0]
        
        if (self.network[0].propagate_state or self.network[1].propagate_state):
            self.last_batch_date = torch.Tensor([-1]).cuda(non_blocking = True)
        
        for x, _ in self.train_loader:
            
            x = x.cuda(non_blocking = True)
            if (self.network[0].propagate_state or self.network[1].propagate_state):
                
                cont_date = (torch.abs(self.last_batch_date-x[0,1].reshape(1,))<0.25)
            else:
                cont_date = False
            '''train the sdf-network with unconditional loss for current batch'''
            unc_assets = torch.ones(x.size(0),self.test_assets_dim).cuda()
            print("Training the sdf network with unconditional loss\n")
            for _ in range(1,self.subepochs[0]+1):
                
                self.optimizer[0].zero_grad()
                with torch.set_grad_enabled(True):
                    # set macro state to current macro state for the RNN if propagate_state
                    if self.network[0].propagate_state:
                        self.network[0].macro_state = self.sdf_macro_states
                        self.network[0].cont_date = cont_date 
                    
                    idx, date_id, new_weights, new_sdf_macro_states = self.network[0].forward(x)
                    new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                    
                    data_loss = self.train_loss(idx = idx, date_id = date_id, new_weights = new_weights, returns = self.returns, 
                                                **self.other_data_loss_args, # this is self.asset_ids here
                                                new_test_assets = unc_assets)
                    
                    loss = data_loss + l1_penalty(params = self.network[0].parameters(), lam = self.l1regparam_sdf)
                    
                    loss.backward(retain_graph=True)
                    self.optimizer[0].step()

            self.unconditional_sdf_loss['train',epoch] += data_loss.detach().cpu().numpy()*x.size(0)
            
            '''train the moments network with conditional loss for current batch'''
            print("Training the moments network with conditional loss\n")
            for _ in range(1,self.subepochs[1]+1):
        
                self.optimizer[1].zero_grad()
                
                with torch.set_grad_enabled(True):
                    # set macro state to current macro state for the RNN if propagate_state across batch
                    if self.network[1].propagate_state:
                        self.network[1].macro_state = self.moments_macro_states
                        self.network[1].cont_date = cont_date
                        
                    idx, date_id, new_test_assets, new_moments_macro_states = self.network[1].forward(x)
                    data_loss = self.train_loss(idx = idx, date_id = date_id, new_weights = new_weights.detach().clone().requires_grad_(True), returns = self.returns, 
                                                **self.other_data_loss_args,
                                                new_test_assets = new_test_assets)
                    
                    loss = - data_loss + l1_penalty(params = self.network[1].parameters(), lam = self.l1regparam_mom)
                    loss.backward()
                    self.optimizer[1].step()
                
            self.moments_loss['train',epoch] -= data_loss.detach().cpu().numpy()*x.size(0)  # note the minus
            
            # update test_assets, and moments_macro_states. Note, we update the sdf macro states only in the end, because
            # the batch is fixed during the subepochs training
            self.update_global_tensor(global_tensor = self.moments_macro_states, indices = idx, 
                                          new_values = new_moments_macro_states.detach())
            self.update_global_tensor(global_tensor = self.test_assets, indices = idx, new_values = new_test_assets.detach())
                
            '''train the sdf network with conditional loss for the current batch'''
            print("Training the sdf network with conditional loss\n")
            for _ in range(1,self.subepochs[2]+1):
            
                self.optimizer[0].zero_grad()
            
                with torch.set_grad_enabled(True):
                    # set macro state to current macro state for the RNN if propagate_state
                    if self.network[0].propagate_state:
                        self.network[0].macro_state = self.sdf_macro_states
                        self.network[0].cont_date = cont_date
                    
                    idx, date_id, new_weights, new_sdf_macro_states = self.network[0].forward(x)
                    
                    new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                    data_loss = self.train_loss(idx = idx, date_id = date_id, new_weights = new_weights, returns = self.returns, 
                                                **self.other_data_loss_args, 
                                                new_test_assets = new_test_assets.detach().clone().requires_grad_(True))
                    
                    loss = data_loss + l1_penalty(params = self.network[0].parameters(), lam = self.l1regparam_sdf)
                    
                    loss.backward()
                    self.optimizer[0].step()
                    
            self.conditional_sdf_loss['train',epoch] += data_loss.detach().cpu().numpy()*x.size(0)
            
            # update weights, and sdf_macro_states. Note, we update the macro states only in the end, because
            # the batch is fixed during the subepochs training
            self.update_global_tensor(global_tensor = self.sdf_macro_states, indices = idx, 
                                      new_values = new_sdf_macro_states.detach())            
            # we update weights only at the end of the batch training, since don't need them before
            self.update_global_tensor(global_tensor = self.weights, indices = idx, new_values = new_weights.detach())
            if (self.network[0].propagate_state or self.network[1].propagate_state):
                self.last_batch_date = x.detach()[-1,1].reshape(1,)
            
        # update epoch losses for output purposes (note: these are averages)
        self.unconditional_sdf_loss['train', epoch] =  self.unconditional_sdf_loss['train',epoch]/data_len
        self.conditional_sdf_loss['train',  epoch] =  self.conditional_sdf_loss['train',epoch]/data_len
        self.moments_loss['train',epoch] =  self.moments_loss['train',  epoch]/data_len
        self.sharpe_ratio['train',epoch] = self.get_sharpe_ratio(self.get_returns(self.t_idx['train'], self.date_ids['train']), phase = 'train')
        
        if self.verbose > 0 and epoch%int(1/self.verbose) == 0:
            print(f"\nFinished training epoch {epoch}")
            print(f"Unc sdf mispricing loss in training set: {self.unconditional_sdf_loss['train', epoch]}")
            print(f"Cond sdf mispricing loss in training set: {self.conditional_sdf_loss['train', epoch]}")
            print(f"Moments mispricing loss in training set: {self.moments_loss['train',  epoch]}")
            print(f"Sharpe ratio in training set: {self.sharpe_ratio['train', epoch]}\n")
        
    def validate_one_ep(self,epoch):        
        for network in self.network:
            network.eval()
            
        self.mp_val_test_loss['val',epoch] = 0.0
        data_len = self.t_idx['val'].shape[0]
        
        for x, _ in self.val_loader:
            x = x.cuda(non_blocking = True) 
            if (self.network[0].propagate_state or self.network[1].propagate_state):
                cont_date = (torch.abs(self.last_batch_date-x[0,1].reshape(1,))<0.25)
            else:
                cont_date = False
            
            with torch.set_grad_enabled(False):
                # set macro state to current macro state for the RNN if propagate_state
                if self.network[0].propagate_state:
                    self.network[0].macro_state = self.sdf_macro_states
                    self.network[0].cont_date = cont_date
                if self.network[1].propagate_state:
                    self.network[1].macro_state = self.moments_macro_states
                    self.network[1].cont_date = cont_date
                
                idx, date_id, new_weights, new_sdf_macro_states = self.network[0].forward(x)
                
                new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                
                _, _, new_test_assets, new_moments_macro_states = self.network[1].forward(x)
                
                data_loss = self.train_loss(idx = idx, date_id = date_id, new_weights = new_weights.detach(), 
                                            returns = self.returns, 
                                            **self.other_data_loss_args, 
                                            new_test_assets = new_test_assets.detach())
                
            self.mp_val_test_loss['val',epoch] += data_loss.detach().cpu().numpy()*x.size(0)
            
            # update macro states, test assets and weights
            self.update_global_tensor(global_tensor = self.sdf_macro_states, indices = idx, 
                                      new_values = new_sdf_macro_states.detach())
            self.update_global_tensor(global_tensor = self.moments_macro_states, indices = idx, 
                                      new_values = new_moments_macro_states.detach())
            self.update_global_tensor(global_tensor = self.weights, indices = idx, 
                                      new_values = new_weights.detach())
            self.update_global_tensor(global_tensor = self.test_assets, indices = idx, 
                                      new_values = new_test_assets.detach())
            if (self.network[0].propagate_state or self.network[1].propagate_state):
                self.last_batch_date = x.detach()[-1,1].reshape(1,)
               
        self.mp_val_test_loss['val',epoch] = self.mp_val_test_loss[('val',epoch)]/data_len
        self.sharpe_ratio['val',epoch] = self.get_sharpe_ratio(self.get_returns(self.t_idx['val'], self.date_ids['val']), phase = 'val')
        
        # early stopping routine
        self.stop_early = False
        if self.early_stopping[0] < np.Inf: 
            early_stopping_active = True
        else:
            early_stopping_active = False
            
        improvement = (self.sharpe_ratio['val', epoch] > self.best_sharpe_ratio + self.early_stopping[1]) and (self.mp_val_test_loss['val', epoch] < 1000000) # don't save stupidly large losses, even with initialization
        
        if improvement:
            print(f'\nUpdating global best model weights based on validation of epoch {epoch}.\n')
            
            self.best_model_weights = [copy.deepcopy(model.state_dict()) for model in self.network] # best weights for the models so far
            self.best_sharpe_ratio = self.sharpe_ratio[('val',epoch)]
            self.best_macro_states = [self.sdf_macro_states, self.moments_macro_states]
            self.best_test_assets = self.test_assets
            self.best_weights = self.weights
            self.earlystop_counter = 0
            
        elif early_stopping_active:
            self.earlystop_counter += 1
            if self.earlystop_counter>= self.early_stopping[0]:
                self.stop_early = True 
                print(f"\n----Early stopping in epoch {epoch} with parameters:\npatience: {self.early_stopping[0]}\ndelta: {self.early_stopping[1]}")
                # there will be a break in the one_shot_gan file, otherwise it will be passed to other gpus. 
        print(f"\nFinished validation epoch {epoch}")
        
        if self.verbose > 0 and epoch%int(1/self.verbose) == 0:
            print(f"Cond Mispricing loss in validation set: {self.mp_val_test_loss['val',epoch]}")
            print(f"Sharpe ratio in validation set (current epoch): {self.sharpe_ratio['val', epoch]}")
            print(f"Best sharpe ratio so far in validation set: {self.best_sharpe_ratio}\n")
                        
    def testset_performance(self):
        # we test in one batch only
        
        if self.best_model_weights != None: 
            for i in [0,1]:
                self.network[i].load_state_dict(self.best_model_weights[i])
        self.mp_val_test_loss['test',0] = 0.0
        
        for network in self.network:
            network.eval()
        data_len = self.t_idx['test'].shape[0]
                
        for x, _ in self.test_loader:
        
            x=x.cuda(non_blocking=True) 
            if (self.network[0].propagate_state or self.network[1].propagate_state):
                cont_date = (torch.abs(self.last_batch_date-x[0,1].reshape(1,))<0.25)
            else:
                cont_date = False
                
            with torch.set_grad_enabled(False):
                # set macro state to current macro state for the RNN if propagate_state
                if self.network[0].propagate_state:
                    self.network[0].macro_state = self.sdf_macro_states
                    self.network[0].cont_date = cont_date
                if self.network[1].propagate_state:
                    self.network[1].macro_state = self.moments_macro_states
                    self.network[1].cont_date = cont_date
                    
                idx, date_id, new_weights, new_sdf_macro_states = self.network[0].forward(x)
                
                new_weights = normalize_weights(new_weights,constraint_type=self.leverage_constraint_type, L = self.leverage)
                
                _, _, new_test_assets, new_moments_macro_states = self.network[1].forward(x)
                
                data_loss = self.train_loss(idx = idx, date_id = date_id, new_weights = new_weights.detach(), returns = self.returns, 
                                            **self.other_data_loss_args, 
                                            new_test_assets = new_test_assets.detach())
                
            self.mp_val_test_loss['test', 0] += data_loss.detach().cpu().numpy()*x.size(0) # no mean reduction since it's one batch
            
            if self.val_loader is not None and self.best_sharpe_ratio > -np.Inf:
                self.update_global_tensor(global_tensor = self.best_macro_states[0], indices = idx, 
                                      new_values = new_sdf_macro_states.detach())
                self.update_global_tensor(global_tensor = self.best_macro_states[1], indices = idx, 
                                      new_values = new_moments_macro_states.detach())
                self.update_global_tensor(global_tensor = self.best_weights, indices = idx, 
                                          new_values = new_weights.detach())
                self.update_global_tensor(global_tensor = self.best_test_assets, indices = idx, 
                                          new_values = new_test_assets.detach())
            else:
                self.update_global_tensor(global_tensor = self.sdf_macro_states, indices = idx, 
                                      new_values = new_sdf_macro_states.detach())
                self.update_global_tensor(global_tensor = self.moments_macro_states, indices = idx, 
                                      new_values = new_moments_macro_states.detach())
                self.update_global_tensor(global_tensor = self.weights, indices = idx, 
                                          new_values = new_weights.detach())
                self.update_global_tensor(global_tensor = self.test_assets, indices = idx, 
                                          new_values = new_test_assets.detach())
                
            if (self.network[0].propagate_state or self.network[1].propagate_state):
                self.last_batch_date = x.detach()[-1,1].reshape(1,)
            
        self.mp_val_test_loss[('test', 0)] = self.mp_val_test_loss[('test', 0)]/data_len
        self.sharpe_ratio['test', 0] = self.get_sharpe_ratio(self.get_returns(self.t_idx['test'], self.date_ids['test']), phase = 'test')
        
        if self.verbose > 0:
            print("\nFinished testing.")
            print(f"Cond sdf mispricing loss in test set: {self.mp_val_test_loss['test',0]}")
            print(f"Sharpe ratio in test set: {self.sharpe_ratio['test', 0]}\n")
                    
    def consolidate_global_vars(self):
        '''prepares final global vars for output
        Note: this step needs to happen after the finished training and evaluation steps like validation and testing.
        Nothing with gradients should happen after this.'''
        
        # test assets and weights
        self.list_test_assets = [torch.zeros_like(self.test_assets) for _ in range(self.world_size)]
        self.list_weights = [torch.zeros_like(self.weights) for _ in range(self.world_size)]
        
        if self.val_loader is not None and self.best_sharpe_ratio>-np.Inf: 
            self.test_assets = self.best_test_assets
            self.weights = self.best_weights
            self.sdf_macro_states, self.moments_macro_states = self.best_macro_states 

        # produce final_test_assets and final_weights
        self.consolidate_sdf_weights_test_assets(test_assets = self.test_assets, 
                                                 weights = self.weights)
        # produce final_macro_states
        self.final_macro_states = torch.cat((self.sdf_macro_states, self.moments_macro_states), dim=1)
        self.final_returns = self.get_final_returns()
            
    
    def gather_performance(self):
       
        self.final_test_assets = self.test_assets
        self.final_weights = self.weights
        self.final_macro_states = torch.cat((self.sdf_macro_states, self.moments_macro_states), dim=1)
        self.final_returns = self.get_final_returns()
        self.sharpe_ratio['best_val', 0] = self.best_sharpe_ratio
        df_dic = {'unc_sdf_train_data_loss': self.unconditional_sdf_loss, 'cond_sdf_train_data_loss': self.conditional_sdf_loss,
                  'mom_train_data_loss': self.moments_loss, 'sharpe_ratio': self.sharpe_ratio}
        
        self.performance_dict = performance_df(df_dic)


        if self.val_loader is not None:
            df_dic = {'val_test_data_loss':self.mp_val_test_loss}
            self.performance_dict.update(performance_df(df_dic))

        
