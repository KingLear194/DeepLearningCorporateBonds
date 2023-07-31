#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:57:55 2021

@author: jed-pitt


Composing Parts for GAN models

"""

import torch
import torch.nn as nn
from collections import OrderedDict

        
class FFNMergeGAN(nn.Module):
    
    '''
    Creates the architecture of a FFN where input data are first split into channels, processed through 
    FFNs and then the results are merged through another FFN. To realize this, the code for the creation of a fully connected 
    FFN is given in a separate function called block which produces a dict of modules
    
    Args:
        num_features_pre_merge (list of int): number of features that each channel will take, needs to be the same for 
        every observation
        
        hidden_units_pre_merge (list of lists of int): for each channel (len of parameter)
        it gives a list which contains the number of units for each consecutive hidden layer of the channel
        
        pre_merge_output_size (list of int): number of dimensions in the output layer of the channel. This indicates
        effectively the dimensionality reduction of the input of each channel
        
        hidden_units_at_merge (list of int or None): if not None, gives a list which contains the number of units 
        for each consecutive hidden layer of the FFN which will do the merging of the channel outputs
        If None and len(hidden_units_pre_merge) == 2, then takes dot product 
        
        final_output_size (int): size of final output of the merge, default = 1
        
        activation (str): activation used for all layers, default = 'SiLU'
        Note: we use uniform activation throughout the network
        
        batchnormalization (bool): if True, adds a layer of Batchnormalization (BN) right after
        every activation. 
        
        dropout (float): if not None, adds a dropout layer right after the last activation
        /right before the output layer, default = None
        
    Note: this is a modified version for the SDF network and hence contains network_type initialization variable with
    default = 'SDF'. In case default, returns output as is; otherwise applies nn.tanh layer to it. Note: we might change this later, since 
    if the noise is not tamed much by the network, it will lead to very flat moments over a batch. Alternative is a shifted logistic function.  
        
    Functions:
        block: given num_features, hidden_units, output_size of a FFN, it creates a standard FFN network
        
        forward: does the forward pass 
        Note: to evaluate an input here use model.forward(x); 
    
        get_architecture: prints the architecture of the model on the console
        
    '''
    
    def __init__(self,num_features_pre_merge, 
                 hidden_units_pre_merge,
                 pre_merge_output_size,
                 hidden_units_at_merge = None,
                 final_output_size = 1, # we want a portfolio weight, or a moment condition
                 network_type = 'SDF',
                 activation = 'SiLU', 
                 batchnormalization = False, 
                 dropout = None,
                 world_size=1):
        
        super().__init__()
        
        self.num_features_pre_merge = num_features_pre_merge # this is a 1-D list containing the number of features [macro_states_dim, indiv_features]
        # per channel
        self.hidden_units_pre_merge = hidden_units_pre_merge  #this is a list of lists
        # of the form [[hidden units of first channel],[hidden units of second channel] and so on...]
        self.final_output_size = final_output_size # 1 for SDF and M for 'Moments'
        self.network_type = network_type
        self.pre_merge_output_size = pre_merge_output_size # this is a list showing how strong the 
        # reduction of dimensionality will be for every channel
        self.hidden_units_at_merge = hidden_units_at_merge  #this is a list of ints if explicit merging,
        self.activation = getattr(nn, activation)
        self.batchnormalization = batchnormalization
        self.dropout = dropout
        self.world_size = world_size
        
        def block(num_features, hidden_units, output_size):
            # takes (int, list of ints) just as done in the create() method of the FFNStandardTorch
            od = nn.ModuleDict()
            od['input layer'] = nn.Linear(num_features,hidden_units[0])
            od[f'activation_{1}'] = self.activation()
                    
            for i in range(1,len(hidden_units)):
                od[f'linear_{i}'] = nn.Linear(hidden_units[i-1],hidden_units[i])
                od[f'activation_{i+1}'] = self.activation()
                if self.dropout is not None: 
                    od['dropout'] = nn.Dropout(self.dropout)
                if self.batchnormalization:
                    if self.world_size >1:
                        od[f'batchnorm_{i+1}'] = nn.SyncBatchNorm(hidden_units[i])
                    else:
                        od[f'batchnorm_{i+1}'] = nn.BatchNorm1d(hidden_units[i])
            # add final dense, linear layer aggregating the outcomes so far
            od['output layer'] = nn.Linear(hidden_units[-1],output_size)       
            
            return od

        self.networks_to_merge = nn.ModuleDict()
        for i in range(len(self.hidden_units_pre_merge)):
            self.networks_to_merge[f'channel{i}'] = block(self.num_features_pre_merge[i],self.hidden_units_pre_merge[i],self.pre_merge_output_size[i])
            
        self.num_tomerge_outputs = sum(self.pre_merge_output_size) 
        # add final dense, FFN aggregating the outcomes so far if hidden_units_at_merge is not None, 
        # otherwise do scalar product
        if self.hidden_units_at_merge is not None:
            self.merge_nn = block(self.num_tomerge_outputs, self.hidden_units_at_merge, self.final_output_size)
        elif (len(self.hidden_units_pre_merge) == 2) and (self.pre_merge_output_size[0] == self.pre_merge_output_size[1]):
            self.merge_nn = 'dot_product'
        else:
            raise ValueError('If no merging, then only dot product of results from two NN paths possible')
            
    def forward(self, x):
        split_x = torch.split(x, self.num_features_pre_merge, 1)
        pre_merge_outputs = OrderedDict() 
        for i in range(len(self.hidden_units_pre_merge)):
            
            fun = self.networks_to_merge[f'channel{i}']
            x = split_x[i]
            for step, layer in fun.items():
                # batchnormalization does not work when size of batch = 1
                if x.shape[0] == 1 and (isinstance(layer, nn.SyncBatchNorm) or isinstance(layer, nn.BatchNorm1d)):
                    continue
                x = layer(x)
            pre_merge_outputs[f'channel{i}'] = x

        if isinstance(self.merge_nn, nn.ModuleDict):
            # merge the pre_merge_outputs into one tensor
            x = torch.cat(list(pre_merge_outputs.values()),1)
            for step, layer in self.merge_nn.items():
                x = layer(x)
        elif self.merge_nn == 'dot_product':
            lis = list(pre_merge_outputs.values())
            x = (lis[0]*lis[1]).sum(1)
            x = torch.reshape(x, (-1,1))
        result = x if self.network_type == 'SDF' else torch.tanh(x)
        
        return result
    
    def get_architecture(self):
        print('Networks for the channels\n', self.networks_to_merge, '\n\nMerger network\n', self.merge_nn)
        

class RNNGAN(nn.Module):
    
    '''
    Recurrent GRU cell.
    
    Allows for intialization through an 
    outside state, for state propagation after each call of the forward method,
    allows for bidirectional RNN.
    
    Initialization:
        - num_features (int): nr of dimensions of each element of the batch
        - seq_len (int): seq_len of the past history
        - hidden_dim (int): the dimension of hidden state 
        - num_layers (int): the number of hidden layers (stacked RNN)
        - outside_state (torch.Tensor or tuple of torch.Tensors): initializing state for the model, 
        for the SDF estimation this will be a full macro vector
            default = None (initializes with zero tensors). If this is not None then we are propagating state
        - dropout (float): dropout rate for both encoder and decoder, default = None
    
    On Output: 
        returns date_id markers, and the corresponding states calculated from the batch
    
    Other parameters:
        - self.current_state: initialized through method init_state, gets propagated after every forward call
        - self.current_batch_states: final states after forward for the whole batch 
          (used for case where need both output and states for whole batch)
    
    Note: here we assume that whenever input changes within a batch, running state needs to be updated, otherwise not updated.
          This is not the default behavior from pytorch. For fin. time series, the current code covers both case where batch comes from
          a dataframe with index (time, asset) as well as only time. Only estimation issue would arise if for two consecutive periods 
          all time series have exactly the same values. 
    
    Note: - To get final states of a batch: load model, do forward on the batch and call 
            .current_batch_states
    
    '''

    def __init__(self,
                 num_features,
                 seq_len,
                 hidden_dim,
                 num_layers,
                 outside_state = None,
                 dropout = None,
                 rank=0):
        
        super().__init__()
        self.num_features = num_features  
        self.seq_len = seq_len 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.outside_state = outside_state
        self.dropout = dropout if dropout is not None else 0
        self.rank = rank
        
        self.recurrent = nn.GRU(
                input_size = self.num_features,
                hidden_size = self.hidden_dim,
                num_layers = self.num_layers,
                batch_first=True,
                bidirectional=False,
                dropout = self.dropout)
        
    def init_state(self, 
                   previous_last_idx = -1, outside = None):
        
        if (outside is not None and previous_last_idx>-1):
            self.previous_state = outside.detach().clone()[previous_last_idx] 
            self.previous_state = torch.cat(tuple(self.previous_state.reshape(-1,self.hidden_dim) for _ in range(self.num_layers)), dim=0).unsqueeze(1)
        else:
            h_0 = torch.zeros(self.num_layers, 1, self.hidden_dim)
            self.previous_state = h_0.detach()
        self.previous_state = self.previous_state.cuda(self.rank, non_blocking=True)
                
    def forward(self, cont_date, previous_last_idx, x):
        N = x[0].shape[0]
        self.init_state(previous_last_idx, self.outside_state) 
        
        if x[1].shape != (N, self.seq_len, self.num_features):
            x[1] = x[1].view((N, self.seq_len, self.num_features))
    
        self.running_states = torch.zeros(size=(N, self.hidden_dim)).cuda(self.rank)
        # process first element of batch
        # not comparing here in the second and with the date index!!
        if (previous_last_idx >-1) and self.outside_state is not None and cont_date:
            self.running_states[0] = self.previous_state[-1,-1,:].squeeze()
        else:
            out, running_state = self.recurrent(torch.unsqueeze(x[1][0],0), self.previous_state)
            self.running_states[0] = running_state[-1,-1,:].squeeze() 
            self.previous_state = running_state
        if N>1:
            for i, input_t in enumerate(x[1][1:,:].chunk(chunks = N-1, dim = 0),1):
                if x[0][i] == x[0][i-1]: 
                    self.running_states[i] = self.previous_state[-1,-1,:].squeeze()
                else: 
                    if x[0][i]-x[0][i-1] !=1: raise RuntimeError("x[0] in the RNN is not as it's supposed to be!")
                    out, running_state = self.recurrent(input_t, self.previous_state)
                    self.running_states[i]= running_state[-1,-1,:].squeeze()
                    self.previous_state = running_state
        return x[0],self.running_states  

class GANNet(nn.Module):
    
    def __init__(self,
                 ffnmerge_params, # except for network_type
                 rnn_params, # except outside_state
                 nr_indiv_features,
                 macro_input_length,
                 macro_states = None, # full macro tensor
                 propagate_state = False, # propagates state across batch
                 network_type = 'SDF', 
                 rank=0,
                 world_size=1):
        
        super().__init__()
        self.ffnmerge_params = ffnmerge_params
        self.rnn_params = rnn_params
        self.nr_indiv_features = nr_indiv_features
        self.macro_input_length = macro_input_length
        self.propagate_state = propagate_state
        self.rank = rank
        self.network_type = network_type
    
        self.macro_states = macro_states
        self.recurrent = RNNGAN(**self.rnn_params, outside_state=self.macro_states, rank=self.rank) # initialize the macro_state from outside
        self.ffnmerge = FFNMergeGAN(**self.ffnmerge_params, network_type=self.network_type, world_size=world_size)
        
        self.cont_date = False # note: this is a feature of the batch that will be passed by trainer-class.
        self.previous_last_idx = -1
        
    @property
    def macro_state(self):
        return self.macro_states
    
    @macro_state.setter 
    def macro_state(self, macro_st=None):
        if macro_st is not None:
            self.macro_states = macro_st.detach().clone() 
        
    def forward(self, x):
        # x is a batch of full features: t_idx, date_id, individual_features, macro_features
        # create four channels: first for t_idx, second for date_id, third for individidual features
        # and fourth for the macro features
        channels = list(torch.split(x, [1,1, self.nr_indiv_features, self.macro_input_length], dim=1))
        # make idx_tensors out of the first two ones
        for i in range(2):
            channels[i] = channels[i].flatten().to(torch.long)
        # update (the full) macro_state if the setter method has been used in the training to reset
        # the macro state. I.e. this requires a step of GANNet.macro_states = new_macro_states in the main body
        if self.propagate_state:
            self.recurrent.outside_state = self.macro_states
            self.previous_last_idx = channels[0][0]-1
        _ , new_macro_states = self.recurrent.forward(self.cont_date,
                                                      self.previous_last_idx,
                                                      [channels[1],channels[3]]) # pass date_ids and macro_features
        # note that channels[1] will have the repetitions according to the batch, i.e. will have size N
        # feed macro-states and individual features to ffnmerge without date_id or t_index
        ffnmerge_input = torch.cat((new_macro_states.detach(),channels[2]), dim=1)
        
        # this is sdf_weights if network_type=SDF, else, it's the test_assets
        ffn_output = self.ffnmerge.forward(ffnmerge_input)
        # return  t_index, date_id and sdf_weights/test_assets of current batch, together with updated macro_state
        return channels[0], channels[1], ffn_output, new_macro_states
        
    