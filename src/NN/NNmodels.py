#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:57:55 2021

@author: jed-pitt

FFN standard, with embedded dropout, batch-norm layers, and multigpu support

"""

import torch.nn as nn


class FFNStandardTorch(nn.Module):
    
    '''
    Creates the architecture of a Standard, fully connected FFN
    As an intermediate step it creates an ordered dict of model type, which is fed to the forward method of nn.
    
    Args:
        num_features (int): number of features for every observation, needs to be the same for 
        every observation
                
        hidden_units (list): number of units for each consecutive hidden layer; it has length equal to the number
        of hidden layers
        
        output_size (int): the dimension of the output of the network
        
        activation (str): activation used for all layers, default = 'SiLU'
        
        batchnormalization (bool): if True, adds a layer of Batchnormalization (BN) right after
        every activation. Note: even though the original paper about BN had BN before activation, since then,
        people have been mostly using it right after the activation.
        
        dropout (float): if not None, adds a dropout layer right after the last activation
        /right before the output layer, default = None
        
    Functions:
        create: returns the network, created from passing the argument; first creates an 
        ordered dict that is then passed to the forward method
        
        forward: does the forward pass 
    
        get_architecture: prints the architecture of the model on the console
        
    Note: Elastic-Net can be modeled with this if we do one layer without biases
        
    '''
    
    def __init__(self,num_features, 
                 hidden_units,
                 output_size,
                 activation = 'SiLU', 
                 bias = True,
                 batchnormalization = False, 
                 dropout = None,
                 world_size = 1):
        
        super(FFNStandardTorch, self).__init__()

        self.num_features = num_features
        self.hidden_units = hidden_units  #list which will contain the number of hidden units for each layer 
        self.output_size = output_size
        self.activation = getattr(nn, activation)
        self.bias = bias
        self.batchnormalization = batchnormalization
        self.dropout = dropout
        self.world_size = world_size
        
        self.od = nn.ModuleDict()
        self.od['input layer'] = nn.Linear(self.num_features,self.hidden_units[0], bias = self.bias)
        self.od[f'activation_{1}'] = self.activation()
        if self.batchnormalization:
                self.od[f'batchnorm_{1}'] = nn.BatchNorm1d(self.hidden_units[0])
        
        for i in range(1,len(self.hidden_units)):
            self.od[f'linear_{i}'] = nn.Linear(self.hidden_units[i-1],self.hidden_units[i], bias = self.bias)
            self.od[f'activation_{i+1}'] = self.activation()
            # optional dropout layer   
            if self.dropout is not None:
                self.od['dropout_{i+1}'] = nn.Dropout(self.dropout)
            # optional batchnorm layer
            if self.batchnormalization:
                if self.world_size >1:
                    self.od[f'batchnorm_{i+1}'] = nn.SyncBatchNorm(hidden_units[i])
                else:
                    self.od[f'batchnorm_{i+1}'] = nn.BatchNorm1d(hidden_units[i])
        # add final dense, linear layer aggregating the outcomes so far
        self.od['output layer'] = nn.Linear(self.hidden_units[-1],self.output_size)       
        
    def forward(self, x):
        
        x = x.view(-1, self.num_features)
        
        for step, layer in self.od.items():
           # switch off batch normalization when batch size = 1
           if x.shape[0] == 1 and (isinstance(layer, nn.SyncBatchNorm) or isinstance(layer, nn.BatchNorm1d)):
               continue
           x = layer(x)
        
        return x
    
    def get_architecture(self):
                
        print(nn.Sequential(self.od))
        
