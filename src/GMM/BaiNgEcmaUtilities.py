#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:22:39 2021

@author: jed-pitt
"""

import pandas as pd
import numpy as np
from python_nw import newey
import math
from pcaversions import pcask


""" 
This file has functions that calculate the PC and IC criteria of the Bai, Ng 2002 Econometrica paper
and also has functions for the calculation of the optimal number of factors

Data input: timeseries in DataFrame format (columns)

Variables in the main functions below:

df - dataframe with the time series in the columns 
m - number of factors given by user
kmax - upper bound on the number of factors given by user, based on prior knowledge of user
mmax - upper bound on the number of factors given by user, for optimization purposes, can be lower than kmax
Example: user knows that there can't be more than 50 latent factors, but wants to see loss for 

lam - positive number scaling factor for the penalty on the number of factors
Note: Theorem 2 and Corollary 1 in paper easily show that every PC, IC criterion with rescaled penalty 
is also a consistent estimator of the correct number of factors

pic - variable for the optimization functions: if True a graph of the costs as a function of the number of components 
is produced

"""


def V(df, m):
    
    T, N = df.shape
    F = pcask(df,m)[0]
    v = 0
    for i in range(N):
        results = newey(df.iloc[:,i],F,0)
        residu = results.resid
        v += sum(residu**2)
        
    final = 1/(T*N)*v
    
    return final

############ Traditional BIC 
def bic(df,m,lam):
    
    T, N = df.shape
    
    penalty = np.log(T)/T
    loss = V(df,m)
    criterion = np.log(loss) + lam*m*penalty
    
    return criterion

############# Traditional AIC 
def aic(df,m,lam):
    
    T, N = df.shape
    
    penalty = 2/T
    loss = V(df,m)
    criterion = np.log(loss) + lam*m*penalty
    
    return criterion

'''
############ IC criteria in page 201 of Bai, Ng paper
'''

def icbaing_1(df,m,lam):
    
    T, N = df.shape
    penalty = ((N+T)/(N*T))*np.log((N*T)/(N+T))
    loss = V(df,m)
    criterion = np.log(loss) + m*penalty*lam
    
    return criterion


def icbaing_2(df,m,lam):
    
    T, N = df.shape
    cs = min(N,T)
    penalty = ((N+T)/(N*T))*np.log(cs)
    loss = V(df,m)
    
    criterion = np.log(loss) + m*penalty*lam
    
    return criterion


def icbaing_3(df,m,lam):
    
    T, N = df.shape
    cs = min(N,T)

    penalty = np.log(cs)/cs
    loss = V(df,m)
    criterion = np.log(loss) + m*penalty*lam
    
    return criterion


'''
### The PC criteria from page 201 in paper. 
They require giving a kmax as an additional parameter, where you know a priori that number of factors <= kmax. 
Then you can estimate sigma-hat-squared in the formulas in pg. 201 with loss of kmax.
'''

def pcbaing_1(df,m,kmax):
    
    T, N = df.shape
    penalty = ((N+T)/(N*T))*np.log((N*T)/(N+T))
    loss = V(df,m)
    sigma2 = V(df,kmax)
    criterion = loss + m*sigma2*penalty
    
    return criterion


def pcbaing_2(df,m,kmax):
    
    T, N = df.shape
    cs = min(N,T)
    penalty = ((N+T)/(N*T))*np.log(cs)
    loss = V(df,m)
    sigma2 = V(df,kmax)

    criterion = loss + m*sigma2*penalty
    
    return criterion


def pcbaing_3(df,m,kmax):
    
    T, N = df.shape
    cs = min(N,T)

    penalty = np.log(cs)/cs
    loss = V(df,m)
    sigma2 = V(df,kmax)
    criterion = loss + m*sigma2*penalty
    
    return criterion

'''
########### Versions of PC criteria with scaling of penalty
### Note: these are the ones used by the optimization functions
'''

def pcbaing_1_scaling(df,m,kmax,lam):
    
    T, N = df.shape
    
    penalty = ((N+T)/(N*T))*np.log((N*T)/(N+T))
    loss = V(df,m)
    sigma2 = V(df,kmax)
    criterion = loss + m*sigma2*penalty*lam
    
    return criterion


def pcbaing_2_scaling(df,m,kmax,lam):
    
    T, N = df.shape
    cs = min(N,T)
    
    penalty = ((N+T)/(N*T))*np.log(cs)
    loss = V(df,m)
    sigma2 = V(df,kmax)

    criterion = loss + m*sigma2*penalty*lam
    
    return criterion


def pcbaing_3_scaling(df,m,kmax,lam):
    
    T, N = df.shape
    cs = min(N,T)

    penalty = np.log(cs)/cs
    loss = V(df,m)
    sigma2 = V(df,kmax)
    criterion = loss + m*sigma2*penalty*lam
    
    return criterion

'''
#################### Functions to calculate the optimal number of factors

######## Minimize function for the IC criteria, page 201 of paper
'''

def minimizeic(df,mmax,ic,lam, pic=False):
    
    print(f"Name of ic: {ic.__name__}, lambda: {lam}, mmax: {mmax}")
    
    vals = []
    data = {}
    for i in range(1,mmax+1):
        w = ic(df,i,lam)
        vals.append(w)
        data[i] = w
        
    minval = min(vals)
    best = [j for j, v in enumerate(vals,1) if v == minval]
    
    if pic:
        df = pd.DataFrame.from_dict(data, orient='index', columns = ['Criterion'])
        ax = df.plot(title=f"IC Criterion from Bai, Ng 2002 with penalty scaling = {lam}")
        ax.set(xlabel="Number of components", ylabel="Criterion")
        
    return best

'''
########## Minimize function for the PC criteria, page 201 of paper
'''

def minimizepc(df,kmax,ic,lam,pic = False):
    
    print(f"Name of ic: {ic.__name__}, lambda: {lam}, kmax: {kmax}")
    
    vals = []
    data = {}
    for i in range(1,kmax+1):
        w = ic(df,i,kmax,lam)
        vals.append(w)
        data[i] = w
        
    minval = min(vals)
    best = [j for j, v in enumerate(vals,1) if v == minval]
    
    if pic:
        df = pd.DataFrame.from_dict(data, orient='index', columns = ['Criterion'])
        ax = df.plot(title=f"PC Criterion with kmax = {kmax} and penalty scaling {lam}")
        ax.set(xlabel="Number of components", ylabel="Criterion")
    
    
    return best

'''
######## IC print function for the optimization of an IC
######## Finds minimizers as a function of penalty lam and prints out the minimizer
######## lo, hi are the boundaries for zooming in for the penalty lam, step is the step of the calculation
######## it stops when 1 component becomes optimal
'''


def icprint(df,ic,mmax,lo,hi,step = None,num = None):
    
    # calculate range given lo, hi and step
    if not num and not step:
        raise ValueError('Either step or number of steps should be given.')
    elif not step and num:
        step = (hi-lo)/num
    else:
        num = np.ceil((hi-lo)/step).astype(int)
        
    dist = math.ceil(abs(math.log10(abs(step))))+2  
    arr = np.linspace(lo,hi,num)

    for lam in arr:
        indices = minimizeic(df,mmax,ic,lam, pic=False)
        print("With penalty weight", f"{round(lam,dist)}", "the minimizers are", indices)
        # if get 1 component as optimal, stop the calculation
        if 1 in indices: 
            print('1 component became optimal')
            return
        
'''
version of icprint utility for pc criteria
'''

def pcprint(df,pc,kmax,lo,hi,step = None,num = None):
    
    # calculate range given lo, hi and step
    if not num and not step:
        raise ValueError('Either step or number of steps should be given.')
    elif not step and num:
        step = (hi-lo)/num
    else:
        num = np.ceil((hi-lo)/step).astype(int)
        
    dist = math.ceil(abs(math.log10(abs(step))))+2  # 2 for tolerance in calculations given randomness of float
    arr = np.linspace(lo,hi,num)

    for lam in arr:
        indices = minimizepc(df,kmax,pc,lam,pic = False)
        print("With penalty weight", f"{round(lam,dist)}", "the minimizers are", indices)
        # if get 1 component as optimal, stop the calculation
        if 1 in indices: 
            print('1 component became optimal')
            return        
      
