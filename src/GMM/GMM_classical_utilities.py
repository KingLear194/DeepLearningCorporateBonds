#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:01:25 2021

@author: jed-pitt
"""

import numpy as np
import requests

from scipy.optimize import minimize
from functools import partial
from numpy.linalg import inv
from scipy.linalg import inv as sciinv
from scipy.stats import norm


# Download and Import Modules for newey west regression
r = requests.get("https://raw.github.com/OGiesecke/EmpiricalAssetPricingUtilities/master/python_nw.py")
r.status_code
with open('python_nw.py', 'wb') as f:
    f.write(r.content)


'''
Different functions for performing GMM in an asset pricing setting

jvalue: calculates the loss of the GMM efficient iteration
nwcov: calculates Newey-West covariance matrix with option for lags
efficient_gmm_iteration: performs the iteration of the efficient GMM algo until convergence

End output: betas, t-stat of betas, p-values of betas
'''


#jvalue
def jvalue(delta,S,returns, stf):
    # set parameters
    b = delta
    T,n = returns.shape
    #m = stf.shape[1]
    
    # stochastic discount factor for each period
    m_v = 1-stf@b.transpose()
    
    #calculate residuals
    summ = np.zeros((1,n))
    for i in range(T):
        summ[0,:] = summ[0,:] + returns[i,:]*m_v[i]
    
    g = (1/T)*summ
    #calculate J-value
    result = T*g @ S @ g.transpose()
    
    return result[0,0]

#jvalue(delta_ini,S0,ret,sdf)

#nwcov; L is the number of lags here
def nwcov(delta,returns,stf, L):
    T, n = returns.shape
    #gamma = np.zeros((n+1, n+1, 2*TT-1))

    ## compute residuals 
    b = delta
    # new stochastic discount factor for each period
    b = delta
    T,n = returns.shape
    #m = stf.shape[1]
    
    # stochastic discount factor for each period
    m_v = 1-stf@b.transpose()
    
    #calculate residuals
    
    resid = np.zeros((n,T))
    #resid contains estimated g for all times and all 33 factors, in the format (factors, dates)
    #resid contains the residuals of the model; if model is correct they need to be close to zero
    for i in range(T):
        resid[:, i] = returns[i,:]*m_v[i] 
    
    ## calculate S-hat matrix with the lags 
    G = np.zeros((n,n))
    l = 0
    while l<L+1:
        summ = np.zeros((n,n))
        for i in range(l,T):
            summ += np.outer(resid[:,i],resid[:,i-l])
        if l == 0:
            G = G + summ
        else:
            G = G + (L+1-l)/(L+1)*(summ + summ.transpose())
        l += 1
        
    S_hat = (1/T)*G  
    return S_hat



def efficient_gmm_iteration(ret,sdf,lag):

    T, n = ret.shape
    m = sdf.shape[1]

## now we come to the GMM iteration

# first we initialize the iteration
    delta_ini = np.ones((1,m))
    S0 = np.identity(n)

    jval = partial(jvalue, S = S0, returns = ret, stf = sdf)
    opt = minimize(jval, delta_ini)
    delta_new = opt.x

    shat = nwcov(delta_new,ret,sdf,lag)

### repeat GMM until convergence of the estimate for the delta parameters of the moment conditions (SDF + market first moment)
    t = 1 # counter for iterations of convergence
    err = 1 ## set initial error
    W_new = inv(shat)
# run GMM iteration until convergence
    while (err>10e-10) & (t<1000):
        delta_ini = delta_new
        W = W_new
    
        jvalaux = partial(jvalue, S = W, returns = ret, stf = sdf)
        delta_new = minimize(jvalaux, delta_ini).x
        shat = nwcov(delta_new,ret,sdf,lag)
        W_new = inv(shat)
    ## error is frobenius norm of covariance matrix differences
        err = np.linalg.norm(W_new-W)
    ## alternative error one can use: measure convergence in GMM estimates
    #err = np.absolute(sum(np.absolute(delta_ini - delta_new)))
    
        t += 1
    
    S_hat = nwcov(delta_new,ret,sdf,lag)
       
    Gbthat = np.zeros((n,m))
    for t in range(T):
        for j in range(m):
            Gbthat[:,j] += ret[t,:]*sdf[t,j]
        #Gbthat[:-1,1] += ret[i,:]*np.square(m_demean[i])
    Ghat = -(1/T)*Gbthat
    

# compute the asymptotic variance-covariance, using the S_hat resulting from the GMM estimation
    S = sciinv(S_hat)
# the asymptotically efficient covariance matrix is given as follows
    Avarhat = sciinv(Ghat.transpose()@S@Ghat)
# the correlations are very very small, except maybe for the CP factor

# compute the SE for the final output
    StdE = np.sqrt(Avarhat.diagonal())/np.sqrt(T) 
# now we can do a t-test to test for significance of the factors
    delta_stat = delta_new/StdE
    p_val = 2*(1 - norm.cdf(np.abs(delta_stat)))
    
        
    return delta_new, delta_stat, p_val




