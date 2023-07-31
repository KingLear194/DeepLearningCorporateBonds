#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 16:53:32 2023

@author: jed169
"""
import numpy as np
from scipy.stats import norm
from scipy.linalg import inv as sciinv


from python_nw import newey


# for SR here for GMM need to run Fama-Macbeth
# note, this is probably not optimal for non-tradeable factors

def SR_GMM(mr,mf,betasdf,shanken):
    sdfbeta = betasdf
    
    T, N = mr.shape
    
    if len(mf.shape) == 2:
        m = mf.shape[1]
    else:
        m = 1
    mf = mf.reshape((T,m))
    
    # time series regression
    mbeta = np.zeros((N,m))
    malpha = np.zeros((N,1))
    ones = np.ones((T,1))
    for i in range(N):
        
        vres = newey(mr[:,i],np.hstack((ones, mf)).reshape(T,m+1),0)
        malpha[i] = vres.beta[0]
        mbeta[i,:] = vres.beta[1:].transpose()
        
    lambdas = np.zeros((T,m))
    
    for t in range(T):
        ones2 = np.ones((N,1))
        vres2 = newey(mr[t,:],np.hstack((ones2,mbeta)).reshape(N,m+1),0)
        lambdas[t,:] = vres2.beta[1:].transpose()
    
    # calculate average riskpremia    
    riskpremia = np.mean(lambdas, axis = 0)    
    tangential = np.zeros((T,1))
    
    for u in range(T):
        tangential[u] = np.dot(sdfbeta,lambdas[u,:])
            
    #### split here according to shanken indicator
    if shanken == 0:
        
        SR = np.mean(tangential)/np.std(tangential, dtype=np.float64)
        
    else:
        m1 = np.mean(lambdas,axis = 0)
        m2 = np.mean(mf,axis = 0)
        
        sigmaf = np.zeros((m,m))
        summ = np.zeros((m,m))
        for t in range(T):
            v = mf[t,:]-m2
            summ += np.outer(v,v)
        sigmaf = (1/(T-m))*summ
        
        ark = np.zeros((m,m))
        summ1 = np.zeros((m,m))
        
        for t in range(T):
            w = lambdas[t,:]-m1
            summ1 += np.outer(w,w)
        ark = (1/T)*summ1
        
        c = m1.reshape(1,-1)@sciinv(sigmaf)@m1.reshape(-1,1)
        
        varcorr = (1+c)*(ark-(1/T)*sigmaf) + (1/T)*sigmaf
        
        varfinal = sdfbeta.reshape(1,-1)@varcorr@sdfbeta.reshape(-1,1)
        
        SR = np.mean(tangential)/np.sqrt(varfinal, dtype=np.float64)
        SR = SR[0][0]
        

    return SR,riskpremia

## here we use a linear model with intercept to evaluate residuals and fitted values for explained variance

def EVar(mr,mf):
    
    T, N = mr.shape
    m = mf.shape[1]
    #mbeta = np.zeros((N,m))
    residu = np.zeros((T,N))
    ones = np.ones((T,1))
    
    for i in range(N):
        vres = newey(mr[:,i],np.hstack((ones, mf)).reshape(T,m+1),0)
        #mbeta[i,:] = vres.beta.transpose()
        residu[:,i] = vres.resid
    
    residvarnum = 1/(T*N)*np.sum(residu**2)
    residvarden = 1/(T*N)*np.sum(mr**2)

    return 1-residvarnum/residvarden

# the classical Rsquared measure for the whole panel data (not test asset, per test asset)

def Rsquared(mr,mf):
    
    T, N = mr.shape
    m = mf.shape[1]
    #mbeta = np.zeros((N,m))
    residu = np.zeros((T,N))
    ones = np.ones((T,1))
    
    for i in range(N):
        vres = newey(mr[:,i],np.hstack((ones, mf)).reshape(T,m+1),0)
        residu[:,i] = vres.resid
        
    m3 = np.mean(mr)
    residvarnum = 1/(T*N)*np.sum(residu**2)
    residvarden2 = 1/(T*N)*np.sum((mr-m3)**2)
    
    rsquared = 1-residvarnum/residvarden2

    return rsquared, residu

def diebold_mariano_test(df_residuals_1, df_residuals_2, date_idx = 'date', lag = 0):

    '''
    Diebold-Mariano test according to Gu, Kelly, Xiu (RFS 2020)
    
    Inputs:
        df_residuals_1, df_residuals_2: [dataframes] with residuals, double index should be (date, asset_id)
            date_idx is time-period identifier, default = 'date'
        lag: [int] lags for the Newey-West variance estimation of the ultimate time series of squared residuals,
            default = 0 (no overlapping returns)
        
    Outputs:
        dm-statistic: [float]
        p-value: [float] dm-statistic is standard normal distributed under H0
    '''
    df_residuals_1 = df_residuals_1.sort_index()
    df_residuals_1.index.set_names(['date','asset_id'],inplace = True)
    df_residuals_2 = df_residuals_2.sort_index()
    df_residuals_2.index.set_names(['date','asset_id'],inplace = True)
    
    if not all(df_residuals_1.index == df_residuals_2.index):
        raise ValueError('Indices of the two residual-df are not the same!')
    else:
        # this step is subotpimal!
        df_residuals_1.index = df_residuals_2.index.copy()
        
    #df = df_residuals_1.merge(df_residuals_2, how = 'inner', left_index = True, right_index = True)
    df = df_residuals_1.merge(df_residuals_2, how = 'inner', left_index = True, right_index = True)
    
        
    df.rename(columns = {df.columns[0]:'residuals_1',df.columns[1]:'residuals_2'}, inplace = True)
    df = df.astype({df.columns[0]: 'float64',df.columns[1]: 'float64'})
    
    #df.index = df.index.set_names(['date','asset_id'],inplace = True)
    df = df.sort_index()
            
    df_aux = df[['residuals_1']]
    df['no_assets'] = df_aux.groupby(level = 'date').transform(np.size)
    df['residuals_1-squared'] = df[['residuals_1']].pow(2)
    df['residuals_2-squared'] = df[['residuals_2']].pow(2)

    df['d_aux'] = (df['residuals_1-squared']-df['residuals_2-squared'])
    df['d_auxsum'] = df.groupby(level = 'date')['d_aux'].transform(np.sum)
    df['d_t'] = df['d_auxsum']/df['no_assets']
    
    df = df[['d_t']].drop_duplicates().droplevel(level = 1)
    
    d1 = df['d_t'].to_numpy().ravel()
    d = np.nanmean(d1)
    T = df.shape[0] #nr dates
    
    # calculate the newey-west standard error
    D = df.to_numpy()
    sigma = 0.0
    l = 0
    while l<lag+1:
        summ = 0.0
        for i in range(l,T):
            summ += D[i]*D[i-l]
        if l == 0:
            sigma = sigma + summ
        else:
            sigma = sigma + (lag+1-l)/(lag+1)*2*summ
        l += 1
        
    sigma = (1/T)*sigma
    
    # calculate the Diebold-Mariano statistic
    if sigma == 0:
        dmstat = np.Inf
    else:
        # multiply by square root of T to get asymptotic normality  (CLT)
        dmstat = np.sqrt(T)*((d/np.sqrt(sigma))[0])
        
    # calculate p-value for H0: forecasts are equally good
    p_value = 2*(1 - norm.cdf(np.abs(dmstat)))
        
    return dmstat, p_value