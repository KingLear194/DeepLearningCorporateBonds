#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:22:39 2021

@author: jed-pitt
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from functools import partial
from numpy.linalg import inv
from scipy.stats import chi2
from scipy.linalg import inv as sciinv

import gmm_paths
from GMM_classical_utilities import *
from GMM_performancemeasures import *


'''
Here the GMM for bonds is executed. Preprocessing was done in the file 
GMM_bonds_preprocessing.

Models are encoded via a pair (i,j) where 
i = number of PCA macro factors, j = 1 if CP08 present, 0 if not present. 

There are functions for creating a dictionary of dataframes for both betas and performance.
Results of final dataframes are saved in excel format in {gmm_paths.datapath}/GMM

Latex codes for final tables are produced in the end. 
'''

def gmm_bonds(i,j):
        
    lag = 0
    gmm = pd.read_pickle(f'{gmm_paths.datapath}/int/GMM/GMM_Data_PCA_{i}_CP08_{j}.pkl')
    gmm_test = gmm.iloc[:,:41]
    ret = gmm_test.iloc[:,1:]
    assets = ret.columns
    dates = gmm['date']
    ret = ret.to_numpy()
    
    gmm_fac = gmm.iloc[:,41:]
    sdf = gmm_fac.to_numpy()
    
    T, n = ret.shape
    m = sdf.shape[1]  
    delta_new, delta_stat, p_val = efficient_gmm_iteration(ret,sdf,lag)
    # note the regression with only CP08 will end here and go directly to performance measures
    # start dataframe for performance measures
    perfor = []
    
    bet = delta_new
    S_hatt = nwcov(delta_new,ret,sdf,lag)

    # stochastic discount factor for each period
    m_v = 1-sdf@bet.transpose()

    g_hat = np.zeros((n,T))
    for i in range(T):
        g_hat[:, i] = ret[i,:]*m_v[i]

    summ = np.zeros((n,1))
    summ = np.sum(g_hat,axis = 1)
    gn_hat = (1/T)*summ    

    # J test statistic
    J_stat = T*np.transpose(gn_hat)@inv(S_hatt)@gn_hat

    # here we have n portfolios, and m instrumental variables in the sdf factors
    # so we have to check against a chi-squared distribution with n-m degrees of freedom
    p_value = (1-chi2.cdf(J_stat,n-m))
    # here if p-value is big, means we cannot reject the H0 that the model is correctly specified from the data
            
    perfor.append(p_value)  ## add p-value of overidentifying restrictions for ALL moments to perfor
            
    if j == 1:
        ## (2) here we try if the model with just CP as a factor is better
        # the Null hypothesis is that the models are similar in the amount of variance they explain
        # we use the same weighting matrix as for the full model

        delta_ini_alt = 1
        # we use the same weighting matrix; recall that S_hat was calculated with the convergence values delta_new from the unrestricted model
        # it is calculated in the efficient_gmm_iteration function from
        # GMM_classical_utilities.py
        S = nwcov(delta_new,ret,sdf,lag)
        W_alt = sciinv(S)
        # extract the cp
        cp = sdf[:,-1].reshape(-1,1)
        # minimization of objective function for the restricted model
        jvalcp = partial(jvalue, S = W_alt, returns = ret, stf = cp)
        opt = minimize(jvalcp, delta_ini_alt)
        delta_new_alt = opt.x
            
        # calculate the SDF
        m_v1 = 1-delta_new_alt*cp

        g_hat1 = np.zeros((n,T))
        for i in range(T):
            g_hat1[:, i] = ret[i,:]*m_v1[i]

        summ = np.zeros((n,1))
        summ = np.sum(g_hat1,axis = 1)
        gn_hat1 = (1/T)*summ    

        # J-test stat
        J_stat1 = T*np.transpose(gn_hat1)@S@gn_hat1

        # The test comparing the two models is then restricted model - unrestricted model
        diffmod_test = J_stat1 - J_stat
        # number of degrees of freedom should be no of factors before - 1 factor now
        p_value_diffmodtest = 1 - chi2.cdf(diffmod_test,m-1)
        # if p-value is very small, then the H0 that both models are similar in terms of explained variance is rejected at all typical significance levels. 

        perfor.append(p_value_diffmodtest) ## add p-value of overidentifying restrictions for macro PCA to perfor

    ############## calculate performance measures 
    # for SR here for GMM need to run Fama-Macbeth
    # we do a version of the SR both with and without Schanken correction
    # it takes mr (test returns) and mf (factors) as numpys, and an indicator about shanken
    Sharpe_shanken, rp_shanken = SR_GMM(ret,sdf,delta_new,1)
    Sharpe_noshanken, rp_noshanken = SR_GMM(ret,sdf,delta_new,0)
           
    ## here we use a linear model with intercept to evaluate residuals and fitted values for explained variance
    evar= EVar(ret,sdf)

    # the classical R^2 value, for all 40 portfolios at once
    rsquared, residu = Rsquared(ret,sdf)
    
    residu = residu.reshape(-1,1)
    idx = list(zip([dt for dt in list(dates) for _ in range(len(assets))], list(assets)*len(dates)))    
    residu = pd.DataFrame(residu, index = pd.MultiIndex.from_tuples(idx, names = ['date','asset_id']), columns = ['gmm_residual'])
    
    perfor.extend([Sharpe_shanken,evar,rsquared]) # add rest of measures to perfor

    # create dataframe for performance
    perform = df_performance(i,j,perfor)
        
    ### create dataframe for the betas
    betas = df_sdf_structure(delta_new, delta_stat, p_val,j)
    
    return betas, perform, residu
    

# function to create df for sdf structure    
def df_sdf_structure(delta_new, delta_stat, p_val, j):
    
    s = len(delta_new)
    
    if j == 1: 
        cols = [f'PC{j}' for j in range(1,s)]
        cols.append('CP')
    else:
        cols = [f'PC{j}' for j in range(1,s+1)]
        
    ind = ['beta', 't-stat', 'p-value']
    
    df = pd.DataFrame(np.vstack([delta_new, delta_stat, p_val]), index = ind, columns = cols, dtype = 'float')
    df.style.format('{:,.2f}')   
    
    return df

# function to create df for performance measures     
def df_performance(i,j,lis):
    i = int(i)
    j = int(j)
    
    cols = ['Sharpe Ratio', 'Explained Variance', 'R-squared']
    
    #if i > 0:
        
    cols.insert(0,'p-value J-test all moments')
        
    if j>0:
        cols.insert(1,'p-value J-test macro+CP vs. CP')
            
    df = pd.DataFrame(data = np.asarray(lis).reshape(1,-1), index = ["values"],
                      columns = cols)
    
    return df
        

def execute_gmm():
    ## execution of gmm
    ## Note: we can't run gmm with 45 pca factors because we only have 40 test assets
    ## but we can run with 30 factors which is the optimal number
    # if we constrain to a max of 39 factors
    betas = {}
    perform = {}
    residu = {}
    for (i,j) in [(0,1),(7,0),(7,1),(8,0),(8,1),(9,0),(9,1),(30,0),(30,1)]:
            
        betas[(i,j)], perform[(i,j)], residu[(i,j)] = gmm_bonds(i,j)  
        residu[(i,j)].sort_values(['date','asset_id']).to_pickle(f'{gmm_paths.savepath}/residuals_PCA_{i}_CP08_{j}.pkl')
    
    performance_final = pd.concat([perform[key] for key in perform.keys()], 
                                  axis = 0,keys = perform.keys(),names = ['Macro PCA','CP'])        
    performance_final.fillna(value = 'n/a', inplace = True) 
    # drop pesky multiindex from index of the dataframes     
    performance_final.reset_index(inplace = True)
    performance_final.drop(columns = ['level_2'],inplace = True) 
    performance_final.set_index(keys = ['Macro PCA','CP'],inplace = True)
    
    
    betas_final = pd.concat([betas[key].T for key in betas.keys()],axis = 0,keys = perform.keys(),names = ['Macro PCA','CP'])        
    betas_final.reset_index(inplace = True)
    betas_final.rename(columns = {'level_2':'Factor'},inplace = True)
    betas_final.set_index(keys = ['Macro PCA','CP','Factor'],inplace = True)
    
    
    ## save as excel files in GMM folder of data
    os.chdir(f'{gmm_paths.savepath}')
    
    betas_final.to_excel(f'{gmm_paths.savepath}/GMMbetas.xlsx')
    performance_final.to_excel(f'{gmm_paths.savepath}/GMMperformance.xlsx')
    
    print("Done!")
    
if __name__=='__main__':
    execute_gmm()

