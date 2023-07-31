#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:57:37 2021

@author: jed-pitt

Various utilities to evaluate performance for all exercises: SDF estimation, beta networks

"""

import pandas as pd, numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import statsmodels as sm
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error


def minus_r2(x,y):
    
   return -r2_score(x,y)

def rmse(x,y):
   return np.sqrt(mean_squared_error(x,y))

epsilon = 1e-10
def sharpe_ratio(ret, epsilon = epsilon):
    '''
        Takes ret tensor of timeseries of returns and returns the sharpe ratio in numpy format.
    '''
    
    returns = ret.flatten().detach().cpu().numpy()
    mean = np.mean(returns)
    stdev = returns.std()
    small = math.isclose(stdev, 0.0, abs_tol= epsilon)
    if len(returns)==1:
        print("Series of returns has just one date!")
        return np.sign(mean)*np.inf if mean>0 else 0.0
    elif small:
        print("Standard deviation of return series inside Sharpe ratio function is very small!\n")
        return np.sign(mean)*np.inf if mean>0 else 0.0
    
    return np.mean(returns)/stdev

def sortino_ratio(ret, epsilon = epsilon, mar = 0.0):
    '''
        Takes ret tensor of timeseries of returns and returns the sortino ratio in numpy format.
        Sortino ratio = (excess)return/sigma_neg, where sigma_neg = sqrt(sum((ret_e-mar)^2)/nr_observations).
        mar = minimum acceptable return
    '''
    
    returns = ret.flatten().detach().cpu().numpy()
    mean = np.mean(returns)
    sigma_neg = (returns[returns<0]-mar)**2
    
    if len(sigma_neg)==0:
        print("No negative returns!")
        return np.inf if mean> 0 else 0.0
    
    sigma_neg = np.sqrt(((sigma_neg-mar)**2).sum()/len(returns))
    small = math.isclose(sigma_neg, 0.0, abs_tol= epsilon)
    if small:
        print("Negative deviation of return series inside sortino ratio function is very small!\n")
        return np.inf if mean> 0 else 0.0
    
    return mean/sigma_neg 


def drawdown_series(ret):
    '''
        input: series of returns in pytorch tensor form
        output: drawdown series
    '''
    returns = ret.flatten().detach().cpu().numpy()
    values = np.cumprod(np.concatenate((np.array(1).reshape(1,-1), (1+returns).reshape(1,-1)),axis = 1).flatten())[1:] 
    maxi = np.maximum.accumulate(values)
    return 1-values/maxi 

def get_drawdown_series(returns_file, trainend = 100, valend = 200, ret_label = 'return_mean', date_idx = 'date_id'):
    
    rets = returns_file[[date_idx, ret_label]].copy()
    rets_train = rets.loc[rets[date_idx]<=trainend]
    rets_train = rets[rets[date_idx]<=trainend].copy()
    rets_train['phase'] = 'train'
    rets_val = rets.loc[(rets[date_idx]>trainend) & (rets[date_idx]<=valend),:].copy()
    rets_val['phase'] = 'val'
    rets_test = rets[rets[date_idx]>valend].copy()
    rets_test['phase'] = 'test'
    
    def fun(ret):
        ret = torch.Tensor(np.array(ret))
        return drawdown_series(ret)
    
    rets_train['drawdown'] = fun(rets_train[ret_label])
    rets_val['drawdown'] = fun(rets_val[ret_label])
    rets_test['drawdown'] = fun(rets_test[ret_label])
    dd = pd.concat([rets_train, rets_val, rets_test], axis = 0)#.drop(columns = [ret_label])
    return dd
    
    
def calmar_ratio(ret, epsilon = epsilon):
    '''
        Calculates calmar ratio of a time series of returns = mean_excess_return/max_drawdown.
    '''
    max_dd = drawdown_series(ret).max()
    mean = np.mean(ret.flatten().detach().cpu().numpy())
    small = math.isclose(max_dd, 0.0, abs_tol= epsilon)
    if small:
        print("Drawdown is too small!")
        return np.inf if mean> 0 else 0.0
    
    return mean/max_dd

def analyze_returns(returns):
    returns = torch.Tensor(returns.to_numpy().flatten())
    dic = {}
    dic['sharpe_ratio'] = sharpe_ratio(returns)
    dic['sortino_ratio'] = sortino_ratio(returns)
    dic['calmar_ratio'] = calmar_ratio(returns)
    
    return pd.DataFrame.from_dict(dic, orient = 'index', columns = ['value'])

def cagr(rets, trainend, valend, date_idx = 'date_id', ret_label = 'return_mean'):
    
    ret = rets.copy()
    cagr = {}
    ret['retplusone'] = ret[ret_label]+1
    cmgr_train = np.array((ret.loc[ret[date_idx]<=trainend,ret_label]+1)).flatten()
    len_train = len(cmgr_train)
    cmgr_train = np.power(np.prod(cmgr_train), 1/len_train)
    cmgr_val = np.array((ret.loc[(ret[date_idx]>trainend)&(ret[date_idx]<=valend),ret_label]+1)).flatten()
    len_val = len(cmgr_val)
    cmgr_val = np.power(np.prod(cmgr_val), 1/len_val)
    cmgr_test = np.array((ret.loc[ret[date_idx]>=valend,ret_label]+1)).flatten()
    len_test = len(cmgr_test)
    cmgr_test = np.power(np.prod(cmgr_test), 1/len_test)
    
    cagr['train'] = -1.0+np.power(cmgr_train,12)
    cagr['val'] = -1.0+np.power(cmgr_val,12)
    cagr['test'] = -1.0+np.power(cmgr_test,12)
    
    return cagr

def max_mean_leverage(weights, trainend, valend, date_idx = 'date_id', weights_label = 'weights_mean', plot = True):
    
    '''
        plot = True plots leverage overall and in the test set
    '''
    
    wei = weights.copy()
    result = {}
    result['train_max'] = wei.loc[wei[date_idx]<=trainend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).max()
    result['val_max'] = wei.loc[(wei[date_idx]>trainend)&(wei[date_idx]<=valend), [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).max()
    result['test_max'] = wei.loc[wei[date_idx]>valend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).max()
    
    result['train_mean'] = wei.loc[wei[date_idx]<=trainend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).mean()
    result['val_mean'] = wei.loc[(wei[date_idx]>trainend)&(wei[date_idx]<=valend), [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).mean()
    result['test_mean'] = wei.loc[wei[date_idx]>valend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).mean()
    
    if plot:
        wei = wei[[date_idx, weights_label]]
        wei['leverage'] = wei.groupby('date_id')[weights_label].transform(lambda x: x.abs().sum())
        wei = wei[[date_idx, 'leverage']].drop_duplicates()
        wei[wei[date_idx]>valend].set_index(date_idx).plot(title = 'leverage on test set')
        wei.set_index(date_idx).plot(title = 'leverage overall')

    return result

def analyze_residuals(df, epsilon = epsilon):
    '''
    input:
        df with columns: date_id, asset_id, ret_e, betas (as estimated from BetaNet FFN network), 
        sorted according to date_id and asset_id
    output:
        explained variation (ev), cross-sectional-R2 (xsr2)
    '''
    df = df.sort_values(by = ['date_id','asset_id'])
    residuals = {}
    ev_num, ev_den = 0,0
    for dt in df.date_id.unique():
        
        rets = df.loc[df['date_id'] == dt, 'ret_e'].to_numpy()
        betas = df.loc[df['date_id'] == dt, 'beta'].to_numpy()
        reg = LinearRegression(fit_intercept = False)
        
        reg.fit(betas.reshape(-1,1),rets.flatten())
        
        residuals[dt] = rets - reg.predict(betas.reshape(-1,1))
        
        ev_num+=np.mean(residuals[dt]**2)
        ev_den+=np.mean(rets**2)
    
    small = math.isclose(ev_den, 0.0, abs_tol= epsilon)
    if small:
        ev = -np.inf
    else:
        ev = 1-ev_num/ev_den
        
    df['res'] = np.concatenate(list(residuals.values()))
    
    df = df.sort_values(by = ['asset_id'])
    xsr2_num, xsr2_den = 0, 0
    for asset in df.asset_id.unique():
        rets = df.loc[df['asset_id']==asset, 'ret_e'].to_numpy().flatten()
        resid = df.loc[df['asset_id']==asset, 'res'].to_numpy().flatten()
        xsr2_num+=len(resid)*np.mean(resid)**2
        xsr2_den+=len(rets)*np.mean(rets)**2
    
    small = math.isclose(xsr2_den, 0.0, abs_tol= epsilon)
    if small:
        xsr2 = -np.inf
    else:
        xsr2 = 1-xsr2_num/xsr2_den
        
    return ev, xsr2, residuals

def get_analyze_residuals(betas, returns_path, end_date_train=300, end_date_val=475):
    residuals = pd.read_pickle(returns_path)
    returns = residuals.iloc[:,3].to_numpy()
    lm = LinearRegression(fit_intercept=False).fit(betas,returns)
    residuals['beta_reg_residuals'] = returns - lm.predict(betas)
        
    return residuals
    
def get_cumrets(returns, date_idx = 'date_id', return_label='return_mean', plot=True):
    
    returns = returns[[date_idx, return_label]].copy()
    returns['retplusone'] = returns[return_label]+1
    returns['cumret'] = returns['retplusone'].cumprod()
    
    returns = returns[[date_idx,'cumret']].copy()
    start = pd.DataFrame([[returns[date_idx].min()-1, 1.0]], columns = [date_idx, 'cumret'])
    returns = pd.concat([start, returns], axis = 0)
    
    if plot:
        returns[[date_idx,'cumret']].plot(x = date_idx, title = 'cumulative return')
    
    return returns
    
'''
Code for loss functions in the training of NN with mispricing and Sharpe ratio
'''

def onehot_encoding(tens):
    '''
        Tensor needs to be in torch.long format.
    '''
    
    onehotcand = F.one_hot(tens)
    idx = onehotcand.sum(dim=0).nonzero().flatten()

    return onehotcand[:,idx]

def weighted_returns(idx, returns, weights): 
    
    ret = returns.detach().clone().flatten()  
    if isinstance(idx, str):
        wret = torch.mul(weights.flatten(),ret)
    else:
        idx = idx.to(torch.long)
        if weights.shape[0]!=idx.shape[0]:
            weights = weights[idx]
        ret = ret[idx].flatten()  
        wret = torch.mul(weights.flatten(),ret)

    return wret

def calculate_pf_return(idx, date_id, weights, returns):
        
    wret = weighted_returns(idx, returns, weights)  
    shapes = onehot_encoding(date_id).sum(dim=0).to('cuda') # since used only during training
    dates_split = torch.split(date_id, list(shapes),dim=0)
    wret_split = torch.split(wret, list(shapes),dim=0)
    pf_returns, dates = torch.zeros(size = (len(dates_split),)).cuda(), torch.zeros(size = (len(dates_split),)).cuda()
    for i in range(len(dates_split)):
        pf_returns[i] = wret_split[i].flatten().sum()
        dates[i] = dates_split[i][0]
    return dates, pf_returns

def mispricing_loss(idx, date_id, new_weights, returns, asset_ids, new_test_assets, loss_weight = 1, 
                    moment_weighting = True):
    '''
    Notes: 
        - leave as tensor to be able to propagate the losses backward
        - All 1D-tensors should be flattened. 
        - Moment weighting is about the division of T_i/T, which is motivated by econometrics.
        - It returns a mean-reduced value in the end.
    '''
    
    _, sdf_ret = calculate_pf_return(idx, date_id, new_weights, returns)
    date_onehot = onehot_encoding(date_id).cuda().to(torch.float)
    sdfs = torch.matmul(date_onehot, sdf_ret.flatten()) 
    mispricings = sdfs*returns[idx]
    mispricings = mispricings.reshape(-1,1)*new_test_assets 
    
    # get asset ids
    asset_ids = asset_ids[idx]
    asset_onehot = onehot_encoding(asset_ids).cuda()
    nr_dates_assets = asset_onehot.sum(dim=0)
    
    # calculate sum of mispricings for each asset
    mispricings = torch.matmul(torch.t(asset_onehot.to(torch.float)), mispricings) 
    mispricings = torch.div(torch.t(mispricings),nr_dates_assets) 
    mispricings = torch.linalg.norm(mispricings, dim = 0).flatten()
    
    if moment_weighting:
        nr_dates = date_onehot.shape[1]
        nr_dates_assets = (nr_dates_assets/nr_dates).flatten()
        loss = nr_dates_assets*mispricings 
        loss = loss_weight*torch.mean(loss)         
    else:
        loss = loss_weight*torch.mean(mispricings)
    
    return loss

def mean_std_utility(idx, date_id, new_weights, returns, risk_aversion):
    returns = calculate_pf_return(idx, date_id, new_weights, returns)[1]
    
    return torch.mean(returns)-risk_aversion*torch.std(returns, unbiased = False)

def minus_mean_std_utility(idx, date_id, new_weights, returns, risk_aversion, new_test_assets=None):
    '''
    new_test_assets is a dummy variable for this function to be able to modularise the training classes
    '''
    return -mean_std_utility(idx, date_id, new_weights, returns, risk_aversion)

'''
Various code for evaluating results
'''
def split_rets(rets, trainend, valend, return_label = 'return_mean', date_idx = 'date_id'):
    if return_label is not None:
        rets = rets[[date_idx, return_label]]
    rets_train = rets[rets[date_idx]<=trainend].copy()
    rets_test = rets[rets[date_idx]>valend].copy()
    rets_val = rets.loc[(rets[date_idx]<=valend) & (rets[date_idx]>trainend), :].copy()
    
    return rets_train, rets_val, rets_test

def calc_ratios(rets, trainend, valend, return_label = 'return_mean', date_idx = 'date_id'):
    
    rets_train, rets_val, rets_test = split_rets(rets, trainend, valend, return_label, date_idx)
    perfdict = {}
    perfdict['train_monthly'] = analyze_returns(rets_train[return_label])
    perfdict['val_monthly'] = analyze_returns(rets_val[return_label])
    perfdict['test_monthly'] = analyze_returns(rets_test[return_label])
    
    perfdict['train_yearly'] = analyze_returns(rets_train[return_label], output_yearly=True)
    perfdict['val_yearly'] = analyze_returns(rets_val[return_label], output_yearly=True)
    perfdict['test_yearly'] = analyze_returns(rets_test[return_label], output_yearly=True)
    perf = pd.concat(list(perfdict.values()), axis = 1)
    perf.columns = list(perfdict.keys())
        
    return perf
    
def simple_ensemble(filelist, trainend=100, valend=300, date_idx = 'date_id'):
    final = pd.read_pickle(filelist[0])
    final.rename(columns = {'pf_return':'pf_return_1'},inplace=True)
    for i, file in enumerate(filelist[1:],2):
        df = pd.read_pickle(file)
        df.rename(columns = {'pf_return':f'pf_return_{i}'}, inplace = True)
        final = final.merge(df, how = 'inner', on = date_idx)
    final.set_index(date_idx,inplace = True)
    final['median_ensemble'] = final.apply(np.median, axis=1)
    final['mean_ensemble'] = final.apply(np.mean, axis=1)
    final.reset_index(inplace = True)
    final = final[[date_idx, 'median_ensemble','mean_ensemble']]
    final_train = final[final[date_idx]<=trainend]
    final_test = final[final[date_idx]>valend]
    final_val = final.loc[(final[date_idx]<=valend) & (final[date_idx]>trainend), :]
    
    perfdict = {}
    for method in ['median', 'mean']:
        perfdict[f'perf_train_{method}'] = analyze_returns(final_train[f'{method}_ensemble'])
        perfdict[f'perf_val_{method}'] = analyze_returns(final_val[f'{method}_ensemble'])
        perfdict[f'perf_test_{method}'] = analyze_returns(final_test[f'{method}_ensemble'])
    
    return perfdict
        
        
'''
Code for time series testing/diagnostics
'''

def adfuller(df):
    '''
    Test for unit root in time series. 
    Allows for multiple time-series (separately)
    Date/timestamp should be as index. 
    '''
    
    results = {}
    for col in df.columns:
        adf_stat, p_value, used_lags, n_obs, cvs, aic = sm.tsa.stattools.adfuller(df[col])
        results[col] = list([adf_stat]+[p_value]+[used_lags]+[n_obs]+list(cvs.values())+[aic])
        
    results = pd.DataFrame.from_dict(results, orient='index', columns = ['adf_stat','adf_pvalue','used_lags','used_n_obs','critical_val_1%','critical_val_5%','critical_val_10%','aic'])
    
    return results

def largest_significant_lag(pacf, alpha, T):
    
    '''calculates significance of pacf and stops when finds last lag that is not significant at the 1-alpha level'''
    
    crit_val = norm.ppf(1-alpha)
    sig_test = lambda tau_h: np.abs(tau_h) > crit_val/np.sqrt(T)
    for i in range(len(pacf)):
        if sig_test(pacf[i]) == False:
            n_steps = i-1
            break;
    return n_steps

def pacf_sig_values(df, n_max_lags = 48, alpha=0.01, plot = False):
    '''
        Calculates pacf and finds first lag that isn't significant
        at level 1-alpha. Then this lag-1 is an upper bound for seq_len in RNNs
    '''
    
    T = len(df)
    pacf={}
    for col in df.columns:
        pacf[col] = sm.tsa.stattools.pacf(df[col], nlags = n_max_lags)
    
    pacf = pd.DataFrame.from_dict(pacf, orient = 'index', columns = [f"lag_{i}" for i in range(n_max_lags+1)]).iloc[:,1:]
    if plot:
        crit_val = norm.ppf(1-alpha)
        plt.plot(pacf.T.reset_index(drop=True))
        plt.plot([crit_val/np.sqrt(T)]*n_max_lags, label = f"{1-alpha}% confidence interval (upper)")
        plt.plot([-crit_val/np.sqrt(T)]*n_max_lags, label = f"{1-alpha}% confidence interval (lower)")
        plt.xlabel('number of lags')
        plt.xticks(np.arange(0,n_max_lags,int(n_max_lags/10)))
        plt.legend(loc = 'upper right')
    
    pacf['last_sig_lag'] = pacf.apply(lambda x: largest_significant_lag(x, alpha, T), axis=1)
        
    return pacf

def ljungbox_test(df_residuals, n_max_lags = 10):
    
    '''
        Ljung-Box test for residuals produced from RNN. 
        Low p_values means residuals correlated and hence RNN has not done its job. 
    '''
    results = {}
    for col in df_residuals.columns:
        results[col] = sm.stats.diagnostic.acorr_ljungbox(df_residuals[col], lags = n_max_lags, boxpierce=False)
        results[col] = results[col]['lb_pvalue'].to_numpy()
        print(results[col].shape)

    results = pd.DataFrame.from_dict(results, orient = 'index')
    results.columns = [f'pvalue_lag_{i}' for i in range(1,n_max_lags+1)]        
    
    return results 
