#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:34:18 2023

@author: jed169

Code for the HMM for EJN Portfolios

"""

import pandas as pd, numpy as np, sys, pickle
from matplotlib import cm, pyplot as plt
from hmmlearn import hmm

n_states = 3 # or 2

project_name = f'EJNPorts_{n_states}'
datapath= '/ix/jduraj/Bonds/Data/final_dataset/dat_nozawa_int_eret_long.pkl'
savepath = f'/ix/jduraj/Bonds/Data/HMM_{project_name[:-2]}'


def eigvals_cov(cov, top=-1, normalized=False):
    '''
    Returns eigvals of symmetric matrix, with options to normalize by L1-norm, or just return top-k instead of all
    '''
    eigs, _ = np.linalg.eigh(cov)
    if top==-1: top=len(eigs)
    if normalized:
        eigs/=np.abs(eigs).sum()
    return np.flip(eigs)[:top]


def plot_states(Z, n_states, savepath = None):
    fig, axs = plt.subplots(n_states, sharex=True, sharey=True)
    colors = cm.rainbow(np.linspace(0,1,n_states))
    for i, (ax, color) in enumerate(zip(axs, colors)):
        mask = Z['hidden_state']==i
        ax.plot_date(Z.index[mask],
                     Z['hidden_state'][mask],
                     ".", linestyle = 'none',
                     c=color)
        ax.set_title(f"Hidden state {i}")
    fig.tight_layout()
    plt.show()
    if savepath:
        plt.savefig(savepath+f'/{n_states}_states')
        
def save_results(savepath, Z, X, model):
    dic = {}
    dic['n_states'] = n_states
    dic['estimated_states'] = Z
    dic['estimated_start_probabilities'] = model.startprob_
    dic['estimated_transition_matrix'] = model.transmat_ 
    dic['estimated_state_means'] = model.means_
    dic['estimated_state_covars'] = model.covars_
    dic['aic'] = model.aic(X)
    
    with open(f"{savepath}/HMM_{project_name}_results.pkl",'wb') as handle:
        pickle.dump(dic, handle)
    
    
def main():
    
    sys.stdout = open(f"{savepath}/HMM_{project_name}_log.out", 'w')
    
    print("Number of states is ", n_states)
    
    returns = pd.read_pickle(datapath).reset_index().sort_values(['date','asset_id'])
    returns = returns.pivot(index = 'date', columns = 'asset_id', values = 'ret_e')

    X = returns.values

    n_iter = 100
    model = hmm.GaussianHMM(n_components = n_states, covariance_type = 'full', n_iter = n_iter, random_state=0)

    model.fit(X)
    Z = model.predict(X)
    states = pd.unique(Z)
    print("States in order of appearance are: ", states)

    Z = pd.DataFrame(Z, index = returns.index, columns = ['hidden_state'])
    
    plot_states(Z,n_states, savepath = savepath)
    
    print("\nModel start probabilities:")
    print(model.startprob_)
    print("\nModel transition matrix:")
    print(model.transmat_)

    print("\nModel means:")
    print(model.means_)
    print("\nEigenvalues of the covariance matrices")
    for st in range(n_states):
        print(f"\nTop 2 eigenvalues of cov of state {st}:")
        print(eigvals_cov(model.covars_[st], top = 2, normalized = False))
        print(f"Top 2 eigenvalues (normalized) of cov of state {st}:")
        print(eigvals_cov(model.covars_[st], top = 2, normalized = True))
    save_results(savepath, Z,X, model)

    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    
if __name__ == '__main__':
    main()
