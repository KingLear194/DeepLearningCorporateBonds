#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:26:55 2021

@author: jed-pitt

this contains several versions of pca

"""

import numpy as np

from numpy.linalg import eigh 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

#### Several PCA procedures

# the typical sklearn pca
def pcask(df, n):
    
    X = df.to_numpy()
    # note, no need to demean, because that is done automatically by pcask
    #create instance
    pca = PCA(n_components = n, svd_solver='full')
    components = pca.fit_transform(X)
    expvar = pca.explained_variance_ratio_
    
    return components, expvar

# pcask with standardization
def pcask_standard(df, n, solver='full'):
    
    X = df.to_numpy()
    X = StandardScaler().fit_transform(X)
    #create instance
    pca = PCA(n_components = n, svd_solver=solver)
    components = pca.fit_transform(X)
    expvar = pca.explained_variance_ratio_
    
    return components, expvar

    
#compo1, explainedvar1 = pcask(macroagg,4)

# create a pca version where you give as input the minimum amount of variance explained
# this does not standardize before PCA
def pcaskpercent(df,x):
    X = df.to_numpy()
    # note, no need to demean, because that is done automatically by pcask
    #create instance
    pca = PCA(x, svd_solver='full')
    components = pca.fit_transform(X)
    expvar = pca.explained_variance_ratio_
    
    return components, expvar

#compo2, explainedvar2 = pcaskpercent(macroagg,0.95)



# Version of pcaskpercent where the data are standardized before use
def pcaskpercent2(df,x):
    X = df.to_numpy()
    #create instance
    X = StandardScaler().fit_transform(X)
    pca = PCA(x, svd_solver='full')
    components = pca.fit_transform(X)
    expvar = pca.explained_variance_ratio_
    
    return components, expvar



## PCA on foot: extracting indices where n-th highest eigenvalues are, and their 
## corresponding eigenvectors; https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c
# need to take into account that python packages doesn't return sorted eigenvalues/vectors

def pcapset(df,n):
    X = df.to_numpy()
    #demean
    Z = X - np.mean(X, axis = 0)
    #calculate the covariance matrix
    S = Z.T@Z
    #calculate eigenvalue decomposition; note: using eigh instead of eig to avoid carrying around
    #the zero imaginary part
    w, v = eigh(S)
    # sort eigenvalues
    # select n-largest eigenvalues
    idx = w.argsort()[::-1][:n]
    w_new = w[idx]
    #calculate the eigenvectors
    eigenvectors = v[:,idx]
    # calculate the components
    components = Z@eigenvectors # note, code in pset has X@v instead..
    expvar = w_new/w.sum()
    return components, expvar

#compo3, explainedvar3 = pcapset(macroagg,4)

# Aside: checking for differences between pca on foot and sklearn pca (which uses SVD)

# check the deviation from the two methods with argument the number of components
# first one raises attentionerror (feature of the procedure) because the sign of one component is flipped
# the second one raises no errors up to rtol=1e-10, atol=1e-10, so that the problem is that the first 
# method switches the sign of one component
#compo3[:,1] = -compo3[:,1]
#assertequal(compo1, compo3, rtol=1e-10, atol=1e-10)
#assertequal(explainedvar1,explainedvar3,rtol=1e-5, atol=1e-5)
#assertequal(np.abs(compo1), np.abs(compo3), rtol=1e-10, atol=1e-10)



#################################### ICA procedure ####################################
def icask(df, n):
    
    X = df.to_numpy()
    # note, no need to demean, because that is done automatically by pcask
    #create instance
    ica = FastICA(n_components = n, random_state = 0)
    components = ica.fit_transform(X)
    
    return components