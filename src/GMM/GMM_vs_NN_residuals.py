#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:01:04 2023

@author: jed169
"""


import sys, pandas as pd, numpy as np
sys.path.insert(0, '/ix/jduraj/Bonds/Code/')
from NN import performancemeasures as pm
from GMM_performancemeasures import diebold_mariano_test
import gmm_paths


def main(test = False):
    nnresid = {}
    dmstat = {}
    p_value = {}
    for nnmodel in ['SR','MP']:
        nnresid[nnmodel] = pd.read_pickle(f"{gmm_paths.savepath}/{nnmodel}_EJNPort_raw_betareg_data_mean.pkl").sort_values(['date_id','asset_id'])
        nnresid[nnmodel][f'{nnmodel}_residual'] = np.concatenate(list(pm.analyze_residuals(nnresid[nnmodel])[2].values()), axis = 0)
    
        enumdates = pd.read_excel(f"{gmm_paths.savepath}/enum_dates_{nnmodel}.xlsx").drop(columns = ['Unnamed: 0'])
        enumassets = pd.read_excel(f"{gmm_paths.savepath}/enum_assets_{nnmodel}.xlsx").drop(columns = ['Unnamed: 0'])
        nnresid[nnmodel] = enumdates.merge(nnresid[nnmodel], how = 'inner', on = 'date_id')
        nnresid[nnmodel] = enumassets.merge(nnresid[nnmodel], how = 'inner', on = 'asset_id').drop(columns = ['asset_id','date_id','t_index','ret_e','beta']).rename(columns = {'asset_idx':'asset_id'}).sort_values(['date','asset_id'])
    
    nnresid = nnresid['SR'].merge(nnresid['MP'], how = 'inner', on = ['date','asset_id'])
    #gmmresid = pd.read_pickle(f"{gmm_paths.savepath}/residuals_PCA_30_CP08_1.pkl").reset_index()
    gmmresid = pd.read_pickle(f"{gmm_paths.datapath}/int/pred_regression_oos_residuals_cp30factors.pkl").reset_index().rename(columns={'residuals':'gmm_residual'})
    if test:
        gmmresid = gmmresid[gmmresid['date']>'2008-01-01']
    resids = nnresid.merge(gmmresid, how = 'inner', on = ['date','asset_id']).set_index(['date','asset_id'])
    for nnmodel in ['SR','MP','SR_vs_MP']:
        if nnmodel!='SR_vs_MP':
            dmstat[nnmodel+' vs. GMM'], p_value[nnmodel+' vs. GMM'] = diebold_mariano_test(df_residuals_1 = resids[[f'{nnmodel}_residual']], df_residuals_2 = resids[['gmm_residual']])
        else:
            dmstat[nnmodel], p_value[nnmodel] = diebold_mariano_test(df_residuals_1 = resids[['SR_residual']], df_residuals_2 = resids[['MP_residual']])
    
    dmstat = pd.DataFrame.from_dict(dmstat, orient = 'index', columns= ['dmstat'])
    p_value = pd.DataFrame.from_dict(p_value, orient = 'index', columns = ['p-value'])
    dmstat = pd.concat([dmstat, p_value], axis = 1)
    if test:
        dmstat.to_pickle(f"{gmm_paths.savepath}/bond_pfs_Diebold_Mariano_test.pkl")
    else:
        dmstat.to_pickle(f"{gmm_paths.savepath}/bond_pfs_Diebold_Mariano.pkl")
    print(dmstat)
    
if __name__=='__main__':
    main()
    main(True)