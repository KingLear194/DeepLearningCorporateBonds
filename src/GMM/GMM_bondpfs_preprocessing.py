#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:52:02 2021

@author: jed-pitt
"""

import os

import settings

import pandas as pd
import numpy as np

from pcaversions import *
from python_nw import newey
from BaiNgEcmaUtilities import *


# check for nans the datasets that will be used for the GMM        
def checkfornans():
    naan = []
    for i in [0,7,8,9,45,30]:
        if i>0:
            gmm1 = pd.read_pickle(f'{settings.datapath}/int/GMM/GMM_Data_PCA_{i}_CP08_1.pkl')
            gmm2 = pd.read_pickle(f'{settings.datapath}/int/GMM/GMM_Data_PCA_{i}_CP08_0.pkl')
            naan.extend((gmm1.isnull().sum().sum(),gmm2.isnull().sum().sum()))
        else:
            gmm1 = pd.read_pickle(f'{settings.datapath}/int/GMM/GMM_Data_PCA_{0}_CP08_1.pkl')
            naan.append(gmm1.isnull().sum().sum())
    
    return naan

def macropcafactors(df,components):
    # Function that adds date to the macro components that were extracted
    # also creates column names
    # note: need df and components to have the same number of rows for the function to work
    n = components.shape[1]
    df1 = df[['date']].copy()
    for i in range(n):
        df1[f"PC{i+1}"] = components[:,i]
    
    return df1


def construct_cp_factor_clean_macro():
    ''' 
    Constructing the CP factor. Ultimately we use the version of the factor from the CP 2008 paper
    '''
    
    ## construct the CP-factor as in pset 5 of columbia's class
    # Pre-process the data
    start_date = "1970-10-01" ## note, the lower bound for dates is binding here
    end_date = "2021-2-28"
    
    currentyear = end_date
    
    df1 = pd.read_csv(f"{settings.datapath}/raw/fed_zerobondyields.csv",skiprows=9)
    
    df1["date"] = pd.to_datetime(df1["Date"])
    
    df1 = df1[(df1["date"]>=start_date) & (df1["date"]<=end_date) ]
    df1 = df1[~df1["BETA0"].isna()]
    df1["month"] = df1["date"].dt.month
    df1["year"] = df1["date"].dt.year
    df1["day"] = df1["date"].dt.day
    eom = df1.groupby(["year","month"])["date"].max().reset_index().rename(columns={"date":"eom"})
    df1 = df1.merge(eom,on=["year","month"],how="inner")
    
    data = df1[df1["date"]==df1["eom"]]
    data = data[["date","year","month"]+["SVENY{:02.0f}".format(x) for x in range(1,11)]+["day"]]
    # Express in percentage
    for mat in range(1,11):
        data["SVENY{:02.0f}".format(mat)] =     data["SVENY{:02.0f}".format(mat)] / 100
    
    data.to_excel(f"{settings.datapath}/int/fedzerobondyields_{currentyear}.xlsx",index=False)
    
    sample1_start = "10-01-1971"   
    sample1_end = "02-28-2021"
    
    sample_offset = pd.to_datetime(sample1_end,format="%m-%d-%Y") - np.timedelta64(1, 'Y')
    sample = data[(data["date"]>=sample1_start) & (data["date"]<=sample1_end)]
    
    # Construct forward rates 
    sample["f_1"] = sample[f"SVENY{1:02.0f}"]
    for n in range(2,11):
        sample[f"f_{n}"] = n * sample[f"SVENY{n:02.0f}"] - ( n - 1 ) * sample[f"SVENY{n-1:02.0f}"]
    
    # Construct annual excess (log) returns
    for n in range(1,11):
        sample[f"pt_{n}"] = - n * sample[f"SVENY{n:02.0f}"] 
       
    for n in range(2,11):
        sample[f"pt+1_{n-1}"] = - (n - 1) * sample[f"SVENY{n-1:02.0f}"].shift(-12)
    
    sample["rx_1"] = sample[f"SVENY{1:02.0f}"]
    for n in range(2,11):
        sample[f"rx_{n}"] = sample[f"pt+1_{n-1}"] - sample[f"pt_{n}"] - sample["SVENY01"]
        
        
    # Construct the CP factor as in Cochrane, Piazessi 2008 
    # Find the forward spreads 
    # see pp. 15 of Cochrane, Piazessi 2009 and pp. 5-6 of Hodrick, Tomunen 2018
    for n in range(2,11):
        sample[f"fs_{n}"] = sample[f"pt_{n-1}"] - sample[f"pt_{n}"] - sample["SVENY01"]
                
    sample.dropna(inplace = True)   
    
    # separate forward spreads and save 
    cols = sample.columns
    cols = cols[1:53]
    forwardspreads = sample.drop(columns = cols) 
    forwardspreads.to_pickle(f"{settings.datapath}/int/GMM_forwardspreads.xlsx")
    
    
    # The (cross-sectional) average returns on 2y, ..., 10y bonds
    sample["constant"] = 1
    subsample = sample[sample["date"]<=sample_offset].copy() 
    vr_e_ave = subsample[[f"rx_{n}" for n in range(2,11)]].mean(axis=1)
    
    # Estimate the CP factor in the original way of Cochrane, Piazessi 05
    # forward rates from 1 to 5
    
    results = newey(vr_e_ave,subsample[["constant"]+[f"f_{n}" for n in range(1,6)]],0)
    vCP = sample[["constant"]+[f"f_{n}" for n in range(1,6)]] @ results.beta
    sample = pd.concat([sample, pd.Series(vCP)],axis=1)
    sample.rename(columns={0:"vCP"},inplace=True)
    
    # Estimate the CP factor as in Cochrane, Piazessi 08 using forward spreads, from 2
    # to 5
    
    results = newey(vr_e_ave,subsample[["constant"]+[f"fs_{n}" for n in range(2,6)]],0)
    CP08 = sample[["constant"]+[f"f_{n}" for n in range(2,6)]] @ results.beta
    sample = pd.concat([sample, pd.Series(CP08)],axis=1)
    sample.rename(columns={0:"CP08"},inplace=True)
    
    
    CP = sample.filter(['year','month','vCP'], axis = 1)
    CP['day'] = 1
    CP['date'] = pd.to_datetime(CP[['year','month','day']], format = "%Y%m%d")
    CP.drop(columns = ['year','month','day'], inplace = True)
    
    CP08 = sample.filter(['year','month','CP08'], axis = 1)
    CP08['day'] = 1
    CP08['date'] = pd.to_datetime(CP08[['year','month','day']], format = "%Y%m%d")
    CP08.drop(columns = ['year','month','day'], inplace = True)
    
    # save CP files
    
    CP.to_pickle(f'{settings.datapath}/int/Cochrane-Piazessi.pkl')
    CP08.to_pickle(f'{settings.datapath}/int/Cochrane-Piazessi_2008.pkl')    
      
    '''Cleaning of macro data and finding the optimal number of PCA factors according to Bai, Ng criteria
    '''
    ## the Macro PCA factors will be constructed from macro data
    ## preprocessing of Macro data
    
    macro = pd.read_pickle(f'{settings.datapath}/int/macro_aggregated_current_transformed.pkl')
    macro.isna().values.sum()
    # 295 nans
    macro.isna().sum()
    
    # we drop ACOGNO which has 230 nans 
    macro.drop(columns = ['ACOGNO'],inplace = True)
    macro.isna().values.sum()
    # still 65 nans, forward fill 
    macro.isna().sum()[macro.isna().sum() != 0]
    macro.fillna(method = 'ffill', inplace = True)
    macro.isna().values.sum()
    # still 62 nans
    np.nonzero(macro.isna().sum().to_numpy())
    macro.iloc[:,129]
    macro.drop(columns = ['UMCSENTx'],inplace = True)
    # we see that UMCSENTx, consumer sentiment has 62 nans at the beginning, so we drop it as well
    macro.isna().sum()[macro.isna().sum() != 0]
    # there is only one value left as nan, it's a start value, that we backfill
    macro.fillna(method = 'bfill', inplace = True)
    macro.isna().values.sum()
    # no more nans
    
    macro.drop(columns = ['year','month','day'],inplace = True)
    
    ## save clean macro file
    
    macro.to_pickle(f'{settings.datapath}/int/macro_clean.pkl')

def bai_ng_analysis():

    # we select just the values
    macroagg = pd.read_pickle(f'{settings.datapath}/int/macro_clean.pkl').drop(columns = ['date'])
    # we have 132 macro time series, where 6 are Goyal, Welch-type TS, the other ones from FRED dataset
    
    ### Bai, Ng Analysis for the macro data in our dataset
    ### Based on Ludvigson, Ng, we will be trying 7,8,9 PCA factors.
    ### Here we try to find what is the optimal number of factors according to Bai, Ng criteria in our dataset
    for perc in [0.95, 0.99]:
        components, expvar = pcaskpercent2(macroagg,perc)
        print(f"{perc} of variance explained by {components.shape[1]}")
    # we see that 58 factors explain 95% of variance
    
    #components, expvar = pcaskpercent2(macroagg,0.99)
    # we see that 82 factors explain 99% of variance
    
    print(minimizeic(macroagg,82,icbaing_1,1,pic = False))
    # result = 1
    
    print(minimizeic(macroagg,82,icbaing_2,1,pic = False))
    # result = 1
    
    print(minimizeic(macroagg,82,icbaing_3,1,pic = False))
    # result = 45
    
    print(minimizepc(macroagg,82,pcbaing_1_scaling,lam = 1, pic = False))
    # result = 45
    
    print(minimizepc(macroagg,82,pcbaing_2_scaling,lam = 1, pic = False))
    # result = 45
    
    print(minimizepc(macroagg,82,pcbaing_3_scaling,lam = 1, pic = False))
    # result = 45
    
    # we look for robustness of the 45 factors result by varying the penalty weight
    
    # do PC1 optimization with scaling from 0.5 to 1.5
    pcprint(macroagg,pcbaing_1_scaling,82,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [59]
    #With penalty weight 0.611 the minimizers are [45]
    #With penalty weight 0.722 the minimizers are [45]
    #With penalty weight 0.833 the minimizers are [45]
    #With penalty weight 0.944 the minimizers are [45]
    #With penalty weight 1.056 the minimizers are [45]
    #With penalty weight 1.167 the minimizers are [45]
    #With penalty weight 1.278 the minimizers are [45]
    #With penalty weight 1.389 the minimizers are [45]
    #With penalty weight 1.5 the minimizers are [45]
    
    # do PC2 optimization with scaling from 0.5 to 1.5
    pcprint(macroagg,pcbaing_2_scaling,82,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [59]
    #With penalty weight 0.611 the minimizers are [45]
    #With penalty weight 0.722 the minimizers are [45]
    #With penalty weight 0.833 the minimizers are [45]
    #With penalty weight 0.944 the minimizers are [45]
    #With penalty weight 1.056 the minimizers are [45]
    #With penalty weight 1.167 the minimizers are [45]
    #With penalty weight 1.278 the minimizers are [45]
    #With penalty weight 1.389 the minimizers are [45]
    #With penalty weight 1.5 the minimizers are [45]
    
    # do PC3 optimization with scaling from 0.5 to 1.5
    pcprint(macroagg,pcbaing_3_scaling,82,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [59]
    #With penalty weight 0.611 the minimizers are [59]
    #With penalty weight 0.722 the minimizers are [45]
    #With penalty weight 0.833 the minimizers are [45]
    #With penalty weight 0.944 the minimizers are [45]
    #With penalty weight 1.056 the minimizers are [45]
    #With penalty weight 1.167 the minimizers are [45]
    #With penalty weight 1.278 the minimizers are [45]
    #With penalty weight 1.389 the minimizers are [45]
    #With penalty weight 1.5 the minimizers are [45]
    
    # do IC3 optimization with scaling from 0.5 to 1.5
    icprint(macroagg,icbaing_3,82,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [59]
    #With penalty weight 0.611 the minimizers are [45]
    #With penalty weight 0.722 the minimizers are [45]
    #With penalty weight 0.833 the minimizers are [45]
    #With penalty weight 0.944 the minimizers are [45]
    #With penalty weight 1.056 the minimizers are [1]
    #1 component became optimal
    
    
    # in the GMM we only have 40 test assets so let's see what's the 
    # optimal number of factors when kmax,mmax<=39
    
    minimizeic(macroagg,39,icbaing_1,1,pic = False)
    # result = 1
    
    minimizeic(macroagg,39,icbaing_2,1,pic = False)
    # result = 1
    
    minimizeic(macroagg,39,icbaing_3,1,pic = False)
    # result = 1
    
    minimizepc(macroagg,39,pcbaing_1_scaling,lam = 1, pic = False)
    # result = 30
    
    minimizepc(macroagg,39,pcbaing_2_scaling,lam = 1, pic = False)
    # result = 30
    
    minimizepc(macroagg,39,pcbaing_3_scaling,lam = 1, pic = False)
    # result = 39
    
    # we look for robustness of the 30 factors out of 39 result by varying the penalty weight
    
    # do PC1 optimization with scaling from 0.5 to 1.5
    pcprint(macroagg,pcbaing_1_scaling,39,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [39]
    #With penalty weight 0.611 the minimizers are [39]
    #With penalty weight 0.722 the minimizers are [39]
    #With penalty weight 0.833 the minimizers are [39]
    #With penalty weight 0.944 the minimizers are [30]
    #With penalty weight 1.056 the minimizers are [30]
    #With penalty weight 1.167 the minimizers are [30]
    #With penalty weight 1.278 the minimizers are [30]
    #With penalty weight 1.389 the minimizers are [30]
    #With penalty weight 1.5 the minimizers are [1]
    #1 component became optimal
    
    # do PC2 optimization with scaling from 0.5 to 1.5
    pcprint(macroagg,pcbaing_2_scaling,39,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [39]
    #With penalty weight 0.611 the minimizers are [39]
    #With penalty weight 0.722 the minimizers are [39]
    #With penalty weight 0.833 the minimizers are [39]
    #With penalty weight 0.944 the minimizers are [30]
    #With penalty weight 1.056 the minimizers are [30]
    #With penalty weight 1.167 the minimizers are [30]
    #With penalty weight 1.278 the minimizers are [30]
    #With penalty weight 1.389 the minimizers are [1]
    #1 component became optimal
    
    # do PC3 optimization with scaling from 0.5 to 1.5
    pcprint(macroagg,pcbaing_3_scaling,39,0.5,1.5,step = None,num = 10)
    #With penalty weight 0.5 the minimizers are [39]
    #With penalty weight 0.611 the minimizers are [39]
    #With penalty weight 0.722 the minimizers are [39]
    #With penalty weight 0.833 the minimizers are [39]
    #With penalty weight 0.944 the minimizers are [39]
    #With penalty weight 1.056 the minimizers are [39]
    #With penalty weight 1.167 the minimizers are [30]
    #With penalty weight 1.278 the minimizers are [30]
    #With penalty weight 1.389 the minimizers are [30]
    #With penalty weight 1.5 the minimizers are [30]

# create a version of the PCA factors that is standardscaled, so that we can get 
# more meaningful coefficients on the data

def produce_macro_factors():
    
    macro = pd.read_pickle(f'{settings.datapath}/int/macro_clean.pkl')
    CP08 = pd.read_pickle(f'{settings.datapath}/int/Cochrane-Piazessi_2008.pkl')
    for i in [7,8,9,45,30]:
        compo1 = pcask(macro.drop(columns = ['date']),i)[0]
        macro_pca = macropcafactors(macro,compo1)
        macro_factors_newcp = macro_pca.merge(CP08, on = ['date'], how = 'inner')
        title = f'{settings.datapath}/int/macro_factors_PCA_{i}_CP08_scaled.pkl'
        macro_factors_newcp.to_pickle(title)


def preprocess_ejnports(nr_facs):
    '''
    ###################### Preprocess the testportfolios ########################
    '''
    os.chdir(f'{settings.datapath}/raw/BondPortfoliosNozawa/Fwd__Corporate_Bond_Data')
    
    downside = pd.read_csv('Downside_Sorted_Portfolio5_c0.csv')
    hkm = pd.read_csv('HKM_Sorted_Portfolio5_c0.csv')
    idiovol = pd.read_csv('Idiovol_Sorted_Portfolio5_c0.csv')
    maturity = pd.read_csv('Maturity_Sorted_Portfolio5_c0.csv')
    rating = pd.read_csv('Rating_Sorted_Portfolio5_c0.csv')
    reversal = pd.read_csv('Reversal_Sorted_Portfolio5_c0.csv')
    spread = pd.read_csv('Spread_Sorted_Portfolio10_c0.csv')
    
    portolist = {'downside':downside,'idiovol':idiovol,'hkm':hkm,'maturity':maturity,
                 'rating':rating,'reversal':reversal,'spread':spread}
    
    #get_df_nam(spread)
    # change names of columns of the files with portfolios
    for name, df in portolist.items():
        new = {}
        listcol = [x for x in df.columns if x != 'date']
        for i, col in enumerate(listcol):
            new[col] = f"{name}_{i+1}"
        df.rename(columns = new, inplace = True)
    
    # merge into one df the portfolios (without dropping missing values which are denoted here by -9999)
    notdown = list(portolist.values())[1:]
    testass = downside
    for porto in notdown:
        testass = testass.merge(porto, on = 'date', how = 'inner')
    
    # modify the dates as we need them
    testass['day'] = 1
    testass['date'] = pd.to_datetime(testass['date'], format = "%Y%m%d")
    testass['year'] = testass['date'].dt.year
    testass['month'] = testass['date'].dt.month
    testass['date'] = pd.to_datetime(testass[['year','month','day']], format = "%Y%m%d")
    testass.drop(columns = ['day','year','month'], inplace = True)
    
    #os.chdir(f'{path}/raw/')
    testass.to_pickle(f'{settings.datapath}/raw/portfolios_nozawa_raw.pkl')
     
    ## create file with test portfolios where the portfolios are in the columns   
    testport = pd.DataFrame(columns = ['date','port_id','return'])
    dates = testass['date'].tolist()
    cols = [x for x in testass.columns if x != 'date']
    
    #testport['port_id'] = 0
    for d in dates:
        for col in cols:
            if testass.loc[testass['date'] == d, [col]].values[0] != -9999:
                #print(testass.loc[testass['date'] == d, [col]].values[0])
                val = testass.loc[testass['date'] == d, [col]].values[0]
                #testport = pd.concat([testport,pd.DataFrame({'date':d,'port_id':col,'return':val[0]})], axis = 0, ignore_index=True)
                testport = pd.concat([testport,pd.DataFrame(np.array([d,col,val[0]]).reshape(1,-1), columns = testport.columns)], axis = 0, ignore_index=True)
    
    # modify the dates as we need them          
    testport['date'] = pd.to_datetime(testport['date'], format = "%Y%m%d")
    testport['year'] = testport['date'].dt.year
    testport['month'] = testport['date'].dt.month
    testport.rename(columns = {'port_id':'asset_id'},inplace = True)
    
    vrf = pd.read_pickle(f'{settings.datapath}/raw/rf_french.pkl').rename(columns={"RF":"rf_rate"})
    testport = testport.merge(vrf, on = ['year','month'], how = 'inner')
    testport['ret_e'] = testport['return'] - testport['rf_rate'].apply(lambda x:x * 100)
    # note, we haven't changed vrf file here
    
    testport['day'] = 1
    testport = testport[['date','asset_id','ret_e']]
    # save file 
    testport.to_pickle(f'{settings.datapath}/int/Nozawaport_excessret.pkl')
    
    ## how many dates with all portfolios? save them as a separate file
    # this file will be the one we use for the GMM procedure
    intersection = testass[~testass.isin([-9999]).any(axis = 1)]
    #intersection.to_pickle('portfolios_nozawa_intersection.pkl')
    
    # see what happens with the dates where a portfolio is missing
    missing = testass[testass.isin([-9999]).any(axis = 1)]
    # find dates where ALL values are missing
    missingdates = []
    assets = [x for x in missing.columns if x != 'date']
    dates = missing['date']
    for d in dates:
        if missing.loc[missing['date'] == d, assets].nunique(axis = 1).item()<=1:
            missingdates.append(d)
    print('The missing dates with no portfolio returns are', missingdates)
    ## there are four completely missing dates!!        
    
    ## calculate the excess returns for the nozawa portfolios in the intersection
    vrf['day'] = 1
    vrf['date'] = pd.to_datetime(vrf[['year','month','day']], format = "%Y%m%d")
    vrf.drop(columns = ['year','month','day'],inplace = True)
    intersection = intersection.merge(vrf, on = 'date', how = 'inner')
    lis = [x for x in intersection.columns if x not in {'date','rf_rate'}]
    
    intersection['rf_rate'] = intersection['rf_rate'].apply(lambda x:x * 100)
    intersection[lis] = intersection[lis].sub(intersection['rf_rate'], axis = 0)
    
    # drop rf_rate column
    intersection.drop(columns = 'rf_rate',inplace = True)
    
    # save resulting file (this is the file we will use for GMM)
    intersection.to_pickle(f'{settings.datapath}/int/portfolios_nozawa_intersection_eret.pkl')
    ## prepare the portfolio returns and the gmm files that we will use

    ## register the dates of the gmm
    gmm_dates = intersection['date']
    gmm_dates.to_pickle(f'{settings.datapath}/int/gmm_dates.pkl')
    
    
    '''
    Create final datasets that will be used to run GMM
    They have the portfolio returns, the macro factors and the CP08 factor
    We also look at versions without CP08 factor, as well as a version with CP08 only
    '''
    
    for i in nr_facs:
        if i>0:
            macro_factors = pd.read_pickle(f'{settings.datapath}/int/macro_factors_PCA_{i}_CP08_scaled.pkl')
            gmm = intersection.merge(macro_factors, on = 'date', how = 'inner')
            gmm.to_pickle(f'{settings.datapath}/int/GMM/GMM_Data_PCA_{i}_CP08_1.pkl')
            gmm.drop(columns = 'CP08', inplace = True)
            gmm.to_pickle(f'{settings.datapath}/int/GMM/GMM_Data_PCA_{i}_CP08_0.pkl')
            
        else:
            macro_factors = pd.read_pickle(f'{settings.datapath}/int/macro_factors_PCA_{7}_CP08_scaled.pkl')
            gmm = intersection.merge(macro_factors, on = 'date', how = 'inner')
            columns = gmm.columns
            gmm.drop(columns = [c for c in columns if c.startswith('P')], inplace = True)
            gmm.to_pickle(f'{settings.datapath}/int/GMM/GMM_Data_PCA_{0}_CP08_1.pkl')
    
    assert sum(checkfornans())==0



if __name__=='__main__':
    
    construct_cp_factor_clean_macro()
    bai_ng_analysis()
    '''
    ### We see overall that 45 factors is robust for our dataset if we include 82 time series
    which explain 99% of the variance. 
    If we confine to 39 factors, which is the maximal number allowed by
    the number of test-assets, we roughly get optimality of 30 factors.
    ### Hence, we will run GMM with 7,8,9 and 30 macro pca factors
    '''
    preprocess_ejnports(nr_facs = [0,7,8,9,30,45])
    produce_macro_factors()

