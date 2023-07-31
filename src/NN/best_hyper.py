#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:54:45 2022


'''
best_hyper fetches best hyperparameters for a given validation criterion from saved result files. 
These are fed then to the ensembling with multiple distinct seeds to create the ensemble members.
It also produces a dataframe that lists all val_test results of all trained models in the folder.
'''

@author: jed169
"""

from settings import savepath as savepth

import sys
import json
import pandas as pd
import glob
import time
import argparse

start_time = time.time()

date_idx = 'date'

def df_from_list(filenames, axis = 0):
    
    dfs = [pd.read_pickle(name) for name in filenames]
    df = pd.concat(dfs, axis)
    return df


parser = argparse.ArgumentParser(description  = 'Arguments for fetching best')
parser.add_argument('-proj','--project_name', type = str, default = 'other_sim', help = 'Project name')
parser.add_argument('-val_crit','--val_criterion', type = str, default = 'sharpe_ratio', help = 'Criterion to choose models from. Needs to match ')
parser.add_argument('-jobnr','--jobnumber', type = int, default = 10, help = 'job number when submitting multiple jobs to the cluster')

def main():
    
    args = parser.parse_args()
    
    if args.val_criterion == 'sharpe_ratio': #for sharpe ratio pick the highest
        ascending = False
    else:
        ascending = True
    
    savepath=f'{savepth}/{args.project_name}'
    FilenamesList = glob.glob(f"{savepath}/{args.project_name}_Model_*_{args.val_criterion}_jnr_{args.jobnumber}_*.pkl")
    
    hyperparams_dict={}
    performance_dict = {}
    for name in FilenamesList:
        df = pd.read_pickle(name)
        name = name.replace(f"{savepath}/{args.project_name}_","").replace(f"{args.val_criterion}_","").replace(".pkl","").replace("'","").replace(f"_jnr_{args.jobnumber}","")
        split = name.split("_")

        hyperparams_dict[name] = dict(zip(split[::2],split[1::2]))
        name = '_'.join(map(str,split))
        performance_dict[name]=df.loc[(df['phase']=='val') & (df['epoch']==df['epoch'].max()),args.val_criterion].values[0]
                
    df = pd.DataFrame.from_dict(performance_dict, orient = 'index', columns = [f'{args.val_criterion}']).reset_index().sort_values(by = f'{args.val_criterion}', 
                                                                ascending = ascending).rename(columns = {'index':'hyperparams'})
    df = df.reset_index().drop(columns = ['index'])
    df.to_pickle(f"{savepath}/{args.project_name}_val_results.pkl")
    df.to_excel(f"{savepath}/{args.project_name}_val_results.xlsx")
    with open(f"{savepath}/{args.project_name}_hyperparams.json", "w") as outfile:
        json.dump(hyperparams_dict, outfile)
        
    best_hyper = df.iloc[0,0].split("_")
    best_hyper = dict(zip(best_hyper[::2],best_hyper[1::2]))
    best_hyper['modelnr'] = best_hyper['Model']
    del best_hyper['Model']
    best_hyper['b'] = best_hyper['bs']
    del best_hyper['bs']
    del best_hyper['rs']
    
    if args.project_name.startswith("GAN"):
        l1regs = best_hyper["l1regs"].replace("(","").replace(")","").split(",")
        lrs = best_hyper["lrs"].replace("(","").replace(")","").split(",")
        for tup in [l1regs,lrs]:
            if tup[0]=='none':
                tup[0]=0.0
            if tup[1]=='none':
                tup[1]=0.0
            tup = (float(tup[0]),float(tup[1]))
            
        best_hyper['l1reg_sdf'], best_hyper['l1reg_mom'] = l1regs[0], l1regs[1]
        best_hyper['lr_sdf'], best_hyper['lr_mom'] = lrs[0], lrs[1]
        del best_hyper["lrs"], best_hyper["l1regs"]
                
    intparams = ['b', 'modelnr']
    
    best_hyper_string = ""
    for key, val in best_hyper.items():
        best_hyper_string+=f"-{key} "
        if key in intparams:
            best_hyper_string+=f"{int(val)} "
        else:
            best_hyper_string+=f"{val} "

    return best_hyper_string

if __name__ == '__main__':
    best = str(main())
    sys.stdout.write(best)
    sys.exit(0)
    
    
