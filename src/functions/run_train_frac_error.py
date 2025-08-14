import sys
sys.path.insert(1,'~/Documents/TabPFN evaluation/src/')
import MICE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime


def train_fraction_error(df, train_frac, path):
    name = 'train_fraction_error'
    now = datetime.datetime.now()
    clock_time = str(now.time())
    print(path +'/' + clock_time)
    Path(path +'/' + clock_time).mkdir()
    path_local = path + '/' + clock_time +'/' 
    error_mean = []
    error_mice = []
    error_KNN = []
    error_tabPFN = []
    error_XGBoost = []
    error_Catboost = []
    error_lightGBM = []
    error_random_forest = []
    N_mean = []
    N_mice = []
    N_KNN = []
    N_tabPFN = []
    N_XGBoost = []
    N_Catboost = []
    N_lightGBM = []
    N_random_forest = []
    for i in range(len(train_frac)):
        Path(path_local+'train_frac_'+str(train_frac[i])).mkdir(exist_ok=True)
        path4sim = path_local+'train_frac_'+str(train_frac[i])+'/'
        manager = MICE.manager_blood(df, 0.05, True)
        #manager.print_parameters(path)
        manager.compare_row_analysis(train_frac[i])
        manager.print_parameters(path4sim)
        manager.store.print_predictions_csv(path4sim)
        manager.store.save_models(path4sim)
        manager.store.save_train_rows(path4sim)
        error_mean.append(manager.store.mean_error)
        error_mice.append(manager.store.mice_error)
        error_KNN.append(manager.store.KNN_error)
        error_tabPFN.append(manager.store.tabPFN_error)
        error_XGBoost.append(manager.store.XGB_error)
        error_Catboost.append(manager.store.Cat_error)
        error_lightGBM.append(manager.store.lightGBM_error)
        error_random_forest.append(manager.store.random_forest_error)
        N_mean.append(manager.store.mean_N)
        N_mice.append(manager.store.mice_N)
        N_KNN.append(manager.store.KNN_N)
        N_tabPFN.append(manager.store.tabPFN_N)
        N_XGBoost.append(manager.store.XGB_N)
        N_Catboost.append(manager.store.Cat_N)
        N_lightGBM.append(manager.store.lightGBM_N)
        N_random_forest.append(manager.store.random_forest_N)


    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(N_mean, error_mean, label = 'mean')
    ax.scatter(N_mice, error_mice, label = 'mice')
    ax.scatter(N_KNN, error_KNN, label = 'KNN')
    ax.scatter(N_tabPFN, error_tabPFN, label = 'tabPFN')
    ax.scatter(N_XGBoost, error_XGBoost, label='XGBoost')
    ax.scatter(N_Catboost,error_Catboost, label='Catboost')
    ax.legend()
    ax.set_xlabel('Row used for train')
    ax.set_ylabel('RMSE')
    fig.savefig(path_local + 'test.png')

    

    


