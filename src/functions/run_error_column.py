import numpy as np
import pandas as pd
import sys
sys.path.insert(1,'~/Documents/TabPFN evaluation/src/')
from visualization.visualize_distribution import manager_visualize
from pathlib import Path
import datetime
import matplotlib.pyplot as plt


def run_error_column(df, nan_frac, path):
    name = 'error_column'
    now = datetime.datetime.now()
    clock_time = str(now.time())
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
    columns = []
    manager4column = manager_visualize(df,'sex',nan_frac)
    for column in manager4column.df_blood_count:
        Path(path_local+str(column)).mkdir(exist_ok=True)
        path4sim = path_local+str(column)+'/'
        manager = manager_visualize(df,column,nan_frac)
        manager.compare()
        manager.print_parameters(path4sim)
        manager.store.print_predictions_csv(path4sim)
        manager.store.save_models(path4sim)
        manager.store.save_rmse(path4sim)
        error_mean.append(manager.store.mean_error)
        error_mice.append(manager.store.mice_error)
        error_KNN.append(manager.store.KNN_error)
        error_tabPFN.append(manager.store.tabPFN_error)
        error_XGBoost.append(manager.store.XGB_error)
        error_Catboost.append(manager.store.Cat_error)
        error_lightGBM.append(manager.store.lightGBM_error)
        error_random_forest.append(manager.store.random_forest_error)
        columns.append(column)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(columns, error_mean, label = 'mean')
    ax.scatter(columns, error_mice, label = 'mice')
    ax.scatter(columns, error_KNN, label = 'KNN')
    ax.scatter(columns, error_tabPFN, label = 'tabPFN')
    ax.scatter(columns, error_XGBoost, label='XGBoost')
    ax.scatter(columns, error_Catboost, label='Catboost')
    ax.scatter(columns, error_random_forest, label='random forest')
    ax.legend()
    ax.set_xlabel('Columns')
    ax.set_xticklabels(columns,rotation=90)
    ax.set_ylabel('RMSE')
    fig.savefig(path_local + 'columns.png')

