import sys
sys.path.insert(1,'~/Documents/TabPFN evaluation/src/')
import MICE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import torch
#torch.cuda.empty_cache()


def run_error_time(df, path):
    name = 'train_fraction_error'
    now = datetime.datetime.now()
    clock_time = str(now.time())
    print(path +'/' + clock_time)
    Path(path +'/' + clock_time).mkdir()
    path_local = path + '/' + clock_time +'/'

    #df = pd.read_csv("file_source/UKB_Patients_blood_copy.csv", sep=';')
    #df = df.replace({pd.NA: np.nan})

    for i in range(4):
        torch.cuda.empty_cache()
        Path(path_local+'sim' + str(i)).mkdir(exist_ok=True)
        path4sim = path_local + 'sim' + str(i)+ '/'

        manager = MICE.manager_blood(df, 0.05, True)
        manager.print_parameters(path4sim)
        manager.compare()
        manager.store.print_predictions_csv(path4sim)
        manager.store.save_models(path4sim)
        manager.store.save_runtimes(path4sim)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(manager.store.mean_error, manager.store.mean_time, label = 'mean')
        ax.scatter(manager.store.mice_error, manager.store.mice_time, label = 'mice')
        ax.scatter(manager.store.KNN_error, manager.store.KNN_time, label = 'KNN')
        ax.scatter(manager.store.tabPFN_error, manager.store.tabPFN_time, label = 'tabPFN')
        ax.scatter(manager.store.Cat_error, manager.store.Cat_time, label='catboost')
        ax.scatter(manager.store.XGB_error, manager.store.XGB_time, label='XGBoost')
        ax.legend()
        ax.set_xlabel('RMSE')
        ax.set_ylabel('Time in s')
        fig.savefig(path4sim + 'RMSE and time simulation ' + str(i)+ '.png')

