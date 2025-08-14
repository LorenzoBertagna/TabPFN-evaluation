from functions.run_train_frac_error import train_fraction_error
from functions.sim_error_time import run_error_time
from functions.run_error_column import run_error_column
from functions.run_feature_matrix import run_feature_matrix_blood_count
from functions.run_feature_matrix import run_feature_matrix_biochemistry
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import torch
torch.cuda.empty_cache()

if __name__ == "__main__":
    today_date = datetime.today().strftime('%Y-%m-%d')
    path='../results/cluster/' + today_date
    Path(path).mkdir(exist_ok = True)
    df = pd.read_csv("file_source/UKB_Patients_blood_copy.csv", sep=';')
    df = df.replace({pd.NA: np.nan})
    df = df.rename(columns={'sex': 'Sex', 'age':'Age'}


    #train_fraction_error(df, train_frac, path)
    #run_error_time(df, path)
    run_feature_matrix_blood_count(df,0.1,path,20)
  



