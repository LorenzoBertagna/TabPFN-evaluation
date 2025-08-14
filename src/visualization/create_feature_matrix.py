import pandas as pd
import sys
from os import path
sys.path.append(path.abspath('../'))
from MICE import manager
from sklearn.model_selection import train_test_split
import numpy as np
import random
from MICE import store
import matplotlib.pyplot as plt
from MICE import tabPFN_Regressor
from MICE import XGBoost
from numba import cuda


class manager_feature_matrix(manager):
    def __init__(self, df, col_name, nan_frac, blood_count, col2drop_blood_count, col2drop_biochemistry):
        # Drop the eids
        df = df.drop('eid', axis=1)
        # Split df
        self.age = df[['Age']]
        self.sex = df[['Aex']]
        self.df = df.drop(['Age', 'Sex'], axis=1)

        self.df_blood_count, self.df_biochemistry = self.split_df()

        self.df_biochemistry = pd.concat([self.df_biochemistry, self.age, self.sex], axis=1)
        self.df_blood_count = pd.concat([self.df_blood_count, self.age, self.sex], axis=1)
        # Drop two columns that have most zeros (catboost has got problems there...)
        self.df_blood_count = self.df_blood_count.drop(col2drop_blood_count, axis=1)
        self.df_biochemistry = self.df_biochemistry.drop(col2drop_biochemistry, axis=1)

        # Consider just complete data        
        self.df_blood_count  = self.df_blood_count.dropna()
        self.df_biochemistry = self.df_biochemistry.dropna()

        # Parameters
        self.col2drop_blood_count = col2drop_blood_count
        self.col2drop_biochemistry = col2drop_biochemistry
        self.col_start = 0
        self.col_end = -1
        self.nan_frac = nan_frac
        self.train_size = 0.01
        
       # Store which df we are imputing
        self.blood_count = blood_count
 
        # Create a train and a test dataframe
        if blood_count:
            self.col = self.df_blood_count.columns.get_loc(col_name)
            self.train, self.test = train_test_split(self.df_blood_count, train_size=self.train_size, random_state=19)
        else:
            self.col = self.df_biochemistry.columns.get_loc(col_name)
            self.train, self.test = train_test_split(self.df_biochemistry, train_size=self.train_size, random_state=19)


        # Update df depending on which columns we want to consider (i.e. col_start and col_end)
        if self.col_end != -1:
            self.test = self.test.iloc[:, self.col_start:self.col_end]
            self.train = self.train.iloc[:, self.col_start:self.col_end]

        # Mask in the train
        #self.trainmask = self.mask(pd.DataFrame(self.train), self.nan_frac)
        self.trainmask = self.train.copy()

        # Mask in the data
        self.testmask = self.mask2visualize(self.test, self.nan_frac, self.col)
        if self.col_end != -1:
            self.testmask = self.testmask.iloc[:, self.col_start:self.col_end]

        # Track where it has been masked
        self.testtrack = self.track(self.testmask)
        if self.col_end != -1:
            self.testtrack = self.testtrack.iloc[:, self.col_start:self.col_end]

        # Store results in the class store - assign it later
        self.store = store()

        real = pd.DataFrame.to_numpy(self.test)
        self.store.real = real[self.testtrack]

        # Name of method we use - we assign it later
        self.name = None

    def tabPFN_shap(self):
        method = tabPFN_Regressor()
        trainmask_copy = self.trainmask.iloc[:,:6]
        testmask_copy = self.testmask.iloc[:,:6]
        method.shapiq(trainmask_copy, testmask_copy, 0.1, self.col)

    def tabPFN_shapPFN(self, n):
        method = tabPFN_Regressor()
        trainmask_copy = self.trainmask
        testmask_copy = self.testmask
        method.shapPFN(trainmask_copy, testmask_copy, 0.1, self.col, n)
        return method.feature_array, method.model

    def XGBoost_shap(self,n):
        method = XGBoost()
        trainmask_copy = self.trainmask
        testmask_copy = self.testmask
        method.XGB_shap(trainmask_copy, testmask_copy, 0.1, self.col, n)
        return method.feature_array, method.model



    def mask2visualize(self,  df, frac, col):
        dflocal = df.copy()
        ix = [row for row in range(dflocal.shape[0])]
        for row in random.sample(ix, int(round(frac * len(ix)))):
            dflocal.iat[row, col] = np.nan
        return dflocal




if __name__ == "__main__":
    df = pd.read_csv("~/Documents/TabPFN evaluation/src/file_source/UKB_Patients_blood_copy.csv", sep=';')
    df = df.replace({pd.NA: np.nan})
    df = df.rename(columns={'sex': 'Sex', 'age':'Age'}
    col2dropblood = ['Nucleated red blood cells', 'Nucleated red blood cell (%)'] 
    col2dropbiochemistry = ['Rheumatoid factor', 'Oestradiol']
    manager_visualize = manager_visualize(df, 'MCV', 0.1,1, col2dropblood, col2dropbiochemistry)
    #plot_error(df)
    #manager_visualize.compare()
    feature_importance = manager_visualize.tabPFN_shapPFN(10)
    #manager_visualize.plot_line_error()

