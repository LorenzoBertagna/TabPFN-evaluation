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
from numba import cuda


class manager_visualize(manager):
    def __init__(self, df, col_name, nan_frac):
        # Drop the eids
        df = df.drop('eid', axis=1)
        # Split df
        self.age = df[['Age']]
        self.sex = df[['Sex']]
        self.df = df.drop(['Age', 'Sex'], axis=1)

        self.df_blood_count, self.df_biochemistry = self.split_df()

        self.df_biochemistry = pd.concat([self.df_biochemistry, self.age, self.sex], axis=1)
        self.df_blood_count = pd.concat([self.df_blood_count, self.age, self.sex], axis=1)
        # Drop two columns that have most zeros (catboost has got problems there...)
        self.df_blood_count = self.df_blood_count.drop(['Nucleated red blood cells', 'Nucleated red blood cell (%)'], axis=1)

        # Consider just complete data        
        self.df_blood_count  = self.df_blood_count.dropna()
        self.df_biochemistry = self.df_biochemistry.dropna()

        # Parameters
        self.col_start = 0
        self.col_end = -1
        self.nan_frac = nan_frac
        self.train_size = 0.01
        self.col = self.df_blood_count.columns.get_loc(col_name)
        self.col_sex = self.df_biochemistry.columns.get_loc('sex')

        # Create a train and a test dataframe
        self.train, self.test = train_test_split(self.df_blood_count, train_size=self.train_size, random_state=19)

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

    def tabPFN_shapPFN(self):
        method = tabPFN_Regressor()
        trainmask_copy = self.trainmask
        testmask_copy = self.testmask
        method.shapPFN(trainmask_copy, testmask_copy, 0.1, self.col)


    def mask2visualize(self,  df, frac, col):
        dflocal = df.copy()
        ix = [row for row in range(dflocal.shape[0])]
        for row in random.sample(ix, int(round(frac * len(ix)))):
            dflocal.iat[row, col] = np.nan
        return dflocal

    def mask_oestradiolMAR(self, df, col, col_sex):
        dflocal = df.copy()
        for i in range(dflocal.shape[0]):
            if dflocal.iloc[i, col_sex] == 1:
                mybool = np.random.choice([0,1], 1, p=[0.5, 0.5])
                if mybool:
                    dflocal.iloc[i, col] = np.nan
            else:
                mybool = np.random.choice([0,1], 1, p=[0.8,0.2])
                if mybool:
                    dflocal.iloc[i, col] = np.nan

        return dflocal

    def mask_oestradiolMNAR(self, df, col):
        dflocal = df.copy()
        col2impute = df.iloc[:,col].values
        lower_q = np.quantile(col2impute, 0.05)
        upper_q = np.quantile(col2impute, 0.90)
        for i in range(dflocal.shape[0]):
            key = dflocal.iloc[i,col]
            if key >= upper_q or key <=lower_q:
                mybool = np.random.choice([0, 1], 1, p=[0.5, 0.5])
                if mybool:
                    dflocal.iloc[i, col] = np.nan
        return dflocal



    def add_random(self):
        dim = self.df_blood_count.shape
        random_col = np.random.normal(0, 1, dim[0])
        new_df = self.df_blood_count
        new_df['Normal'] = random_col
        self.df_blood_count = new_df
        return new_df

    # Scatter plots + box plots
    def plot_scatter(self):
        labels = ['mean + real','mice', 'KNN', 'XGBoost', 'Catboost', 'TabPFN']
        '''mean_x = []
        mice_x = []
        KNN_x = []
        XGB_x = []
        cat = []
        tabPFN_x = []
        for i in range(len(self.store.mean_pred)):
            mean_x.append('mean')
            mice_x.append('mice')
            KNN_x.append('KNN')
            XGB_x.append('XGBoost')
            cat.append('Catboost')
            tabPFN_x.append('TabPFN')
        '''
        # Impute to the correct x value the corresponding predicted values
        mean_x = [1] * len(self.store.mean_pred)
        mice_x = [2] * len(self.store.mice_pred)
        KNN_x = [3] * len(self.store.KNN_pred)
        XGB_x = [4] * len(self.store.XGB_pred)
        cat_x = [5] * len(self.store.Cat_pred)
        tabPFN_x = [6] * len(self.store.tabPFN_pred)


        fig, ax = plt.subplots()

        # Scatter plots
        plt.scatter(mean_x, self.store.mean_pred,zorder=3, s=10)
        plt.scatter(mice_x, self.store.mice_pred, zorder=3, s=10)
        plt.scatter(KNN_x, self.store.KNN_pred, zorder=3, s =10)
        plt.scatter(XGB_x, self.store.XGB_pred, zorder=3, s =10)
        plt.scatter(cat_x, self.store.XGB_pred, zorder=3, s =10)
        plt.scatter(tabPFN_x, self.store.XGB_pred, zorder=3, s =10)
        plt.xlabel('methods')
        ax.set_ylabel('col values')
        plt.title('Scatter of the predictions and box plot of the real distributions for different imputation methods -'
                  'used column: 19')

        # Predicted values for the corresponding column
        col_values = [self.store.real, self.store.mice_pred, self.store.KNN_pred, self.store.XGB_pred,
            self.store.Cat_pred,self.store.tabPFN_pred]
        colors = ['gray', 'lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray']

        # Boxplots
        bplot = ax.boxplot(col_values, patch_artist=True, tick_labels=labels)  # will be used to label x-ticks

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        for median in bplot['medians']:
            median.set_color('black')

        plt.show()

    # Line plots of predicted and real values
    def plot_line_error(self):
        # Labels of methods
        labels = ['\n\nmean','\n\nmice', '\n\nKNN', '\n\nXGBoost', '\n\nCatboost', '\n\nTabPFN']
        # Get predicted values
        y_real = self.store.real
        y_cat = self.store.Cat_pred
        y_XGB = self.store.XGB_pred
        y_tabPFN = self.store.tabPFN_pred
        y_mean = self.store.mean_pred
        y_KNN = self.store.KNN_pred
        y_mice = self.store.mice_pred
        # Where to use the labels
        x_ticks = [1.5,3.5, 5.5, 7.5, 9.5, 11.5]
        fig, ax = plt.subplots()
        sec = ax.secondary_xaxis(location=0)
        sec.set_xticks(x_ticks, labels=labels)
        sec.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12], labels=['real', 'pred', 'real', 'pred', 'real', 'pred',
                                                             'real', 'pred', 'real', 'pred', 'real', 'pred'])
        # Plot all predicted values
        for i in range(len(self.store.real)):
            plt.plot([9,10], [y_real[i], y_cat[i]], color='b')
            plt.plot([7,8], [y_real[i], y_XGB[i]], color='g')
            plt.plot([3,4], [y_real[i], y_mice[i]], color='r')
            plt.plot([5,6], [y_real[i], y_KNN[i]], color='c')
            plt.plot([1,2], [y_real[i], y_mean[i]], color='y')
            plt.plot([11,12], [y_real[i],y_tabPFN[i]], color='orange')


        plt.xticks(x_ticks, labels=labels)
        plt.ylabel('Real and pred')
        plt.title('Line plots of real and predicted values for different methods - col ' + str(self.train.columns[self.col]))
        plt.show()


def plot_error(df):
    error_mean = []
    error_mice = []
    error_KNN = []
    error_tabPFN = []
    error_XGBoost = []
    error_Catboost = []
    for j in range(5):
        manager = manager_visualize(df)
        manager.compare()
        error_mean.append(manager.store.mean_error)
        error_mice.append(manager.store.mice_error)
        error_KNN.append(manager.store.KNN_error)
        error_tabPFN.append(manager.store.tabPFN_error)
        error_XGBoost.append(manager.store.XGB_error)
        error_Catboost.append(manager.store.Cat_error)

    labels = ['Sim1', 'Sim2', 'Sim3', 'Sim4', 'Sim5']
    plt.plot(labels, error_mean, marker='s', label='mean')
    plt.plot(labels, error_KNN, marker='s', label='KNN')
    plt.plot(labels, error_mice, marker='s', label='mice')
    plt.plot(labels, error_tabPFN, marker='s', label='tabPFN')
    plt.plot(labels, error_XGBoost, marker='s', label='XGBoost')
    plt.plot(labels, error_Catboost, marker='s', label='Catboost')
    plt.title('Error of different models in imputing Oestradiol (MNAR)')
    plt.xlabel('Simulations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("~/Documents/TabPFN evaluation/src/file_source/UKB_Patients_blood_copy.csv", sep=';')
    df = df.replace({pd.NA: np.nan})
    df = df.rename(columns={'sex': 'Sex', 'age':'Age'}
    manager_visualize = manager_visualize(df, 'Oestradiol', 0.1)
    #plot_error(df)
    #manager_visualize.compare()
    manager_visualize.tabPFN_shapPFN()
    #manager_visualize.plot_line_error()

