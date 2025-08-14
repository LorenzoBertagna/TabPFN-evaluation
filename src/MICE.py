import pandas as pd
import numpy as np
import random
import joblib
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from math import sqrt
import multiprocessing
import sys
from joblib import Parallel, delayed
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
#from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
import timeit
import shapiq
import os # Somehow this doesn't give me the error
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WORLD_SIZE"] = "1"
import torch
from random import randrange
from IPython.display import display
from pathlib import Path
from tabpfn_extensions import interpretability
#from Import import import_age_sex
from numba import cuda
import matplotlib.pyplot as plt
import shap


# Class for vanilla Mice (i.e. we impute the data with the mean)
class mean:
    def __init__(self):
        self.name = 'Mean'
        self.N = None
        self.model = None
    # Impute the mean
    def predict(self, train, test):
        start = timeit.default_timer()
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # Train the imputer
        imp.fit(train)
        # Impute the test
        newdf = imp.transform(test)
        self.model = imp
        stop = timeit.default_timer()
        time = stop-start
        print('Time: ', time)
        return newdf, time

    # Impute the mean - train size analysis
    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        train_copy = train.copy()
        test_copy = test.copy()
        train_copy = train_copy.sample(frac=frac)
        self.N = train_copy.shape[0]
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # Train the imputer
        imp.fit(train_copy)
        # Impute the test
        newdf = imp.transform(test_copy)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return newdf

    # Plot the values of a coloumn and its mean
    def plot_vanilla_mean(self, dftr, mean, col):
        colTr = dftr[dftr.columns[col]]
        colTr = colTr.to_numpy()
        x = np.ones(len(colTr))
        plt.scatter(x, colTr)
        plt.scatter(1, mean, color='red')
        plt.show()

    # Find the value assigned from the scikit imputation
    def find_mean(self, df, dfnan, col):
        mean = -1
        for i in range(len(df)):
            if dfnan.iloc[i, col]:
                mean = df[i, col]
                break
        return mean

# Multivariate Mice
class multi_mice():
    def __init__(self):
        self.name = 'Multi MICE'
        self.N = None
        self.model = None

    def predict(self, train, test):
        start = timeit.default_timer()
        imp = IterativeImputer(max_iter=50, random_state=0)
        # Train the imputer
        imp.fit(train)
        # Impute
        newdf = imp.transform(test)
        self.model = imp
        stop = timeit.default_timer()
        time = stop -start
        print('Time: ', time)
        return newdf, time

    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        train_copy = train.copy()
        test_copy = test.copy()
        train_copy = train_copy.sample(frac=frac)
        self.N = train_copy.shape[0]
        imp = IterativeImputer(max_iter=50, random_state=0)
        # Train the imputer
        imp.fit(train_copy)
        # Impute
        newdf = imp.transform(test_copy)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return newdf

# KNN imputer (is it EM with fixed weights?)
class KNNimputer():
    def __init__(self):
        self.name = 'KNN imputer'
        self.N = None
        self.model = None

    def predict(self, train, test):
        start = timeit.default_timer()
        imp = KNNImputer(n_neighbors=2, weights="uniform")
        # Train the imputer
        imp.fit(train)
        newdf = imp.transform(test)
        self.model = imp
        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)
        return newdf, time

    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        train_copy = train.copy()
        test_copy = test.copy()
        train_copy = train_copy.sample(frac=frac)
        self.N = train_copy.shape[0]
        imp = KNNImputer(n_neighbors=2, weights="uniform")
        # Train the imputer
        imp.fit(train_copy)
        newdf = imp.transform(test_copy)
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return newdf

# Code inspired from: https://medium.com/@tzhaonj/imputing-missing-data-using-xgboost-802757cace6d
# XGBoost method
class XGBoost:
    def __init__(self):
        self.name = 'XGBoost'
        self.N = None
        self.model = []

    # Using the train df to predict the target (instead of test df)
    def predict(self, train, test):
        start = timeit.default_timer()
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                if test[column].dtype == np.float64 or test[column].dtype == np.int64:
                    model = XGBRegressor(n_estimators=100, random_state=42)
                else:
                    model = XGBClassifier(n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)
                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)
        return test_copy, time

    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        # Compute how many rows are used to predict values (roughly)
        self.N = train_copy.shape[0]*(1-nan_frac)*frac

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                #print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                non_missing = non_missing.sample(frac=frac)
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                if test[column].dtype == np.float64 or test[column].dtype == np.int64:
                    model = XGBRegressor(n_estimators=100, random_state=42)
                else:
                    model = XGBClassifier(n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)
                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return test_copy

    def XGB_shap(self, train, test, frac, col, n):
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        column = train.columns[col]

        # Train on the non missing values
        non_missing = train_copy.loc[train[column].notna()]
        non_missing = non_missing.sample(frac=frac)
        missing = test_copy.loc[test[column].isna()]

        X_train = non_missing.drop(columns=[column])
        y_train = non_missing[column]
        X_missing = missing.drop(columns=[column])

        row_index_int = randrange(X_train.shape[0] - 1)
        row_index = X_train.index[row_index_int]
        x_explain = X_train.values[row_index_int]
        y_explain = y_train.values[row_index_int]

        X_train = X_train.drop(row_index, axis=0)
        y_train = y_train.drop(row_index, axis=0)
        feature_names = np.array(X_train.columns)

        # Convert to values
        X_train = X_train.values
        y_train = y_train.values

        # Initiate model
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.model = model
        # Calculate SHAP values, code from https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_missing)
        shap_values = explanation.values
        print(shap_values.shape)
        df_shap_values = pd.DataFrame(data=shap_values,columns=feature_names)
        df_feature_importance = pd.DataFrame(columns=['feature','importance'])
        for col in df_shap_values.columns:
            importance = df_shap_values[col].abs().mean()
            df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
        #df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)
        display(df_feature_importance)
        self.feature_array = df_feature_importance.iloc[:,1]




# TabPFN class (Regression case)
class tabPFN_Regressor():
    def __init__(self):
        self.name = 'TabPFN Regressor'
        self.N = None
        self.model = []

    def predict(self, train, test):
        start = timeit.default_timer()
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                non_missing = non_missing.sample(frac=1)
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Initialise and train regressor
                regressor = TabPFNRegressor(device='cuda')
                regressor.fit(X_train, y_train)

                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = regressor.predict(X_missing)
                
                # Save the model
                self.model.append(regressor)
                
                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)
        return test_copy, time

    def shapPFN(self, train, test, frac, col, n):
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        column = train.columns[col]

        # Train on the non missing values
        non_missing = train_copy.loc[train[column].notna()]
        non_missing = non_missing.sample(frac=frac)
        missing = test_copy.loc[test[column].isna()]

        X_train = non_missing.drop(columns=[column])
        y_train = non_missing[column]
        X_missing = missing.drop(columns=[column])

        row_index_int = randrange(X_train.shape[0] - 1)
        row_index = X_train.index[row_index_int]
        x_explain = X_train.values[row_index_int]
        y_explain = y_train.values[row_index_int]

        X_train = X_train.drop(row_index, axis=0)
        y_train = y_train.drop(row_index, axis=0)
        feature_names = np.array(X_train.columns)

        # Convert to values
        X_train = X_train.values
        y_train = y_train.values

        # Initiate model
        model = TabPFNRegressor(device='cuda')
        model.fit(X_train, y_train)
        self.model = model
        # Calculate SHAP values
        shap_values = interpretability.shap.get_shap_values(
            estimator=model,
            test_x=X_missing[:n],
            #attribute_names=feature_names,
            algorithm="permutation",
        )

        df_shap_values = pd.DataFrame(data=shap_values.values,columns=feature_names)
        df_feature_importance = pd.DataFrame(columns=['feature','importance'])
        for col in df_shap_values.columns:
            importance = df_shap_values[col].abs().mean()
            df_feature_importance.loc[len(df_feature_importance)] = [col,importance]
        #df_feature_importance = df_feature_importance.sort_values('importance',ascending=False)
        display(df_feature_importance)
        self.feature_array = df_feature_importance.iloc[:,1]
        
        '''with open('shap_values.txt', 'w') as output:
            for i in range(len(shap_values)):
                output.write(str(shap_values[i]))
                output.write('\n')
                output.write('\n')

        output.close()
        '''
        # Create visualization
        #fig = interpretability.shap.plot_shap(shap_values)
        #fig.savefig(format='png')
  

    def shapiq(self, train, test, frac, col):
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        column = train.columns[col]

        # Train on the non missing values
        non_missing = train_copy.loc[train[column].notna()]
        non_missing = non_missing.sample(frac=frac)
        missing = test_copy.loc[test[column].isna()]

        X_train = non_missing.drop(columns=[column])
        y_train = non_missing[column]

        row_index_int = randrange(X_train.shape[0] - 1)
        row_index = X_train.index[row_index_int]
        x_explain = X_train.values[row_index_int]
        y_explain = y_train.values[row_index_int]

        X_train = X_train.drop(row_index, axis=0)
        y_train = y_train.drop(row_index, axis=0)
        feature_names = np.array(X_train.columns)

        # Convert to values
        X_train = X_train.values
        y_train = y_train.values

        # Initialise and train regressor
        model = TabPFNRegressor(device='cuda')
        explainer = shapiq.Explainer(
            model=model,
            data=X_train,
            labels=y_train,
            index="SV",  # Shapley values
            max_order=1,  # first order Shapley values
            #empty_prediction=float(average_prediction),  # Optional, can also be inferred from the model
        )
        print(f"Explainer Class: {explainer.__class__.__name__} inferred from the model.")

        imputer = explainer._imputer
        if not os.path.exists("tabpfn_values_explainer.npz"):
            imputer.verbose = True  # see the pre-computation progress
            imputer.fit(x_explain)
            imputer.precompute()
            imputer.save_values("tabpfn_values_explainer.npz")
        imputer.load_values("tabpfn_values_explainer.npz")

        shapley_values = explainer.explain(x_explain)
        display(shapley_values.dict_values)
        shapley_values.plot_waterfall(feature_names=feature_names, abbreviate=False)




    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        # Compute how many rows are used to predict values (roughly)
        self.N = train_copy.shape[0]*(1-nan_frac)*frac

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                #print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                non_missing = non_missing.sample(frac=frac)
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Initialise and train regressor
                regressor = TabPFNRegressor()
                regressor.fit(X_train, y_train)

                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = regressor.predict(X_missing)
                # Save the model
                self.model.append(regressor)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return test_copy

    # Consider single column to parallelize the execution
    def process_column(self, column, train, test, train_copy, test_copy):
        if test_copy[column].isnull().sum() > 0:
            process_id = multiprocessing.current_process().pid  # Get process ID
            print(f"Process {process_id} is imputing column: {column}")

            # Train on the non missing values
            non_missing = train_copy.loc[train[column].notna()]
            non_missing = non_missing.sample(frac=0.5)
            missing = test_copy.loc[test[column].isna()]

            X_train = non_missing.drop(columns=[column])
            y_train = non_missing[column]
            X_missing = missing.drop(columns=[column])

            # Initialise and train regressor
            regressor = TabPFNRegressor()
            regressor.fit(X_train, y_train)
            # After the model has been trained on non missing values, we apply it to the missing ones in test set
            predictions = regressor.predict(X_missing)

            return column, predictions

    # Parallel execution from chatgpt
    def predict_parallel(self, train, test):
        start = timeit.default_timer()
        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()

        # Run in parallel
        results = Parallel(n_jobs=15)(
            delayed(self.process_column)(column, train, test, train_copy, test_copy)
            for column in test_copy.columns
        )

        for column, y_result in results:
            test_copy.loc[test[column].isna(), column] = y_result

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return test_copy

# Catboost
class catboost:
    def __init__(self):
        self.name = 'Catboost'
        self.N = None
        self.model = []

    def predict(self, train, test):
        start = timeit.default_timer()
        # Initialise the model with RMSE loss
        model = CatBoostRegressor(loss_function='RMSE',task_type="GPU")

        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        
        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                self.N = X_train.shape[0]
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Fit model on training dataset
                model.fit(X_train, y_train, silent=True)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)
        return test_copy, time

    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        # Initialise the model with RMSE loss
        model = CatBoostRegressor(loss_function='RMSE',task_type="GPU")

        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()
        
        # Compute how many rows are used to predict values (roughly)
        self.N = train_copy.shape[0]*(1-nan_frac)*frac

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                non_missing = non_missing.sample(frac=frac)
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Fit model on training dataset
                model.fit(X_train, y_train, silent=True)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return test_copy

# Random forest
class random_forest:
    def __init__(self):
        self.name = 'random forest'
        self.N = None
        self.model = []

    def predict(self, train, test):
        start = timeit.default_timer()
        # Initialise the model with RMSE loss
        model = RandomForestRegressor(max_depth=2, random_state=0)
        test_copy = test.copy()
        train_copy = train.copy()
        
        # Compute how many rows are used to predict values (roughly)
        #self.N = train_copy.shape[0]*(1-nan_frac)*frac
        
        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Fit model on training dataset
                model.fit(X_train, y_train)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)
        return test_copy, time

    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        # Initialise the model with RMSE loss
        model = RandomForestRegressor(max_depth=2, random_state=0)

        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                non_missing = non_missing.sample(frac=frac)
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Fit model on training dataset
                model.fit(X_train, y_train)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return test_copy


# LightGBM
class lightGBM:
    def __init__(self):
        self.name = 'lightGBM'
        self.N = None
        self.model = []

    def predict(self, train, test):
        start = timeit.default_timer()
        # Initialise the model with RMSE loss
        model = LGBMRegressor(metric='rmse')
        test_copy = test.copy()
        train_copy = train.copy()

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Fit model on training dataset
                model.fit(X_train, y_train)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        time = stop - start
        print('Time: ', time)
        return test_copy, time

    def predict_size(self, train, test, frac, nan_frac):
        start = timeit.default_timer()
        # Initialise the model with RMSE loss
        model = LGBMRegressor(metric='rmse')

        # Copy test, train and original dfs
        test_copy = test.copy()
        train_copy = train.copy()

        # Compute how many rows are used to predict values (roughly)
        self.N = train_copy.shape[0]*(1-nan_frac)*frac

        # Iterate over all columns
        for column in test_copy.columns:
            if test_copy[column].isnull().sum() > 0:
                print(f"Imputing column: {column}")

                # Train on the non missing values
                non_missing = train_copy.loc[train[column].notna()]
                non_missing = non_missing.sample(frac=frac)
                missing = test_copy.loc[test[column].isna()]

                X_train = non_missing.drop(columns=[column])
                y_train = non_missing[column]
                X_missing = missing.drop(columns=[column])

                # Fit model on training dataset
                model.fit(X_train, y_train)
                # After the model has been trained on non missing values, we apply it to the missing ones in test set
                predictions = model.predict(X_missing)
                # Save the model
                self.model.append(model)

                # Complete the whole dataset
                test_copy.loc[test[column].isna(), column] = predictions

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        return test_copy



# To manage all the needed operations, i.e. the dataframe analysis, the masking, the computation of errors
class manager:
    def __init__(self, df, nan_frac):
        # Drop the eids
        df = df.drop('eid', axis=1)
        # Consider just complete data
        self.df = df.dropna()


        # Parameters
        self.col_start = 0
        self.col_end = 58
        self.nan_frac = nan_frac
        self.train_size = 0.75

        # Create a train and a test dataframe
        self.train, self.test = train_test_split(self.df, train_size=self.train_size, random_state=19)

        # Update df depending on which columns we want to consider (i.e. col_start and col_end)
        self.test = self.test.iloc[:,self.col_start:self.col_end]
        self.train = self.train.iloc[:,self.col_start:self.col_end]

        # Mask in the train
        self.trainmask = self.mask(pd.DataFrame(self.train), self.nan_frac)

        # Mask in the data
        self.testmask = self.mask(pd.DataFrame(self.test), self.nan_frac)
        self.testmask = self.testmask.iloc[:, self.col_start:self.col_end]

        # Track where it has been masked
        self.testtrack = self.track(self.testmask)
        self.testtrack = self.testtrack.iloc[:, self.col_start:self.col_end]

        # Create store-class to store predicted results
        self.store = store()
        real = pd.DataFrame.to_numpy(self.test)
        self.store.real = real[self.testtrack]

        # Name of method we use - we impute it later
        self.name = None

    # Mask the data
    def mask(self, df, frac):
        dflocal = df.copy()
        ix = [(row, col) for row in range(dflocal.shape[0]) for col in range(dflocal.shape[1])]
        for row, col in random.sample(ix, int(round(frac * len(ix)))):
            dflocal.iat[row, col] = np.nan

        return dflocal

    # Track where it has been masked
    def track(self, df):
        return df.isna()

    # Compute the prediction and return error for models different than tabPFN
    def compute(self, model):
        # Instatiate method
        method = model()
        # Predict
        pred, time = method.predict(self.trainmask, self.testmask)
        # Assign predicted
        self.store.assign(method.name, pred, self.testtrack)
        # Assign time
        self.store.assign_time(method.name, time)
        # Compute error
        error = self.compute_error(pred)
        # Assign error
        self.store.assign_error(method.name, error)
        # Assign N (i.e. size of train df)
        self.store.assign_N(method.name, method.N)
        # Store models
        self.store.assign_model(method.name, method.model)
        return error

    # Parallel implementation for tabPFN
    def compute_parallel(self, model):
        method = model()
        
        # Predict the nan values
        pred = method.predict_parallel(self.trainmask, self.testmask)
        self.store.assign(method.name, pred, self.testtrack)
        error = self.compute_error(pred)
        self.store.assign_error(method.name, error)
        return error


    def compute_size(self, model, train_frac):
        # Instatiate method
        method = model()
        # Predict
        pred = method.predict_size(self.trainmask, self.testmask, train_frac, self.nan_frac)
        # Assign predicted
        self.store.assign(method.name, pred, self.testtrack)
        # Compute error
        error = self.compute_error(pred)
        # Assign error
        self.store.assign_error(method.name, error)
        # Assign N (i.e. size of train df)
        self.store.assign_N(method.name, method.N)
        # Store models
        self.store.assign_model(method.name, method.model)
        return error


    # Compute the error with the 2-Norm
    def compute_error(self, df):
        if not isinstance(df, np.ndarray):
            df = pd.DataFrame.to_numpy(df)
        pred = df[self.testtrack]
        real = pd.DataFrame.to_numpy(self.test)
        real = real[self.testtrack]
        diff = pred - real
        # Different expression for different version of scikit
        rms = sqrt(mean_squared_error(real, pred))
        return rms

    # Compare the performance
    def compare(self):
        # Compute standard mice
        print('Error of mean: ', self.compute(mean))

        # Compute Multi imputation chain equations
        print('Error of multi mice: ', self.compute(multi_mice))

        # Compute KNNimputer
        print('Error of KNNimputer: ', self.compute(KNNimputer))

        # Compute tabPFN
        print('Error of tabPFN: ', self.compute(tabPFN_Regressor))

        # Compute XGBoost
        print('Error of XGBoost: ', self.compute(XGBoost))

        # Compute CatBoost
        print('Error of Catboost: ', self.compute(catboost))

        # Compute Random forest
        print('Error of random forest: ', self.compute(random_forest))

        # Compute LightGBM
        #print('Error of lightGBM: ', self.compute(lightGBM))


    def compare_row_analysis(self, train_size):
        print('train fraction: ', str(train_size), '\n')

        # Compute standard mice
        print('Error of mean: ', self.compute_size(mean, train_size))

        # Compute Multi imputation chain equations
        print('Error of multi mice: ', self.compute_size(multi_mice, train_size))
        
        # Compute KNNimputer
        print('Error of KNNimputer: ', self.compute_size(KNNimputer, train_size))

        # Compute tabPFN
        print('Error of tabPFN: ', self.compute_size(tabPFN_Regressor, train_size))

        # Compute XGBoost
        print('Error of XGBoost: ', self.compute_size(XGBoost, train_size))
        
        # Compute CatBoost
        print('Error of Catboost: ', self.compute_size(catboost, train_size))
        
        # Compute Random forest
        print('Error of random forest: ', self.compute_size(random_forest, train_size))

        # Compute LightGBM
       # print('Error of lightGBM: ', self.compute_size(lightGBM, train_size))



    # Split df into df_blood_count and df_biochemistry
    def split_df(self):
        meta_df = pd.read_excel('~/Documents/TabPFN evaluation/src/file_source/Metadata_UKB.xlsx')

        a, labels_descr = self.labels_meta(self.df.columns, meta_df)
        b = ~a
        self.df.columns = labels_descr
        df1 = self.df.iloc[:, a]
        df2 = self.df.iloc[:, b]
        return df1, df2

    # Find the columns that corresponds to either blood count or biochemistry
    def labels_meta(self, col, meta_df):
        labels_bool = []
        labels_descr = []
        for column in col:
            for j in range(meta_df.shape[0]):
                tmp = meta_df.iloc[j, 0]
                tmp = str(tmp) + '-0.0'
                if tmp == column:
                    labels_descr.append(meta_df.iloc[j, 5])
                    if meta_df.iloc[j, 4] == 'Blood count':
                        labels_bool.append(True)
                    else:
                        labels_bool.append(False)
        return np.array(labels_bool), np.array(labels_descr)


    # Plot error
    def plot_error(self, error):
        x = np.arange(len(error))
        plt.plot(x, error)
        plt.show()



    # Print parameters
    def print_parameters(self,path):
        with open(path + 'parameters.txt', 'w') as f:
            if self.blood_count:
                f.write('Imputed dataframe: blood count' + '\n')
            else: 
                f.write('Imputed dataframe: biochemistry')
            f.write('Nan frac: ' + str(self.nan_frac) + '\n')
            f.write('Train %: ' + str(self.train_size) + '\n')
            if self.blood_count:
                f.write('Shape of df blood_count: ' + str(self.df_blood_count.shape) + '\n')
            else: 
                f.write('Shape of df biochemistry: ' + str(self.df_biochemistry.shape) + '\n')
            f.write('Imputed values: ' + str(len(self.store.real)) + '\n')
            

# Class to store the results computed for the different imputation techniques
class store:
    def __init__(self):
        # Real values
        self.real = None
        # Predicted by the models
        self.mean_pred = None
        self.mice_pred = None
        self.KNN_pred = None
        self.XGB_pred = None
        self.Cat_pred = None
        self.tabPFN_pred = None
        self.lightGBM_pred = None
        self.random_forest_pred = None
        # RMSE 
        self.mean_error = []
        self.mice_error = []
        self.KNN_error = []
        self.XGB_error = []
        self.Cat_error = []
        self.tabPFN_error = []
        self.lightGBM_error = []
        self.random_forest_error = []
        # Runtime
        self.mean_time = None
        self.mice_time = None
        self.KNN_time = None
        self.XGB_time = None
        self.Cat_time = None
        self.tabPFN_time = None
        self.lightGBM_time = None
        self.random_forest_time = None
        # Rows used to train the models
        self.mean_N = None
        self.mice_N = None
        self.KNN_N = None
        self.XGB_N = None
        self.Cat_N = None
        self.tabPFN_N = None
        self.lightGBM_N = None
        self.random_forest_N = None
        # Store models
        self.mean_model = None
        self.mice_model = None
        self.KNN_model = None
        self.XGB_model = []
        self.Cat_model = []
        self.tabPFN_model = []
        self.lightGBM_model = []
        self.random_forest_model = []

    def print_predictions_csv(self, path):
        dataset = pd.DataFrame({'real': self.real, 'mice': self.mice_pred, 'mean': self.mean_pred, 'KNN_pred': self.KNN_pred, 'XGB': self.XGB_pred, 'Catboost': self.Cat_pred, 'tabPFN': self.tabPFN_pred, 'random forest': self.random_forest_pred})
        dataset.to_csv(path+'predictions.csv')


    def assign(self, name, pred, testtrack):
        pred_copy = pred.copy()
        # Check that it is an np.array
        if not isinstance(pred_copy, np.ndarray):
            pred_copy = pd.DataFrame.to_numpy(pred_copy)

        if name == 'Mean':
            self.mean_pred = pred_copy[testtrack]

        if name == 'Multi MICE':
            self.mice_pred = pred_copy[testtrack]

        if name == 'KNN imputer':
            self.KNN_pred = pred_copy[testtrack]

        if name == 'XGBoost':
            self.XGB_pred = pred_copy[testtrack]

        if name == 'Catboost':
            self.Cat_pred = pred_copy[testtrack]

        if name == 'TabPFN Regressor':
            self.tabPFN_pred = pred_copy[testtrack]

        if name == 'lightGBM':
            self.ligthGBM_pred = pred_copy[testtrack]

        if name == 'random forest':
            self.random_forest_pred = pred_copy[testtrack]

    def assign_error(self, name, error):
        # assign errors
        if name == 'Mean':
            self.mean_error.append(error)

        if name == 'Multi MICE':
            self.mice_error.append(error)

        if name == 'KNN imputer':
            self.KNN_error.append(error)

        if name == 'XGBoost':
            self.XGB_error.append(error)

        if name == 'Catboost':
            self.Cat_error.append(error)

        if name == 'TabPFN Regressor':
            self.tabPFN_error.append(error)

        if name == 'lightGBM':
            self.ligthGBM_error.append(error)

        if name == 'random forest':
            self.random_forest_error.append(error)

    
    def assign_time(self, name, time):
        # assign time
        if name == 'Mean':
            self.mean_time= time

        if name == 'Multi MICE':
            self.mice_time = time

        if name == 'KNN imputer':
            self.KNN_time = time

        if name == 'XGBoost':
            self.XGB_time = time

        if name == 'Catboost':
            self.Cat_time = time

        if name == 'TabPFN Regressor':
            self.tabPFN_time = time

        if name == 'lightGBM':
            self.ligthGBM_time = time

        if name == 'random forest':
            self.random_forest_time = time


    def assign_N(self, name,  N):
        # assign time
        if name == 'Mean':
            self.mean_N= N

        if name == 'Multi MICE':
            self.mice_N = N

        if name == 'KNN imputer':
            self.KNN_N = N

        if name == 'XGBoost':
            self.XGB_N = N

        if name == 'Catboost':
            self.Cat_N = N

        if name == 'TabPFN Regressor':
            self.tabPFN_N = N

        if name == 'lightGBM':
            self.ligthGBM_N = N

        if name == 'random forest':
            self.random_forest_N = N

    def assign_model(self, name, model):
        # assign models
        if name == 'Mean':
            self.mean_model= model

        if name == 'Multi MICE':
            self.mice_model = model

        if name == 'KNN imputer':
            self.KNN_model = model

        if name == 'XGBoost':
            self.XGB_model = model

        if name == 'Catboost':
            self.Cat_model = model

        if name == 'TabPFN Regressor':
            self.tabPFN_model = model

        if name == 'lightGBM':
            self.ligthGBM_model = model

        if name == 'random forest':
            self.random_forest_model = model

    def save_models(self, path):
        mean_filename = path + 'mean_model.sav'
        joblib.dump(self.mean_model, mean_filename)
        mice_filename = path+'mice_model.sav'
        joblib.dump(self.mice_model, mice_filename)
        KNN_filename = path+'KNN_model.sav'
        joblib.dump(self.KNN_model, KNN_filename)
        Path(path + 'catboost_joblib').mkdir(exist_ok=True)
        Path(path + 'xgboost_joblib').mkdir(exist_ok=True)
        Path(path + 'tabpfn_joblib').mkdir(exist_ok=True)
        Path(path + 'random_forest_joblib').mkdir(exist_ok=True)
        for i in range(len(self.XGB_model)):
            XGB_filename = path + 'xgboost_joblib/XGB_model_col_' + str(i) + '.sav'
            joblib.dump(self.XGB_model[i], XGB_filename)
            catboost_filename = path + 'catboost_joblib/catboost_model_col_' + str(i) + '.sav'
            joblib.dump(self.Cat_model[i], catboost_filename)
            tabPFN_filename = path + 'tabpfn_joblib/tabpfn_model_col_' + str(i) + '.sav'
            joblib.dump(self.tabPFN_model[i], tabPFN_filename)
            random_forest_filename = path + 'random_forest_joblib/random_forest_model_col_' + str(i) + '.sav'
            joblib.dump(self.random_forest_model[i], random_forest_filename)

    def save_train_rows(self, path):
        dataset = pd.DataFrame({'N mean':[self.mean_N], 'N_mice': [self.mice_N], 'N_KNN': [self.KNN_N], 'N_catboost': [self.Cat_N], 'N_xgboost': [self.XGB_N], 'N_tabPFN': [self.tabPFN_N], 'N_random_forest':[self.random_forest_N]})
        dataset.to_csv(path+'train_rows.csv')
    
    def save_runtimes(self, path):
        dataset = pd.DataFrame({'time_mean':[self.mean_time], 'time_mice':[self.mice_time], 'time_KNN':[self.KNN_time], 'time_catboost':[self.Cat_time],'time_xboost': [self.XGB_time], 'time_tabPFN':[self.tabPFN_time], 'time_random_forest':[self.random_forest_time]})
        dataset.to_csv(path + 'runtime.csv')
       
    def save_rmse(self, path):
        dataset = pd.DataFrame({'error_mean':[self.mean_error], 'error_mice':[self.mice_error], 'error_KNN':[self.KNN_error], 'error_catboost':[self.Cat_error],'error_xboost': [self.XGB_error], 'error_tabPFN':[self.tabPFN_error], 'error_random_forest':[self.random_forest_error]})
        dataset.to_csv(path + 'error.csv')

    def print_out(self):
        with open('results/Fifth test/Predictions/real.txt', "w") as f:
            f.write('Real values: ')
            for i in range(len(self.real)):
                f.write('\n')
                f.write(str(self.real[i]))


        with open('results/Third test/Predictions/mean.txt', "w") as f:
            f.write('Mean predictions: ')
            for i in range(len(self.mean_pred)):
                f.write('\n')
                f.write(str(self.mean_pred[i]))


        with open('results/Third test/Predictions/mice.txt', "w") as f:
            f.write('Mice predictions: ')
            for i in range(len(self.mice_pred)):
                f.write('\n')
                f.write(str(self.mice_pred[i]))


        with open('results/Third test/Predictions/KNN.txt', "w") as f:
            f.write('KNN predictions: ')
            for i in range(len(self.KNN_pred)):
                f.write('\n')
                f.write(str(self.KNN_pred[i]))


        with open('results/Third test/Prediction/XGB.txt', "w") as f:
            f.write('XGB predictions: ')
            for i in range(len(self.XGB_pred)):
                f.write('\n')
                f.write(str(self.XGB_pred[i]))

        with open('results/Third test/Prediction/Cat.txt', "w") as f:
            f.write('Cat predictions: ')
            for i in range(len(self.Cat_pred)):
                f.write('\n')
                f.write(str(self.Cat_pred[i]))

        with open('results/Third test/Prediction/tabPFN.txt', "w") as f:
            f.write('TabPFN predictions: ')
            for i in range(len(self.tabPFN_pred)):
                f.write('\n')
                f.write(str(self.tabPFN_pred[i]))


class store_feature_matrix():
    def __init__(self):
        self.joblib_models = []
        self.feature_matrix = []
    
    def save_models(self, path):
        Path(path + 'tabpfn_joblib').mkdir(exist_ok=True)
        for i in range(len(self.joblib_models)):
            tabPFN_filename = path + 'tabpfn_joblib/tabpfn_model_col_' + str(i) + '.sav'
            joblib.dump(self.joblib_models[i], tabPFN_filename)
    
    def save_matrix(self, path):
        np.savetxt(path + 'feature_matrix.csv', np.matrix(self.feature_matrix), delimiter=",")


# Consider just a part of df
class manager_blood(manager):
    def __init__(self, df, nan_frac, blood_count):
        # Drop the eids
        df = df.drop('eid', axis=1)
        self.age = df[['Age']]
        self.sex = df[['Sex']]
        self.df = df.drop(['Age','Sex'], axis=1)

        # Split into blood count and biochemistry
        self.df_blood_count, self.df_biochemistry = self.split_df()
        self.df_blood_count['Age'] = self.age
        self.df_blood_count['Sex'] = self.sex
        self.df_biochemistry['Age'] = self.age
        self.df_biochemistry['Sex'] = self.sex
        # Drop two columns that have most zeros (catboost has got problems there...)
        self.df_blood_count = self.df_blood_count.drop(['Nucleated red blood cells', 'Nucleated red blood cell (%)'], axis=1)
        self.df_biochemistry = self.df_biochemistry.drop(['Oestradiol','Rheumatoid factor'], axis=1)
        # Consider just complete data        
        self.df_blood_count  = self.df_blood_count.dropna()
        self.df_biochemistry = self.df_biochemistry.dropna()
        
        #self.df = self.df.drop(['age','sex'], axis=1)
        #df_age_sex = pd.read_csv('C:/Users/tmp/projects/work/projects/TabPFN/covariates.csv', sep=';')
        #df_age_sex = df_age_sex.replace({pd.NA: np.nan})
        #col_age, col_row = import_age_sex.merge_df(self.df, df_age_sex)

        # Split into blood count and biochemistry
        #self.df_blood_count, self.df_biochemistry = self.split_df()

        # Parameters
        self.col_start = 0
        self.col_end = -1
        self.nan_frac = nan_frac
        self.train_size = 0.01

        # Store which df we are imputing
        self.blood_count = blood_count

        # Create a train and a test dataframe
        if blood_count:
            self.train, self.test = train_test_split(self.df_blood_count, train_size=self.train_size, random_state=19)
        else:
            self.train, self.test = train_test_split(self.df_biochemistry, train_size=self.train_size, random_state=19)

        self.N = self.train.shape[0]
        # Update df depending on which columns we want to consider (i.e. col_start and col_end)
        if self.col_end != -1:
            self.test = self.test.iloc[:, self.col_start:self.col_end]
            self.train = self.train.iloc[:, self.col_start:self.col_end]

        # Mask in the train
        self.trainmask = self.mask(pd.DataFrame(self.train),self.nan_frac)
 
        #Mask in the data
        self.testmask = self.mask(pd.DataFrame(self.test), self.nan_frac)
        if self.col_end != -1:
            self.testmask = self.testmask.iloc[:, self.col_start:self.col_end]

        # Track where it has been masked
        self.testtrack = self.track(self.testmask)
        if self.col_end != -1:
            self.testtrack = self.testtrack.iloc[:, self.col_start:self.col_end + 1]

        # Name of method we use - we impute it later
        self.name = None

        # Create store-class to store predicted results
        self.store = store()
        
        real = pd.DataFrame.to_numpy(self.test)
        self.store.real = real[self.testtrack]




if __name__ == "__main__":
    df = pd.read_csv("file_source/UKB_Patients_blood_copy.csv", sep=';')
    #df_age_sex = pd.read_csv('C:/Users/tmp/projects/work/projects/TabPFN/covariates.csv', sep=';')
    df = df.replace({pd.NA: np.nan})
    df = df.rename(columns={'sex': 'Sex', 'age':'Age'}

    '''
    dat = import_age_sex.map_blood2covariates(df, df_age_sex)
    df = pd.concat([df, dat], axis=1)
    dat.to_csv('C:/Users/tmp/projects/work/projects/TabPFN/metabolomics_age_sex.csv', index=False, sep=';')
    '''

    manager = manager_blood(df,0.05, 1)
    '''manager.compare()
    
    plt.scatter(manager.store.mean_error, manager.store.mean_time, label = 'mean')
    plt.scatter(manager.store.mice_error, manager.store.mice_time, label = 'mice')
    plt.scatter(manager.store.KNN_error, manager.store.KNN_time, label = 'KNN')
    plt.scatter(manager.store.tabPFN_error, manager.store.tabPFN_time, label = 'tabPFN')
    plt.scatter(manager.store.Cat_error, manager.store.Cat_time, label='catboost')
    plt.scatter(manager.store.XGB_error, manager.store.XGB_time, label='XGBoost')
    plt.legend()
    plt.xlabel('RMSE')
    plt.ylabel('Time in s')
    plt.savefig('Blood count 0.05.png')
    '''




