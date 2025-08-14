import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'~/Documents/TabPFN evaluation/src/')
from visualization.create_feature_matrix import manager_feature_matrix
from pathlib import Path
import datetime
from MICE import store_feature_matrix
from IPython.display import display
import seaborn as sns


def run_feature_matrix_blood_count(df, nan_frac, path,n):
    now = datetime.datetime.now()
    clock_time = str(now.time())
    Path(path +'/' + clock_time).mkdir()
    path_local = path + '/' + clock_time +'/'
    col2drop_blood = ['Nucleated red blood cells', 'Nucleated red blood cell (%)', 'Basophill (%)','Eosinophill (%)','High light scatter retic (%)', 'Lymphocyte (%)', 'Monocytes (%)','Neutrophills (%)','Reticulocyte (%)']
    col2drop_biochemistry = ['Rheumatoid factor', 'Oestradiol']
    manager4column = manager_feature_matrix(df,'sex',nan_frac, True, col2drop_blood, col2drop_biochemistry)
    store_data = store_feature_matrix()
    for column in manager4column.df_blood_count:
        print('imputing column' + str(column))
        manager = manager_feature_matrix(df,column,nan_frac,True, col2drop_blood, col2drop_biochemistry)
        feature_array, model = manager.XGBoost_shap(n)
        store_data.feature_matrix.append(np.array(feature_array))
        store_data.joblib_models.append(model)
        #print(feature_array)
        #print(store_data.feature_matrix)
        #if i == 10:
            #break
    
    #store_data.feature_matrix = np.matrix(store_data.feature_matrix)
    print(store_data.feature_matrix)
    store_data.feature_matrix = np.transpose(store_data.feature_matrix)
    #store_data.feature_matrix = create_df(store_data.feature_matrix)
    #df_feature = pd.DataFrame(store_data.feature_matrix, index=manager4column.df_blood_count.columns, columns=manager4column.df_blood_count.columns)
    df_feature = pd.DataFrame(store_data.feature_matrix)
    df_feature = df_feature.div(df_feature.sum(axis=0), axis=1)
    df_feature = create_df_pd(df_feature,  manager4column.df_blood_count.columns, manager4column.df_blood_count.columns)
    df_feature.columns = manager4column.df_blood_count.columns
    df_feature.index = manager4column.df_blood_count.columns
    store_data.save_models(path_local)
    store_data.save_matrix(path_local)
    display(df_feature)
    sns.set_theme(rc={'figure.figsize':(15.7,11.7)})
    svm = sns.heatmap(df_feature, annot=False)    
    figure = svm.get_figure()    
    figure.savefig(path_local + 'heatmap_blood_count_XGB.png', dpi=800)


def run_feature_matrix_biochemistry(df, nan_frac, path, n):
    now = datetime.datetime.now()
    clock_time = str(now.time())
    Path(path +'/' + clock_time).mkdir()
    path_local = path + '/' + clock_time +'/'
    col2drop_blood = ['Nucleated red blood cells', 'Nucleated red blood cell (%)']
    col2drop_biochemistry = ['Rheumatoid factor', 'Oestradiol']
    manager4column = manager_feature_matrix(df,'sex',nan_frac,False, col2drop_blood, col2drop_biochemistry)
    manager4column.print_parameters(path_local)
    store_data = store_feature_matrix()
    for column in manager4column.df_biochemistry:
        print('imputing column' + str(column))
        manager = manager_feature_matrix(df,column,nan_frac,False)
        feature_array, model = manager.XGBoost_shap(n)
        store_data.feature_matrix.append(np.array(feature_array))
        store_data.joblib_models.append(model)

    
    store_data.feature_matrix = np.transpose(store_data.feature_matrix)
    df_feature = pd.DataFrame(store_data.feature_matrix)
    df_feature = df_feature.div(df_feature.sum(axis=0), axis=1)
    df_feature = create_df_pd(df_feature,  manager4column.df_biochemistry.columns, manager4column.df_biochemistry.columns)
    df_feature.columns = manager4column.df_biochemistry.columns
    df_feature.index = manager4column.df_biochemistry.columns
    store_data.save_models(path_local)
    store_data.save_matrix(path_local)
    display(df_feature)
    sns.set_theme(rc={'figure.figsize':(15.7,11.7)})
    svm = sns.heatmap(df_feature, annot=False, cmap="crest")
    figure = svm.get_figure()
    figure.savefig(path_local + 'heatmap_biochemistry_XGB.png', dpi=800)




def create_df(matrix):
    new_df = []
    for i in range(matrix.shape[1]):
        array = matrix[:,i]
        array = np.insert(array,i,1)
        print(array)
        new_df.append(array)
    new_df = np.transpose(new_df)
    return new_df

def create_df_pd(df, rows, cols):
    new_df = pd.DataFrame(index=rows, columns=cols, dtype=float)
    for i in range(df.shape[1]):
        array = df.iloc[:,i]
        array = np.insert(array,i, 1)
        new_df.iloc[:,i] = array
    
    return new_df

        









