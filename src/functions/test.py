import numpy as np
import pandas as pd



def create_df(matrix):
    new_df = []
    for i in range(matrix.shape[1]):
        array = np.array(matrix[:,i])
        print(array)
        array = np.insert(array,i,1)
        new_df.append(array)
    return new_df




matrix = np.matrix(np.ones(5))
print(matrix.shape)
print(create_df(matrix))
