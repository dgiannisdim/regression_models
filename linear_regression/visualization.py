import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing


#remove useless data
def del_useless_features(data, columns):
    return data.drop(columns, axis=1)


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data.sort_values(by=['household_id', 'month'])
data = data.reset_index(drop=True)


print(data)
