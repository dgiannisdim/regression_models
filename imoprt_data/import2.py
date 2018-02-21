import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing




data = pd.read_csv(r'C:\regression models\imoprt_data/final_dataset.csv')

pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

del data['Unnamed: 0']
data = data.dropna()

data = data.sort_values(by=['household_id', 'month'])
data = data.reset_index(drop=True)




print(data)



#data = pd.get_dummies(data)


#data.to_csv('final_dataset.csv')


#scatter_matrix(data)
#plt.show()