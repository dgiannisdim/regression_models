import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing



pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data[data.electricity_total != 0]
data = data.drop(data.index[293])
data = data.drop(data.index[433])
data = data.sort_values(by=['household_id', 'month'])
data = data.reset_index(drop=True)



def plot_histogram(x):
    plt.hist(x)
    plt.show()




def scatter_plots(data):
    features = data.columns
    x = features[14:]
    y = features[2:14]
    print(x, y)
    print(len(y))


    plt.figure()
    for i in range(1,10):
        a = 331
        plt.subplot(a)
        for k in range(0, 2):
            plt.scatter(data[x[0]], data[y[0]])

    plt.show()



data = data.iloc[:, [13, [25:35]]]
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)

#data.hist()
plt.show()

