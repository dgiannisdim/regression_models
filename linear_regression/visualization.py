import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing



pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset.csv')




def plot_histogram(x):
    plt.hist(x)
    plt.show()




def scatter_plots(data):
    features = data.columns
    for f in features:
        x = data[f]
        y = data.loc[:, 'dishwasher' : 'digital_tv_box']
        frames = [x, y]
        c = pd.concat(frames, axis=1)
        scatter_matrix(c)

        plt.show()

scatter_plots(data)


def correlation_matrix(data):
    data = data.iloc[:, 35:45]
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

    plt.show()


def box_plots(data):
    x = data.loc[:, 'electricity_always_on' : 'electricity_total']
    x.plot(kind='box', subplots=True, layout=(4, 3), sharex=False, sharey=False)
    plt.show()


