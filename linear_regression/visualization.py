import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from string import ascii_letters
import seaborn as sns



pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

# Import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')


# Import new datasets
data_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage_new.csv')


#sort drop NaN values
def sort_and_drop(data):
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data


data = sort_and_drop(data)
data_new = sort_and_drop(data_new)
data_percentage = sort_and_drop(data_percentage)
data_percentage_new = sort_and_drop(data_percentage_new)




#create correlation matrix
def correlation_matrix(data):
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlation matrix')
    plt.savefig('Correlation matrix data percentage new')



#correlation_matrix(data_percentage_new)



def plot_histogram(x):
    plt.hist(x)
    plt.show()




def scatter_plots(data):
    features = data.columns
    features = features[: 11]
    for f in features:
        x = data[f]
        y = data.loc[:, 'fridge' : 'grill']
        frames = [x, y]
        c = pd.concat(frames, axis=1)
        scatter_matrix(c)

        plt.show()

scatter_plots(data_new)


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


