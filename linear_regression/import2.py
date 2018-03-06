import pandas as pd
import numpy as np




pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data.sort_values(by=['household_id', 'month'])
data = data.reset_index(drop=True)



#data = pd.get_dummies(data)


#how many categories from each feature
def analyze_values(data):
    data = data.iloc[:, 14:]
    features = data.columns

    for f in features:
        print(data[f].value_counts())

analyze_values(data)


#which categorical variables to use in model
def analyze_values_2(data):
    data = data.iloc[:, 14:]

    for col_name in data.columns:
        if data[col_name].dtypes == 'object':
            unique_cat = len(data[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))




#outlier detection
def find_outliers(x):
    q1 = np.percentile(x, 5)
    q3 = np.percentile(x, 95)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indicies = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indicies])

    return outlier_indicies, outlier_values


consumptions = data.loc[:, 'electricity_always_on' : 'electricity_total']
consumptions = list(consumptions.columns)

for c in consumptions:
    lighting_indices, lighting_values = find_outliers(data[c])
    #print ('Outliers in ', c, ' : ' , np.sort(lighting_indices))



#feature 'month' in the end of the dataset
def reorder_features(data):
    cols = data.columns.tolist()
    cols = cols[2:]
    data_temp = data[cols]
    data_temp['month'] = data['month']
    return data_temp

data = reorder_features(data)


#drop outliers and features with 0 values
def drop_outliers(data):
    data = data[(data.electricity_total < 2000) & (data.electricity_total != 0) & (data.electricity_other < 1000)]
    data = data.drop(['swimming_pool', 'sauna', 'electricity_pool_or_sauna'], axis=1)
    data = data.reset_index(drop=True)

    return data

data = drop_outliers(data)

print(data.describe())
#export final dataset to csv file
data.to_csv('final_dataset.csv', index=False)