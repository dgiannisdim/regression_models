import pandas as pd
import numpy as np




pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data[data.electricity_total != 0]
data = data.sort_values(by=['household_id', 'month'])
data = data.reset_index(drop=True)



#data = pd.get_dummies(data)


#how many categories from each feature
def analyze_values(data):
    data = data.iloc[:, 14:]
    features = data.columns

    for f in features:
        print(data[f].value_counts())
        if len(data[f].value_counts()) == 1:
            data = data.drop(f, axis=1)    #drop columns with only 1 category

    return data


#which categorical variables to use in model
def analyze_values_2(data):
    data = data.iloc[:, 14:]

    for col_name in data.columns:
        if data[col_name].dtypes == 'object':
            unique_cat = len(data[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))



#outlier detection
def find_outliers(x):
    q1 = np.percentile(x, 10)
    q3 = np.percentile(x, 90)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indicies = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indicies])

    return outlier_indicies, outlier_values

lighting_indices, lighting_values = find_outliers(data['electricity_total'])
print (np.sort(lighting_indices))
