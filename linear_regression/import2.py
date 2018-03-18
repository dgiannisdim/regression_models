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
def analyze_data(data):
    data = data.loc[:, 'property_type':]
    features = data.columns

    for f in features:
        print(data[f].value_counts())
        #print(f, ': ', data[f].unique())




#which categorical variables to use in model
def analyze_data_2(data):
    data = data.loc[:, 'property_type':]

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
    data_temp = data.loc[:, 'electricity_always_on':]
    data_temp['month'] = data['month']
    return data_temp

data = reorder_features(data)


#drop outliers and features with 0 values
def drop_outliers(data):
    data = data[(data.electricity_total < 2000) & (data.electricity_total != 0) & (data.electricity_other < 1000)]
    data = data.drop(['swimming_pool', 'sauna', 'electricity_water_heating', 'water_heating', 'grill',
                      'electricity_pool_or_sauna', 'space_heating', 'photo_voltaic', 'oven', 'hob'], axis=1)
    data = data.reset_index(drop=True)

    return data



#merge installation charasteristics to 'electric' and 'non_electric'
def merge_characteristics(data):
    features = ['water_heating', 'grill_heating', 'oven_heating']

    for i in data.index:
        for f in features:
            a, b = f.split('_')
            a = a[0].upper()
            b = b[0].upper()
            c = a + b + '.02'
            if data[f][i] == c:
                data[f][i] = 'electric'
            else:
                data[f][i] = 'other'

        # merge space_heating
        if data['space_heating'][i] == 'SH.02' or data['space_heating'][i] == 'SH.03':
            data['space_heating'][i] = 'electric'
        else:
            data['space_heating'][i] = 'other'

        # merge stove_heating
        if data['stove_heating'][i] == 'STH.02':
            data['stove_heating'][i] = 'electric'
        else:
            data['stove_heating'][i] = 'other'


    return data




#change categorical variables to ordinal
def change_to_ordinal(data):
    #change property_size
    for i in data.index:
        for j in range(1, 6):
            if data['property_size'][i] == 'PS.0' + str(j):
                data['property_size'][i] = j


            # change occupants
            if data['occupants'][i] == 'OCC.0' + str(j):
                data['occupants'][i] = j


    return data





#change ordinal variables to categorical
def change_to_categorical(data):
    # change month
    for i in data.index:
        for j in range(0, 13):
            if data['month'][i] == j:
                data['month'][i] = str(j) + 'th'
                break

    return data




#analyze_data(data)

merge_characteristics(data)
data = drop_outliers(data)
change_to_ordinal(data)
change_to_categorical(data)



#export final dataset to csv file
#data.to_csv('final_dataset_merged.csv', index=False)