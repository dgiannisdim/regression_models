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
    data = data.iloc[:, 11:]
    features = data.columns

    for f in features:
        print(data[f].value_counts())




#which categorical variables to use in model
def analyze_values_2(data):
    data = data.iloc[:, 11:]

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
    data = data.drop(['swimming_pool', 'sauna', 'electricity_pool_or_sauna'], axis=1)
    data = data.reset_index(drop=True)

    return data

data = drop_outliers(data)


#merge installation charasteristics to 'electric' and 'non_electric'
def merge_characteristics(data):
    for i in data.index:
        #merge space_heating
        if data['space_heating'][i] == 'SH.02' or data['space_heating'][i] == 'SH.03':
            data['space_heating'][i] = 'electric'
        else:
            data['space_heating'][i] = 'non_electric'

        # merge water_heating
        if data['water_heating'][i] == 'WH.02':
            data['water_heating'][i] = 'electric'
        else:
            data['water_heating'][i] = 'non_electric'

        # merge stove_heating
        if data['stove_heating'][i] == 'STH.02':
            data['stove_heating'][i] = 'electric'
        else:
            data['stove_heating'][i] = 'non_electric'

        # merge grill_heating
        if data['grill_heating'][i] == 'GH.02':
            data['grill_heating'][i] = 'electric'
        else:
            data['grill_heating'][i] = 'non_electric'

        # merge oven_heating
        if data['oven_heating'][i] == 'OH.02':
            data['oven_heating'][i] = 'electric'
        else:
            data['oven_heating'][i] = 'non_electric'

    return data



#change categorical variables to ordinal
def change_to_ordinal(data):
    #change property_size
    for i in data.index:
        if data['property_size'][i] == 'PS.01':
            data['property_size'][i] = 1
        elif data['property_size'][i] == 'PS.02':
            data['property_size'][i] = 2
        elif data['property_size'][i] == 'PS.03':
            data['property_size'][i] = 3
        else:
            data['property_size'][i] = 4

        # change occupants
        if data['occupants'][i] == 'OCC.01':
            data['occupants'][i] = 1
        elif data['occupants'][i] == 'OCC.02':
            data['occupants'][i] = 2
        elif data['occupants'][i] == 'OCC.03':
            data['occupants'][i] = 3
        elif data['occupants'][i] == 'OCC.04':
            data['occupants'][i] = 4
        else:
            data['occupants'][i] = 5

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



merge_characteristics(data)
change_to_ordinal(data)
change_to_categorical(data)
analyze_values(data)


#export final dataset to csv file
#data.to_csv('final_dataset_merged.csv', index=False)