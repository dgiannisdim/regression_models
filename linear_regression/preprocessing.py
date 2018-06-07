import pandas as pd
import numpy as np



pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset.csv')
data_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_new.csv')



# Drop NaN values and sort by id and month
def sort_and_drop(data, id):
    data = data.dropna()
    data = data.sort_values(by=[id, 'month'])
    data = data.reset_index(drop=True)
    del data[id]

    return data

data = sort_and_drop(data, 'household_id')
data_new = sort_and_drop(data_new, 'eui64')



# Number of non zero values from each category
def check_for_non_zeros(data):
    print('number of rows: ' + str(len(data)))
    print('number of non zero per category: ')
    print((data != 0).sum(axis=0))
    print('\n')

#check_for_non_zeros(data)
#check_for_non_zeros(data_new)


# How many categories from each feature
def analyze_data(data):
    data = data.loc[:, 'property_type':]
    features = data.columns

    print('check values of each feature:')
    for f in features:
        print(data[f].value_counts())
        #print(f, ': ', data[f].unique())

    print('\n')

#analyze_data(data)
#analyze_data(data_new)



# Which categorical variables to use in model
def analyze_data_2(data):
    data = data.loc[:, 'property_type':]

    for col_name in data.columns:
        if data[col_name].dtypes == 'object':
            unique_cat = len(data[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))



# Outlier detection
def find_outliers(x):
    q1 = np.percentile(x, 5)
    q3 = np.percentile(x, 95)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indicies = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indicies])

    return outlier_indicies, outlier_values

# Get the consumption categories
consumptions = data_new.loc[:, 'electricity_always_on' : 'electricity_total']
consumptions = list(consumptions.columns)

# Find values and index number of outliers
for c in consumptions:
    indices, values = find_outliers(data_new[c])
    #print ('Outliers in ', c, ' : ' , np.sort(indices))



# Feature 'month' in the end of the dataset
def reorder_features(data):
    data_temp = data.loc[:, 'electricity_always_on':]
    data_temp['month'] = data['month']
    return data_temp

data = reorder_features(data)

# Drop outliers and features with 0 values
def drop_outliers(data):
    data = data[(data.electricity_total < 2000) & (data.electricity_total != 0) & (data.electricity_other < 1000)]
    data = data.drop(['swimming_pool', 'sauna', 'electricity_water_heating', 'water_heating', 'grill',
                      'electricity_pool_or_sauna', 'space_heating', 'photo_voltaic', 'oven', 'hob',
                      'electricity_space_heating'], axis=1)
    data = data.reset_index(drop=True)

    return data



# Merge installation charasteristics to 'electric' and 'other'
def merge_characteristics(data):
    # merge water_heating, grill_heating and oven_heating
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




# Change categorical variables property_size and occupants to ordinal
def change_to_ordinal(data):
    # change property_size
    for i in data.index:
        for j in range(1, 6):
            if data['property_size'][i] == 'PS.0' + str(j):
                data['property_size'][i] = j


            # change occupants
            if data['occupants'][i] == 'OCC.0' + str(j):
                data['occupants'][i] = j


    return data




#change ordinal variable month to categorical
def change_to_categorical(data):
    # change month
    for i in data.index:
        for j in range(0, 13):
            if data['month'][i] == j:
                data['month'][i] = str(j) + 'th'
                break

    return data



#analyze_data(data)

#merge_characteristics(data)
#data = drop_outliers(data)
#change_to_ordinal(data)
#change_to_categorical(data)


merge_characteristics(data_new)
change_to_ordinal(data_new)
change_to_categorical(data_new)

#analyze_data(data_new)


# Transform consumption to %
def transfotm_to_percentage(data, features, base):
    for f in features:
        data[f] = (data[f]/data[base])*100
    data = data.dropna()
    return data



## Export final dataset to csv file
base = 'electricity_total'

# Old dataset
features = list(data_new.columns)
features = features[:10]
#data.to_csv('final_dataset_merged.csv', index=False)
#transfotm_to_percentage(data, features, base).to_csv('final_dataset_merged_percentage.csv', index=False)

# New dataset
features = list(data_new.columns)
features = features[:12]
#data_new.to_csv('final_dataset_merged_new.csv', index=False)
#transfotm_to_percentage(data_new, features, base).to_csv('final_dataset_merged_percentage_new.csv', index=False)

