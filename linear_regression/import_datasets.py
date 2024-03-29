import pandas as pd
import numpy as np



data = pd.read_csv(r'C:\regression_models\linear_regression/Installation Metadata.csv')
data2 = pd.read_csv(r'C:\regression_models\linear_regression/Disaggregation Results.csv')

data_new = pd.read_csv(r'C:\regression_models\linear_regression/Installation Metadata_new.csv', sep=',"')
data2_new = pd.read_csv(r'C:\regression_models\linear_regression/Disaggregation Results_new.csv', sep=',"')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_row', 9999)


#explore dataset
def explore(data):
    print('Head of the dataframe: ')
    print(data.head())
    print('\n')
    print('Describe the dataframe: ')
    print(data.describe())
    print('\n')
    print('Shape of the dataframe: ')
    print(data.shape)




#remove useless data
def del_useless_features(data, columns):
    return data.drop(columns, axis=1)



#define useless data for each dataframe
columns = ['Id', 'eui64', 'appliance_type', 'usage_in_kwh',
           'usage_frequency', 'score', 'period_first', 'period_last',
           'snapshot_taken_at', 'calculated_until', 'seconds_of_data_missing']

columns2 = ['eui64', 'disaggregation_id', 'year', 'gas_space_heating',
           'gas_water_heating', 'gas_cooking', 'gas_total', 'description']


data = del_useless_features(data,columns)
data2 = del_useless_features(data2,columns2)



#sort data2 by household id, month
data2 = data2.sort_values(by=['household_id', 'month'])
data2 = data2.reset_index(drop=True)



##create dataframes from new dataset
columns_new = ["eui64","property_type","property_size","property_age","property_ownership",
            "occupants","occupant_type","space_heating","water_heating","stove_heating",
            "grill_heating","oven_heating","photo_voltaic","fridge","freezer",
            "combi_fridge_freezer","oven","grill","hob","microwave","kettle","toaster",
            "dishwasher","washing_machine","tumble_dryer","iron","tv","dvd_or_blueray",
            "digital_tv_box","games_console","computer","tablet","electric_shower",
            "ev_charger","swimming_pool","sauna","month","year"]

columns2_new = ["eui64","electricity_always_on","electricity_refrigeration","electricity_space_heating",
               "electricity_water_heating","electricity_cooking","electricity_laundry",
               "electricity_lighting","electricity_entertainment","electricity_electric_vehicle",
               "electricity_pool_or_sauna","electricity_other","electricity_total",
               "gas_space_heating","gas_water_heating","gas_cooking","gas_total","month","year"]


data_new = data_new.replace('"', '', regex=True)
data_new.columns = columns_new

data2_new = data2_new.replace('"', '', regex=True)
data2_new.columns = columns2_new


def check_na_values(data):
    data = data.replace('', np.nan, regex=True)

    return data


data_new = check_na_values(data_new)
data2_new = check_na_values(data2_new)



#keep only 1 row of installation metadata for each household
data_new = data_new.drop_duplicates(subset=['eui64'], keep='first')
data_new = data_new.reset_index(drop=True)
data = data.drop_duplicates(subset=['household_id'], keep='first')
data = data.reset_index(drop=True)


#define useless data for each dataframe
columns_to_drop = ["month","year"]
columns_to_drop2 = ["gas_space_heating","gas_water_heating","gas_cooking","gas_total","year"]
data_new = del_useless_features(data_new,columns_to_drop)
data2_new = del_useless_features(data2_new,columns_to_drop2)


# Replace Nan values with most frequent of each feature
def replace_nan(data):
    data = data.loc[:, 'property_type':]
    features = data.columns

    # Create array with most frequent value of each feature
    for f in features:
        # Most frequent value of each feature
        freq = data[f].value_counts().index[0]

        # If there is a Nan value replace with most frequent
        for i in range(0, len(data.index)):
            # NaN is always != NaN
            if data_new[f][i] != data_new[f][i]:
                data_new[f][i] = freq

    return data_new


data_new = replace_nan(data_new)



#merge installation and disaggregation datasets
def merge_datasets(data, data2, id):
    # Keep feature names
    features = list(data.columns)
    # Delete household_id column
    del features[0]

    # Create empty dataframe for merging
    e = pd.DataFrame(np.nan, index=range(0, len(data2.index)), columns=features)

    # merge datasets by id
    for i in range(0, len(data.index)):
        for j in range(0, len(data2.index)):
            if data[id][i] == data2[id][j]:
                e.iloc[j] = data.iloc[i]

    # merge dataframes into a final
    data2[features] = e

    return data2


##write datasets to csv
#old dataset
name = 'final_dataset.csv'
id = 'household_id'
#merge_datasets(data, data2, id).to_csv(name, index=False)

#new dataset
name = 'final_dataset_new.csv'
id = 'eui64'
#merge_datasets(data_new, data2_new, id).to_csv(name, index=False)



