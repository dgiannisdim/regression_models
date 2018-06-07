import pandas as pd
import numpy as np
import random


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


# Read csv files
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage.csv')
data_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage_new.csv')

# Read feature selection datasets
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')



# Sort drop NaN values
def sort_and_drop(data):
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data


data = sort_and_drop(data)
data_new = sort_and_drop(data_new)
data_percentage = sort_and_drop(data_percentage)
data_percentage_new = sort_and_drop(data_percentage_new)


#create datasets with only useful consumption categories (old and new)
data = data.drop(['electricity_space_heating','electricity_electric_vehicle'],axis=1)
#data.to_csv('final_old.csv', index=False)
data_percentage = data_percentage.drop(['electricity_space_heating','electricity_electric_vehicle'],axis=1)
#data_percentage.to_csv('final_old_percentage.csv', index=False)

data_new = data_new.drop(['electricity_always_on', 'electricity_refrigeration',
       'electricity_cooking', 'electricity_laundry', 'electricity_lighting',
       'electricity_entertainment', 'electricity_other', 'electricity_total'],axis=1)
#data_new.to_csv('final_new.csv', index=False)
data_percentage_new = data_percentage_new.drop(['electricity_always_on', 'electricity_refrigeration',
       'electricity_cooking', 'electricity_laundry', 'electricity_lighting',
       'electricity_entertainment', 'electricity_other', 'electricity_total'],axis=1)
#data_percentage_new.to_csv('final_new_percentage.csv', index=False)



# Function to replace values of features related to a consumption category
def replace_missing():
    electricity_space_heating_new.space_heating = 'electric'
    electricity_water_heating_new.water_heating = 'electric'
    electricity_electric_vehicle_new.ev_charger = 1

    # create a random sauna list of 0 and 1
    sauna = [random.randrange(0, 2) for _ in range(0, len(electricity_pool_or_sauna_new))]
    electricity_pool_or_sauna_new.sauna = sauna

    #set swimming pool to 1 if a household does not have a sauna
    for i in range(0, len(electricity_pool_or_sauna_new)):
        if electricity_pool_or_sauna_new.sauna[i] == 0:
            electricity_pool_or_sauna_new.swimming_pool[i] = 1


    electricity_space_heating_percentage_new.space_heating = 'electric'
    electricity_water_heating_percentage_new.water_heating = 'electric'
    electricity_electric_vehicle_percentage_new.ev_charger = 1

    # create a random sauna list of 0 and 1
    sauna = [random.randrange(0, 2) for _ in range(0, len(electricity_pool_or_sauna_percentage_new))]
    electricity_pool_or_sauna_percentage_new.sauna = sauna

    #set swimming pool to 1 if a household does not have a sauna (percentage)
    for i in range(0, len(electricity_pool_or_sauna_percentage_new)):
        if electricity_pool_or_sauna_percentage_new.sauna[i] == 0:
            electricity_pool_or_sauna_percentage_new.swimming_pool[i] = 1




# Create vectors with 4 categories of new dataset (raw values)
# Keep only rows with non zero electricity for categories: electricity_space_heating, electricity_water_heating,
# electricity_electric_vehicle, electricity_pool_or_sauna
electricity_space_heating_new = data_new.drop(['electricity_water_heating',
                                               'electricity_electric_vehicle',
                                               'electricity_pool_or_sauna'],axis=1)
electricity_space_heating_new = electricity_space_heating_new[electricity_space_heating_new.electricity_space_heating != 0]


electricity_water_heating_new = data_new.drop(['electricity_space_heating',
                                               'electricity_electric_vehicle',
                                               'electricity_pool_or_sauna'],axis=1)
electricity_water_heating_new = electricity_water_heating_new[electricity_water_heating_new.electricity_water_heating != 0]


electricity_electric_vehicle_new = data_new.drop(['electricity_water_heating',
                                               'electricity_space_heating',
                                               'electricity_pool_or_sauna'],axis=1)
electricity_electric_vehicle_new = electricity_electric_vehicle_new[electricity_electric_vehicle_new.electricity_electric_vehicle != 0]


electricity_pool_or_sauna_new = data_new.drop(['electricity_water_heating',
                                               'electricity_electric_vehicle',
                                               'electricity_space_heating'],axis=1)
electricity_pool_or_sauna_new = electricity_pool_or_sauna_new[electricity_pool_or_sauna_new.electricity_pool_or_sauna != 0]


# Drop NaN values and reset index
electricity_space_heating_new = sort_and_drop(electricity_space_heating_new)
electricity_water_heating_new = sort_and_drop(electricity_water_heating_new)
electricity_electric_vehicle_new = sort_and_drop(electricity_electric_vehicle_new)
electricity_pool_or_sauna_new = sort_and_drop(electricity_pool_or_sauna_new)




#create vectors with 4 categories of new dataset (percentage)
# Keep only rows with non zero electricity for categories: electricity_space_heating, electricity_water_heating,
# electricity_electric_vehicle, electricity_pool_or_sauna
electricity_space_heating_percentage_new = data_percentage_new.drop(['electricity_water_heating',
                                               'electricity_electric_vehicle',
                                               'electricity_pool_or_sauna'],axis=1)
electricity_space_heating_percentage_new = electricity_space_heating_percentage_new[electricity_space_heating_percentage_new.electricity_space_heating != 0]


electricity_water_heating_percentage_new = data_percentage_new.drop(['electricity_space_heating',
                                               'electricity_electric_vehicle',
                                               'electricity_pool_or_sauna'],axis=1)
electricity_water_heating_percentage_new = electricity_water_heating_percentage_new[electricity_water_heating_percentage_new.electricity_water_heating != 0]


electricity_electric_vehicle_percentage_new = data_percentage_new.drop(['electricity_water_heating',
                                               'electricity_space_heating',
                                               'electricity_pool_or_sauna'],axis=1)
electricity_electric_vehicle_percentage_new = electricity_electric_vehicle_percentage_new[electricity_electric_vehicle_percentage_new.electricity_electric_vehicle != 0]


electricity_pool_or_sauna_percentage_new = data_percentage_new.drop(['electricity_water_heating',
                                               'electricity_electric_vehicle',
                                               'electricity_space_heating'],axis=1)
electricity_pool_or_sauna_percentage_new = electricity_pool_or_sauna_percentage_new[electricity_pool_or_sauna_percentage_new.electricity_pool_or_sauna != 0]


# Drop NaN values and reset index
electricity_space_heating_percentage_new = sort_and_drop(electricity_space_heating_percentage_new)
electricity_water_heating_percentage_new = sort_and_drop(electricity_water_heating_percentage_new)
electricity_electric_vehicle_percentage_new = sort_and_drop(electricity_electric_vehicle_percentage_new)
electricity_pool_or_sauna_percentage_new = sort_and_drop(electricity_pool_or_sauna_percentage_new)


replace_missing()

#write new dataframes to csv
def write_to_csv():
    electricity_space_heating_new.to_csv('electricity_space_heating_new.csv', index=False)
    electricity_water_heating_new.to_csv('electricity_water_heating_new.csv', index=False)
    electricity_electric_vehicle_new.to_csv('electricity_electric_vehicle_new.csv', index=False)
    electricity_pool_or_sauna_new.to_csv('electricity_pool_or_sauna_new.csv', index=False)

    electricity_space_heating_percentage_new.to_csv('electricity_space_heating_percentage_new.csv', index=False)
    electricity_water_heating_percentage_new.to_csv('electricity_water_heating_percentage_new.csv', index=False)
    electricity_electric_vehicle_percentage_new.to_csv('electricity_electric_vehicle_percentage_new.csv', index=False)
    electricity_pool_or_sauna_percentage_new.to_csv('electricity_pool_or_sauna_percentage_new.csv', index=False)


#write_to_csv()
