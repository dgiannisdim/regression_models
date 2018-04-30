import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


#read csv files
data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage.csv')
data_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage_new.csv')

#read feature selection datasets
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage.csv')
selected_features_rfe_new = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_new.csv')
selected_features_rfe_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage_new.csv')



#create datasets with only useful consumption categories (old and new)
data = data.drop(['electricity_space_heating','electricity_electric_vehicle'],axis=1)
data.to_csv('final_old.csv', index=False)
data_percentage = data_percentage.drop(['electricity_space_heating','electricity_electric_vehicle'],axis=1)
data_percentage.to_csv('final_old_percentage.csv', index=False)

data_new = data_new.drop(['electricity_space_heating','electricity_electric_vehicle',
                     'electricity_water_heating',
                     'electricity_pool_or_sauna'],axis=1)
data_new.to_csv('final_new.csv', index=False)
data_percentage_new = data_percentage_new(['electricity_space_heating','electricity_electric_vehicle',
                     'electricity_water_heating',
                     'electricity_pool_or_sauna'],axis=1)
data_percentage_new.to_csv('final_new_percentage.csv', index=False)

