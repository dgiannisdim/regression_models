import pandas as pd
import numpy as np
from visualization import del_useless_features


data = pd.read_csv(r'C:\Energy_Consumption_Data\Installation Metadata.csv')
data2 = pd.read_csv(r'C:\Energy_Consumption_Data\Disaggregation Results.csv')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_row', 99)




#define useless data
columns = ['Id', 'eui64', 'appliance_type', 'usage_in_kwh',
           'usage_frequency', 'score', 'period_first', 'period_last',
           'snapshot_taken_at', 'calculated_until', 'seconds_of_data_missing']

columns2 = ['eui64', 'disaggregation_id', 'year', 'gas_space_heating',
           'gas_water_heating', 'gas_cooking', 'gas_total', 'description']



data = del_useless_features(data2,columns2)
data2 = del_useless_features(data2,columns2)






#data = pd.get_dummies(data)