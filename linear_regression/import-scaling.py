import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Energy_Consumption_Data\Installation Metadata.csv')
data2 = pd.read_csv(r'C:\Energy_Consumption_Data\Disaggregation Results.csv')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_row', 99)



#explore dataset
def explore(data):
    print(data.head())
    print('\n')
    print(data.describe())
    print('\n')
    print(data.shape)

#explore(data)


'''
#write dataframe to excel file
writer = pd.ExcelWriter('example.xlsx', engine = 'xlsxwriter')
a.to_excel(writer, 'sheet')
writer.save()
'''


#remove useless data
def del_useless_features(data, columns):
    return data.drop(columns, axis=1)



#define useless data
columns = ['Id', 'eui64', 'appliance_type', 'usage_in_kwh',
           'usage_frequency', 'score', 'period_first', 'period_last',
           'snapshot_taken_at', 'calculated_until', 'seconds_of_data_missing']

columns2 = ['eui64', 'disaggregation_id', 'year', 'gas_space_heating',
           'gas_water_heating', 'gas_cooking', 'gas_total', 'description']


data = del_useless_features(data,columns)
data2 = del_useless_features(data2,columns2)


#transform consumption to %
features2 = list(data2.columns.values)
del features2[0:2]


def transfotm_to_percentage(data, features, base):
    for f in features:
        data[f] = (data[f]/data[base])*100
    data = data.dropna()
    return data



#sort data2 by household id, month
data2 = data2.sort_values(by=['household_id', 'month'])
data2 = data2.reset_index(drop=True)



'''
#merge Installation and Disaggregation to one dataset
def merge_datasets(data):
    features = list(data.columns.values)
    del features[0]
    
    e = pd.DataFrame(np.nan, index=range(0, len(data2.index)), columns=features)
    
    for f in features:
        for i in range(0, len(data2.index)):
            for j in range(0,len(data.index)):
                if data2['household_id'][i] == data['household_id'][j]:
                    e[f][i] = data[f][j]
                    break
    
    
    #merge dataframes into a final
    data2[features] = e
    print(data2)
    data2.to_csv('final_temp.csv')

'''

