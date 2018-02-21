import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r'C:\Energy_Consumption_Data\Installation Metadata.csv')
data2 = pd.read_csv(r'C:\Energy_Consumption_Data\Disaggregation Results.csv')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_row', 99)
 
'''
print(data.head())
print(data2.head())

'''

#explore dataset
def explore(data):
    print(data.head())
    print('\n')
    print(data.describe())
    print('\n')
    print(data.shape)

explore(data)


'''
#write dataframe to excel file
writer = pd.ExcelWriter('example.xlsx', engine = 'xlsxwriter')
a.to_excel(writer, 'sheet')
writer.save()
'''


#remove useless features
def del_useless_features(data, columns):
    return data.drop(columns, axis=1)



#define useless features
columns = ['Id', 'eui64', 'appliance_type', 'usage_in_kwh',
           'usage_frequency', 'score', 'period_first', 'period_last',
           'snapshot_taken_at', 'calculated_until', 'seconds_of_data_missing']

columns2 = ['eui64', 'disaggregation_id', 'year', 'gas_space_heating',
           'gas_water_heating', 'gas_cooking', 'gas_total', 'description']


data = del_useless_features(data,columns)
data2 = del_useless_features(data2,columns2)

#print(data.describe())

#how many months of each household
#print(data2['household_id'].value_counts())


#check features with few values
#print(sum(float(num) > 0 for num in data['sauna']))
arr=np.array(data['sauna']).astype(float)
num_sauna = len(arr[arr > 0])

arr=np.array(data['photo_voltaic']).astype(str)
num_photo_voltaic = len(arr[arr == 'SOL.01'])
#print("photo_voltaic appears in ", num_photo_voltaic, 'rows')
#print("sauna appears in ", num_sauna, 'rows')

arr=np.array(data['swimming_pool']).astype(float)
num_swim_pool = len(arr[arr > 0])
#print("swimming pool appears in ", num_swim_pool, 'rows')



#transform consumption to %
features2 = list(data2.columns.values)
del features2[0:2]


for f in features2:
    data2[f] = (data2[f]/data2['electricity_total'])*100


#remove NaN values (0 the raw data)
data2 = data2.dropna()

#sort data2 by household id, month
data2 = data2.sort_values(by=['household_id', 'month'])
data2 = data2.reset_index(drop=True)

#print(data2.loc[data2['household_id'] == 7])

#print(data.head())

features = list(data.columns.values)
del features[0]

'''
#create dataframe with installation columns
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
data2.to_csv('final.csv')

'''


