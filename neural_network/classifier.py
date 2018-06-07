import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_old.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression/final_old_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe.csv')
selected_features_random_forest = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage.csv')
selected_features_random_forest_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest_percentage.csv')

#import new datasets
data_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage_new.csv')


data = data.dropna()
data_percentage = data_percentage.dropna()
data_new = data_new.dropna()
data_percentage_new = data_percentage_new.dropna()



electricity_space_heating_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_space_heating_new.csv')
electricity_water_heating_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_water_heating_new.csv')
electricity_electric_vehicle_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_electric_vehicle_new.csv')
electricity_pool_or_sauna_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_pool_or_sauna_new.csv')

electricity_space_heating_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_space_heating_percentage_new.csv')
electricity_water_heating_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_water_heating_percentage_new.csv')
electricity_electric_vehicle_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_electric_vehicle_percentage_new.csv')
electricity_pool_or_sauna_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/electricity_pool_or_sauna_percentage_new.csv')
data_percentage_new = data_percentage_new.dropna()

def neural_eval(data):
    data = pd.get_dummies(data, drop_first=True)

    consumption_name_list = ['electricity_always_on', 'electricity_refrigeration',
                             'electricity_cooking', 'electricity_laundry', 'electricity_lighting',
                             'electricity_entertainment', 'electricity_other']

    for cf in consumption_name_list:
        X = data.loc[:, 'property_size':]
        y = data[cf]


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25,
                                                                             random_state=0)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        mlp = MLPRegressor(hidden_layer_sizes=(5,),
                           alpha=0.001, learning_rate_init=0.005,
                           max_iter=1000)
        mlp.fit(X_train, y_train)

        predictions = mlp.predict(X_test)

        print(cf, 'r2: ', r2_score(y_test, predictions))


neural_eval(data_percentage)




'''
rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)

print(r2_score(test_labels, predictions))

'''