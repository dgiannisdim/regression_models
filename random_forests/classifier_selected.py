import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn import cross_validation, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import pickle
from sklearn.externals import joblib

pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)



# Import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_old.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_old_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')

# Import new datasets
selected_features_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_new.csv')
selected_features_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_percentage_new.csv')

electricity_space_heating_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_space_heating_new.csv')
electricity_water_heating_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_water_heating_new.csv')
electricity_electric_vehicle_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_electric_vehicle_new.csv')
electricity_pool_or_sauna_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_pool_or_sauna_new.csv')

electricity_space_heating_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_space_heating_percentage_new.csv')
electricity_water_heating_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_water_heating_percentage_new.csv')
electricity_electric_vehicle_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_electric_vehicle_percentage_new.csv')
electricity_pool_or_sauna_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/electricity_pool_or_sauna_percentage_new.csv')


def calculate_metrics(X_test, y_test, predictions):
    SS_Residual = sum((y_test - predictions) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (float(SS_Residual)) / SS_Total
    smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return r2, smape, rmse



##create model for every category with selected features (percentage)


# Initialize columns of metrics
r2_column = []
smape_column = []
rmse_column = []

# Create model with best params
clf = RandomForestRegressor(random_state=420, max_features='sqrt',
                            min_samples_split=8, criterion='mse',
                            max_depth=12, min_samples_leaf=1,
                            bootstrap=False, n_estimators=7000)

# Function to calculate metrics for consumprtions of old dataset (except electricity_total)
def random_forest_eval_old(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_name_list = ['electricity_always_on', 'electricity_refrigeration',
                             'electricity_cooking', 'electricity_laundry', 'electricity_lighting',
                             'electricity_entertainment', 'electricity_other']


    for consumption in consumption_name_list:

        features = list(selected_features.loc[:, consumption])

        #features = features[: 20]


        X = data.loc[:, features]
        # X = preprocessing.StandardScaler().fit_transform(X)
        # X = sm.add_constant(X)
        y = data[consumption]

        random_range = range(0, 500, 50)
        r2_sum = 0
        smape_sum = 0
        rmse_sum = 0

        for random in random_range:
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                random_state=random, test_size=0.25)


            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            r2, smape, rmse = calculate_metrics(X_test, y_test, predictions)
            r2_sum = r2_sum + r2
            smape_sum = smape_sum + smape
            rmse_sum = rmse_sum + rmse

        r2_column.append(r2_sum / 10)
        smape_column.append(smape_sum / 10)
        rmse_column.append(rmse_sum / 10)

    return r2_column, smape_column, rmse_column


# Function to calculate metrics for consumprtions of new dataset and electricity_total of old dataset
def random_forest_eval_new(data, consumption, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    features = list(selected_features.loc[:, consumption])

    features = features[: 20]

    X = data.loc[:, features]
    # X = preprocessing.StandardScaler().fit_transform(X)
    # X = sm.add_constant(X)
    y = data[consumption]

    random_range = range(0, 500, 50)
    r2_sum = 0
    smape_sum = 0
    rmse_sum = 0

    for random in random_range:

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=random, test_size=0.25)


        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        r2, smape, rmse = calculate_metrics(X_test, y_test, predictions)

        r2_sum = r2_sum + r2
        smape_sum = smape_sum + smape
        rmse_sum = rmse_sum + rmse

    r2 = r2_sum / 10
    smape = smape_sum / 10
    rmse = rmse_sum / 10

    return r2, smape, rmse

# Keep the names of consumption variables
consumption_name_list = ['electricity_total', 'electricity_always_on', 'electricity_refrigeration',
       'electricity_cooking', 'electricity_laundry', 'electricity_lighting',
       'electricity_entertainment', 'electricity_other', 'electricity_space_heating',
        'electricity_water_heating','electricity_electric_vehicle',
        'electricity_pool_or_sauna']


# Calculate metrics for electricity_total with real values
data_temp = data.drop(['electricity_always_on', 'electricity_refrigeration',
                             'electricity_cooking', 'electricity_laundry', 'electricity_lighting',
                             'electricity_entertainment', 'electricity_other'], axis=1)
r2, smape, rmse = random_forest_eval_new(data_temp,'electricity_total', selected_features_rfe)
r2_column.append(r2)
smape_column.append(smape)
rmse_column.append(rmse)

# Calculate metrics for electricity_always_on, electricity_refrigeration, electricity_cooking,
# electricity,_laundry, electricity_lighting, electricity_entertainment,electricity_other
#  with percentage values
r2_column, smape_column, rmse_column = random_forest_eval_old(data_percentage, selected_features_rfe)


# Calculate metrics for electricity_space_heating with percentage values
r2, smape, rmse = random_forest_eval_new(electricity_space_heating_percentage_new,'electricity_space_heating', selected_features_new)
r2_column.append(r2)
smape_column.append(smape)
rmse_column.append(rmse)


# Calculate metrics for electricity_water_heating with percentage values
r2, smape, rmse = random_forest_eval_new(electricity_water_heating_percentage_new,'electricity_water_heating', selected_features_new)
r2_column.append(r2)
smape_column.append(smape)
rmse_column.append(rmse)


# Calculate metrics for electricity_electric_vehicle with percentage values
r2, smape, rmse = random_forest_eval_new(electricity_electric_vehicle_percentage_new,'electricity_electric_vehicle', selected_features_new)
r2_column.append(r2)
smape_column.append(smape)
rmse_column.append(rmse)


# Calculate metrics for electricity_pool_or_sauna with percentage values
r2, smape, rmse = random_forest_eval_new(electricity_pool_or_sauna_percentage_new,'electricity_pool_or_sauna', selected_features_new)
r2_column.append(r2)
smape_column.append(smape)
rmse_column.append(rmse)


# Create dataframe with the results for all consumptions
df = pd.DataFrame({'Consumption': consumption_name_list, 'R2': r2_column, 'SMAPE': smape_column, 'RMSE': rmse_column},
                      columns=['Consumption', 'R2', 'SMAPE', 'RMSE'])



# Print the dataframe
print(df)

# Write dataframe to excel file
#df.to_excel('models_selected_features.xlsx', index=False)