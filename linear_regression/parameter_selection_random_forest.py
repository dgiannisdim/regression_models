import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from time import time
from basic_functions import calculate_metrics


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

# Import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')


# Import new datasets
data_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage_new.csv')

#drop Nan values
data = data.dropna()
data_percentage = data_percentage.dropna()
data_new = data_new.dropna()
data_percentage_new = data_percentage_new.dropna()

# One hot encoding for categorical features
data = pd.get_dummies(data, drop_first=True)

# Create a list with most important features related with electricity_total
consumption_selected = list(selected_features_rfe.loc[:, 'electricity_total'])

# X is the dataframe of selected features for electricity_total
X = data.loc[:, consumption_selected]

# y is the electricity_total column
y = data['electricity_total']

# Split the dataset into training (75%) and testing (25%) set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                                     random_state=0, test_size=0.25)

# Create dataframe to compare different values of parameters
def create_dataframe(param_name, param, r2_column, smape_column, rmse_column):
    df = pd.DataFrame({param_name: param, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column, },
                      columns=[param_name, 'R2', 'SMAPE', 'RMSE'])
    return df



# Create function to evaluate 'max_features' parameter
def random_forest_parameters_max_features():
    max_features = ['auto', 'sqrt', 'log2']

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_features:
        clf = RandomForestRegressor(random_state=420, max_features=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('max_features', max_features, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(random_forest_parameters_max_features())

## Write final dataframe to excel file
#random_forest_parameters_max_features(data, selected_features_rfe).to_excel('random_forest_parameters_max_features.xlsx', index=False)


# Create function to evaluate 'min_samples_split' parameter
def random_forest_parameters_min_samples_split():
    min_samples_split = 2 ** np.arange(1, 6)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_samples_split:
        clf = RandomForestRegressor(random_state=420,max_features='sqrt',
                                    min_samples_split=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('min_samples_split', min_samples_split, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(random_forest_parameters_min_samples_split())

## Write final dataframe to excel file
#random_forest_parameters_min_samples_split(data, selected_features_rfe).to_excel('random_forest_parameters_min_samples_split.xlsx', index=False)


# Create function to evaluate 'criterion' parameter
def random_forest_parameters_criterion():
    criterion = ['mse', 'mae']

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in criterion:
        clf = RandomForestRegressor(random_state=420,max_features='sqrt',
                                    min_samples_split=8, criterion=c)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('criterion', criterion, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(random_forest_parameters_criterion())

## Write final dataframe to excel file
#random_forest_parameters_criterion(data, selected_features_rfe).to_excel('random_forest_parameters_criterion.xlsx', index=False)


# Create function to evaluate 'max_depth' parameter
def random_forest_parameters_max_depth():
    max_depth = [1, 10, 20, 30, 40, 50, None]
    max_depth2 = [10, 12, 14, 16, 18, 20 , None]

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_depth:
        clf = RandomForestRegressor(random_state=420,max_features='sqrt',
                                    min_samples_split=8, criterion='mse',
                                    max_depth=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('max_depth', max_depth, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(random_forest_parameters_max_depth())

## Write final dataframe to excel file
#random_forest_parameters_max_depth(data, selected_features_rfe).to_excel('random_forest_parameters_max_depth.xlsx', index=False)
#random_forest_parameters_max_depth(data, selected_features_rfe).to_excel('random_forest_parameters_max_depth_detail.xlsx', index=False)


# Create function to evaluate 'min_samples_leaf' parameter
def random_forest_parameters_min_samples_leaf():
    min_samples_leaf = [1, 2, 4, 6, 8, 10]

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_samples_leaf:
        clf = RandomForestRegressor(random_state=420,max_features='sqrt',
                                    min_samples_split=8, criterion='mse',
                                    max_depth=14, min_samples_leaf=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('min_samples_leaf', min_samples_leaf, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(random_forest_parameters_min_samples_leaf())

## Write final dataframe to excel file
#random_forest_parameters_min_samples_leaf(data, selected_features_rfe).to_excel('random_forest_parameters_min_samples_leaf.xlsx', index=False)


# Create function to evaluate 'bootstrap' parameter
def random_forest_parameters_bootstrap():
    bootstrap = [True, False]
    r2_column = []
    smape_column = []
    rmse_column = []


    for b in bootstrap:
        clf = RandomForestRegressor(random_state=420,max_features='sqrt',
                                    min_samples_split=8, criterion='mse',
                                    max_depth=12, min_samples_leaf=1,
                                    bootstrap=b)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('bootstrap', bootstrap, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(random_forest_parameters_bootstrap())

## Write final dataframe to excel file
#random_forest_parameters_bootstrap(data, selected_features_rfe).to_excel('random_forest_parameters_bootstrap.xlsx', index=False)


# Create function to evaluate 'n_estimators' parameter
def random_forest_parameters_n_estimators():
    n_estimators = 10 ** np.arange(0, 6)
    n_estimators2 = np.arange(1000, 11000, 1000)

    r2_column = []
    smape_column = []
    rmse_column = []
    time_column = []

    for n in n_estimators2:
        t0 = time()

        clf = RandomForestRegressor(random_state=420,max_features='sqrt',
                                    min_samples_split=8, criterion='mse',
                                    max_depth=12, min_samples_leaf=1,
                                    bootstrap=False, n_estimators=n)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)
        time_column.append(time() - t0)

    df = pd.DataFrame({'n_estimators': n_estimators2, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,
                       'Time(s)': time_column},
                      columns=['n_estimators', 'R2', 'SMAPE', 'RMSE', 'Time(s)'])

    return df

## Print final dataframe
#print(random_forest_parameters_n_estimators())

## Write final dataframe to excel file
#random_forest_parameters_n_estimators(data, selected_features_rfe).to_excel('random_forest_parameters_n_estimators.xlsx', index=False)
#random_forest_parameters_n_estimators(data, selected_features_rfe).to_excel('random_forest_parameters_n_estimators_detail.xlsx', index=False)



