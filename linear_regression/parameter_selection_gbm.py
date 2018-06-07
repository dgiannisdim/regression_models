import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
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

# Drop Nan values
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


# Create function to evaluate 'loss' parameter
def gbm_parameters_loss():
    loss = ['ls', 'lad', 'huber', 'quantile']

    r2_column = []
    smape_column = []
    rmse_column = []

    for l in loss:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss=l)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('loss', loss, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(gbm_parameters_loss())

## Write final dataframe to excel file
#gbm_parameters_loss(data, selected_features_rfe).to_excel('gbm_parameters_loss.xlsx', index=False)



# Create function to evaluate 'learning_rate' parameter
def gbm_parameters_learning_rate():
    learning_rate = [0.01, 0.1, 1, 2]
    learning_rate2 = np.arange(0.05, 0.51, 0.05)

    r2_column = []
    smape_column = []
    rmse_column = []

    for l in learning_rate2:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=l)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = create_dataframe('learning_rate', learning_rate2, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(gbm_parameters_learning_rate())

## Write final dataframe to excel file
#gbm_parameters_learning_rate(data, selected_features_rfe).to_excel('gbm_parameters_learning_rate.xlsx', index=False)
#gbm_parameters_learning_rate(data, selected_features_rfe).to_excel('gbm_parameters_learning_rate_detail.xlsx', index=False)



# Create function to evaluate 'n_estimators' parameter
def gbm_parameters_n_estimators():
    n_estimators = 10 ** np.arange(0, 6)
    n_estimators2 = np.arange(100, 1001, 100)

    r2_column = []
    smape_column = []
    rmse_column = []
    time_column = []

    for n in n_estimators2:
        t0 = time()

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=n)

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
#print(gbm_parameters_n_estimators())

## Write final dataframe to excel file
#gbm_parameters_n_estimators(data, selected_features_rfe).to_excel('gbm_parameters_n_estimators.xlsx', index=False)
#gbm_parameters_n_estimators(data, selected_features_rfe).to_excel('gbm_parameters_n_estimators_detail.xlsx', index=False)



# Create function to evaluate 'max_depth' parameter
def gbm_parameters_max_depth():
    max_depth = np.arange(1, 10)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_depth:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = create_dataframe('max_depth', max_depth, r2_column, smape_column, rmse_column)


    return df


## Print final dataframe
#print(gbm_parameters_max_depth())

## Write final dataframe to excel file
#gbm_parameters_max_depth(data, selected_features_rfe).to_excel('gbm_parameters_max_depth.xlsx', index=False)



# Create function to evaluate 'criterion' parameter
def gbm_parameters_criterion():
    criterion = ['friedman_mse', 'mse', 'mae']

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in criterion:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion=c)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('criterion', criterion, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(gbm_parameters_criterion())

## Write final dataframe to excel file
#gbm_parameters_criterion(data, selected_features_rfe).to_excel('gbm_parameters_criterion.xlsx', index=False)



# Create function to evaluate 'min_samples_split' parameter
def gbm_parameters_min_samples_split():
    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_samples_split:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('min_samples_split', min_samples_split, r2_column, smape_column, rmse_column)

    return df


## Print final dataframe
#print(gbm_parameters_min_samples_split())

## Write final dataframe to excel file
#gbm_parameters_min_samples_split(data, selected_features_rfe).to_excel('gbm_parameters_min_samples_split.xlsx', index=False)



# Create function to evaluate 'min_samples_leaf' parameter
def gbm_parameters_min_samples_leaf():
    min_samples_leaf = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_samples_leaf:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('min_samples_leaf', min_samples_leaf, r2_column, smape_column, rmse_column)

    return df


## Print final dataframe
#print(gbm_parameters_min_samples_leaf())

## Write final dataframe to excel file
#gbm_parameters_min_samples_leaf(data, selected_features_rfe).to_excel('gbm_parameters_min_samples_leaf.xlsx', index=False)



# Create function to evaluate 'min_weight_fraction_leaf' parameter
def gbm_parameters_min_weight_fraction_leaf():
    min_weight_fraction_leaf = np.arange(0,0.6, 0.1)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_weight_fraction_leaf:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=m)


        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('min_weight_fraction_leaf', min_weight_fraction_leaf, r2_column, smape_column, rmse_column)

    return df


## Print final dataframe
#print(gbm_parameters_min_weight_fraction_leaf())

## Write final dataframe to excel file
#gbm_parameters_min_weight_fraction_leaf(data, selected_features_rfe).to_excel('gbm_parameters_min_weight_fraction_leaf.xlsx', index=False)



# Create function to evaluate 'subsample' parameter
def gbm_parameters_subsample():
    subsample = np.arange(0.1, 1.1, 0.1)

    r2_column = []
    smape_column = []
    rmse_column = []

    for s in subsample:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0,
                                                 subsample=s)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('subsample', subsample, r2_column, smape_column, rmse_column)

    return df


## Print final dataframe
#print(gbm_parameters_subsample())

## Write final dataframe to excel file
#gbm_parameters_subsample(data, selected_features_rfe).to_excel('gbm_parameters_subsample.xlsx', index=False)



# Create function to evaluate 'max_features' parameter
def gbm_parameters_max_features():
    max_features = ['auto', 'sqrt', 'log2', None]
    max_features2 = np.arange(5, 30, 5)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_features2:
        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0,
                                                 subsample=1,
                                                 max_features=m)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('max_features', max_features2, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(gbm_parameters_max_features())

## Write final dataframe to excel file
#gbm_parameters_max_features(data, selected_features_rfe).to_excel('gbm_parameters_max_features_int.xlsx', index=False)

