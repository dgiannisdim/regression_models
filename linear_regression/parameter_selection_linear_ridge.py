import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
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


# Create function to evaluate 'alpha_range' parameter
def linear_parameters_ridge_alpha():
    alpha_range = [0, 1, 10, 100, 1000]
    alpha_range2 = range(1, 20, 2)

    r2_column = []
    smape_column = []
    rmse_column = []

    for a in alpha_range2:
        clf = linear_model.Ridge(random_state=0, alpha=a)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('alpha_range', alpha_range2, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(linear_parameters_ridge_alpha())

## Write final dataframe to excel file
#linear_parameters_ridge_alpha(data, selected_features_rfe).to_excel('linear_parameters_ridge_alpha.xlsx', index=False)
#linear_parameters_ridge_alpha(data, selected_features_rfe).to_excel('linear_parameters_ridge_alpha_detail.xlsx', index=False)



# Create function to evaluate 'solver' parameter
def linear_parameters_ridge_solver():
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    r2_column = []
    smape_column = []
    rmse_column = []

    for s in solver:
        clf = linear_model.Ridge(random_state=0, alpha=13, solver=s)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('solver', solver, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(linear_parameters_ridge_solver())

## Write final dataframe to excel file
#linear_parameters_ridge_solver(data, selected_features_rfe).to_excel('linear_parameters_ridge_solver.xlsx', index=False)



# Create function to evaluate 'fit_intercept' parameter
def linear_parameters_ridge_fit_intercept():
    fit_intercept = [True, False]

    r2_column = []
    smape_column = []
    rmse_column = []

    for f in fit_intercept:
        clf = linear_model.Ridge(random_state=0, alpha=13, solver='sag',
                                 fit_intercept=f)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('fit_intercept', fit_intercept, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(linear_parameters_ridge_fit_intercept())

## Write final dataframe to excel file
#linear_parameters_ridge_fit_intercept(data, selected_features_rfe).to_excel('linear_parameters_ridge_fit_intercept.xlsx', index=False)



# Create function to evaluate 'max_iter' parameter
def linear_parameters_ridge_max_iter():
    max_iter = np.arange(5, 55, 5)
    max_iter2 = np.arange(10, 21)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_iter2:
        clf = linear_model.Ridge(random_state=0, alpha=13, solver='sag',
                                 fit_intercept=False, max_iter=m)


        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('max_iter', max_iter2, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(linear_parameters_ridge_max_iter())

## Write final dataframe to excel file
#linear_parameters_ridge_max_iter(data, selected_features_rfe).to_excel('linear_parameters_ridge_max_iter.xlsx', index=False)
#linear_parameters_ridge_max_iter(data, selected_features_rfe).to_excel('linear_parameters_ridge_max_iter_detail.xlsx', index=False)