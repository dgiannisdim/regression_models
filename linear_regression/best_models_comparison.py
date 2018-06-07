import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection
from statistics import mean
from sklearn.ensemble import RandomForestRegressor




pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

#import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')


#import new datasets
data_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage_new.csv')


data = data.dropna()
data_percentage = data_percentage.dropna()
data_new = data_new.dropna()
data_percentage_new = data_percentage_new.dropna()



def calculate_metrics(y_test, predictions):
    SS_Residual = sum((y_test - predictions) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (float(SS_Residual)) / SS_Total
    smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return r2, smape, rmse


def compare_best_models(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])
    X = data.loc[:, consumption_selected]
    # X = preprocessing.StandardScaler().fit_transform(X)
    # X = sm.add_constant(X)
    y = data['electricity_total']

    random_range = range(5, 500, 50)
    r2_svr_rbf = 0
    smape_svr_rbf = 0
    rmse_svr_rbf = 0

    r2_column = []
    smape_column = []
    rmse_column = []

    for random in random_range:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=random, test_size=0.25)

        clf = SVR(kernel='rbf', C=5000, gamma=0.025, epsilon=80)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        r2, smape, rmse = calculate_metrics(y_test, predictions)
        r2_svr_rbf = r2_svr_rbf + r2
        smape_svr_rbf = smape_svr_rbf + smape
        rmse_svr_rbf = rmse_svr_rbf + rmse



    r2_column.append(r2_svr_rbf/10)
    smape_column.append(smape_svr_rbf/10)
    rmse_column.append(rmse_svr_rbf/10)


    method = ['SVR (Rbf)']
    method2 = ['SVR (Rbf)', 'SVR (Polynomial)', 'SVR (Linear)', 'SVR (Sigmoid)',
              'Random Forests', 'GBM', 'Linear (Lasso)', 'Linear (Ridge)']

    df = pd.DataFrame({'method': method, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column},
                      columns=['method', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(compare_best_models(data, selected_features_rfe))
#compare_best_models(data, selected_features_rfe).to_excel('compare_best_models.xlsx', index=False)

