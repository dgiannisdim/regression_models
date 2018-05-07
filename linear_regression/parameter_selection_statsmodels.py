import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection
import pickle
from sklearn.pipeline import make_pipeline


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

#import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe.csv')
selected_features_random_forest = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage.csv')
selected_features_random_forest_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest_percentage.csv')

#import new datasets
data_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage_new.csv')
selected_features_rfe_new = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_new.csv')
selected_features_rfe_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage_new.csv')

selected_features_rfe_new2 = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_new2.csv')


data = data.dropna()
data_percentage = data_percentage.dropna()
data_new = data_new.dropna()
data_percentage_new = data_percentage_new.dropna()




def linear_parameters(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    fit_intercept = [True, False]

    r2_column = []
    smape_column = []
    rmse_column = []

    for fit in fit_intercept:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = sm.WLS()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'fit_intercept': fit_intercept, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['fit_intercept', 'R2', 'SMAPE', 'RMSE'])

    return df


print(linear_parameters(data, selected_features_rfe))

