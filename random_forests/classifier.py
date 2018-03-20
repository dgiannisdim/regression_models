import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn import cross_validation

pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe.csv')
selected_features_random_forest = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage.csv')
selected_features_random_forest_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest_percentage.csv')



def random_forest_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    for cf in selected_features_rfe.columns:
        consumption_selected = list(selected_features.loc[:, cf])
        consumption_selected = consumption_selected[: 25]

        X = data.loc[:, consumption_selected]
        y = data[cf]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        rf = RandomForestRegressor(n_estimators = 1000, random_state = 0)
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)

        print(cf, 'r2: ', r2_score(y_test, predictions))



random_forest_eval(data, selected_features_rfe)