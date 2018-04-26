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




def random_forest_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    r_squared_column = []
    r_squared_adjusted_column = []
    smape_column = []
    explained_variance_column = []
    root_mean_square_error_column = []

    for cf in selected_features_rfe.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        X = data.loc[:, consumption_selected]
        y = data[cf]
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        clf = RandomForestRegressor(n_estimators = 1000, random_state = 0)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        #save the model
        switch = False
        if switch:
            model_name = cf + '_percentage_model_random_forest.pkl'
            model_pkl = open(model_name, 'wb')
            pickle.dump(clf, model_pkl)


        #calculate r2
        r2 = r2_score(y_test, predictions)

        # calculate adjusted r2
        rsquared_adj = 1 - (1 - r2) * (len(y_test) - 1) / (
                    len(y_test) - X_test .shape[1] - 1)

        # calculate SMAPE
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))

        # create columns to make dataframe with metrics for evaluation
        r_squared_column.append(r2)
        r_squared_adjusted_column.append(rsquared_adj)
        explained_variance_column.append(explained_variance_score(y_test, predictions))
        root_mean_square_error_column.append(np.sqrt(mean_squared_error(y_test, predictions)))
        smape_column.append(smape)

    df = pd.DataFrame({'consumption': selected_features.columns,
                       'r_squared': r_squared_column,
                       'adjusted_r_squared': r_squared_adjusted_column,
                       'smape': smape_column,
                       'root_mean_squared_error': root_mean_square_error_column,
                       'explained_variance_score': explained_variance_column},
                      columns=['consumption', 'r_squared', 'adjusted_r_squared', 'smape',
                               'root_mean_squared_error', 'explained_variance_score'])

    return df


#random_forest_eval(data_percentage, selected_features_rfe_percentage)
print(random_forest_eval(data_new, selected_features_rfe_new))
#print(random_forest_eval(data, selected_features_rfe))

#random_forest_eval(data, selected_features_rfe).to_excel('random_forest_eval_results.xlsx', index=False)
#random_forest_eval(data_percentage, selected_features_rfe_percentage).to_excel('random_forest_eval_percentage_results.xlsx', index=False)