import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection
import pickle
from sklearn.externals import joblib
import  os


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
#selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage.csv')


data_new.rename(columns={'dvd_or_bluray': 'dvd_or_blueray'}, inplace=True)
data_percentage_new.rename(columns={'dvd_or_bluray': 'dvd_or_blueray'}, inplace=True)



#predict total electreicity raw values and use percentage models for the rest
def predictor(data, data_percentage, selected_features, selected_features_percentage):
    data = pd.get_dummies(data, drop_first=True)
    data_percentage = pd.get_dummies(data_percentage, drop_first=True)

    consumption_selected = list(selected_features.loc[:, 'electricity_total'])


    # load the electricity_total model from disk
    path = r'C:\regression_models\random_forests\models'
    model_name = path + '/electricity_total_model.pkl'
    model_pkl = open(model_name, 'rb')
    pickle_electricity_total_model = pickle.load(model_pkl)


    #prediction examples
    row = 128
    features_electricity_total = [data[f][row] for f in consumption_selected]
    features_electricity_total = np.reshape(features_electricity_total, (1, -1))
    electricity_total_prediction = pickle_electricity_total_model.predict(features_electricity_total)
    print('electricity_real: ', data.loc[row, 'electricity_total'])
    print('electricity_prediction: ', electricity_total_prediction)
    print('\n')


    percentage_column = []
    raw_column = []
    percentage_pred_column = []
    raw_pred_column = []

    for cf in selected_features_percentage.columns:
        consumption_selected_percentage = list(selected_features_percentage.loc[:, cf])


        X = data_percentage.loc[:, consumption_selected_percentage]
        y = data_percentage[cf]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        # load the electricity_total model from disk
        path = r'C:\regression_models\random_forests\models'
        model_name = path + '/' + cf + '_percentage_model.pkl'
        model_pkl = open(model_name, 'rb')
        pickle_model = pickle.load(model_pkl)


        #create columns with real and prediction values prediction examples
        features = [data_percentage[f][row] for f in consumption_selected_percentage]
        features = np.reshape(features, (1, -1))
        prediction =  pickle_model.predict(features)

        percentage_column.append(data_percentage.loc[row, cf])
        percentage_pred_column.append(prediction)
        raw_column.append(data.loc[row, cf])
        raw_pred_column.append(prediction*0.01*electricity_total_prediction)

    #create dataframe with real and prediction values
    df = pd.DataFrame({'consumption' : selected_features.columns,
                       'percentage' : percentage_column,
                       'percentage_pred' : percentage_pred_column,
                       'raw' : raw_column,
                       'raw_pred' : raw_pred_column},
                      columns= ['consumption', 'percentage', 'percentage_pred', 'raw', 'raw_pred'])

    return df


print(predictor(data_new, data_percentage_new, selected_features_rfe, selected_features_rfe))

#predictor(data, data_percentage, selected_features_rfe, selected_features_rfe_percentage).to_excel('test.xlsx', index=False)