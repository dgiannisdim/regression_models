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


# Import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')

# Import new datasets
selected_features_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_new.csv')
selected_features_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_percentage_new.csv')





#predict total electreicity raw values and use percentage models for the rest
def predictor(data, data_percentage, selected_features, selected_features_percentage):
    # One hot encoding
    data = pd.get_dummies(data, drop_first=True)
    data_percentage = pd.get_dummies(data_percentage, drop_first=True)

    # Set the subset for features
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])


    # load the electricity_total model from disk
    path = r'C:\regression_models\random_forests\models'
    model_name = path + '/electricity_total_model.pkl'
    model_pkl = open(model_name, 'rb')
    pickle_electricity_total_model = pickle.load(model_pkl)



    ## Example for a specific household
    # Predict the electricity_total consumption
    row = 386
    features_electricity_total = [data[f][row] for f in consumption_selected]
    features_electricity_total = np.reshape(features_electricity_total, (1, -1))
    electricity_total_prediction = pickle_electricity_total_model.predict(features_electricity_total)
    print('electricity_total_real: ', data.loc[row, 'electricity_total'])
    print('electricity_total_prediction: ', electricity_total_prediction)
    print('\n')


    # Create empty columns for the dataframe
    percentage_column = []
    raw_column = []
    percentage_pred_column = []
    raw_pred_column = []

    # Predict percentage and raw consumptions for every consumption category
    for cf in selected_features_percentage.columns:
        # Subset of most important features in each consumption category
        consumption_selected_percentage = list(selected_features_percentage.loc[:, cf])

        # load the electricity_total model from disk
        path = r'C:\regression_models\random_forests\models'
        model_name = path + '/' + cf + '_percentage_model.pkl'
        model_pkl = open(model_name, 'rb')
        pickle_model = pickle.load(model_pkl)



        # Create columns with real and prediction values
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


print(predictor(data, data_percentage, selected_features_rfe, selected_features_rfe_percentage))

#predictor(data, data_percentage, selected_features_rfe, selected_features_rfe_percentage).to_excel('test.xlsx', index=False)