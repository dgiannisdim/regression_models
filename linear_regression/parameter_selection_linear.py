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
from statistics import mean
from matplotlib import style



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


# Create function to evaluate 'loss' parameter
def linear_parameters_statsmodels(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])


    X = data.loc[:, consumption_selected]
    # X = preprocessing.StandardScaler().fit_transform(X)
    X = sm.add_constant(X)
    y = data['electricity_total']


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                    random_state=0, test_size=0.25)

    clf = linear_model.SGDRegressor()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    SS_Residual = sum((y_test - predictions) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (float(SS_Residual)) / SS_Total

    smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(r2, clf.score(X_test, y_test))
    print(smape)
    print(rmse)



## Print final dataframe
#print(linear_parameters_statsmodels(data, selected_features_rfe))

## Write final dataframe to excel file
#linear_parameters_statsmodels(data, selected_features_rfe).to_excel('linear_parameters_statsmodels.xlsx', index=False)





def linear_parameters_sklearn(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])


    X = data.loc[:, consumption_selected]
    # X = preprocessing.StandardScaler().fit_transform(X)
    X = sm.add_constant(X)
    y = data['electricity_total']


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                    random_state=0, test_size=0.25)

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    SS_Residual = sum((y_test - predictions) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (float(SS_Residual)) / SS_Total

    smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(r2, clf.score(X_test, y_test))
    print(smape)
    print(rmse)

    xs = X_test.iloc[:, 25]
    ys = y_test

    def best_fit_slope_and_intercept(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
             ((mean(xs) * mean(xs)) - mean(xs * xs)))

        b = mean(ys) - m * mean(xs)

        return m, b

    m, b = best_fit_slope_and_intercept(xs, ys)

    regression_line = [(m * x) + b for x in xs]
    style.use('ggplot')

    plt.scatter(xs, ys, color='#003F72')
    plt.plot(xs, regression_line)
    plt.show()



#print(linear_parameters_sklearn(data, selected_features_rfe))
#linear_parameters_statsmodels(data, selected_features_rfe).to_excel('linear_parameters_statsmodels.xlsx', index=False)




