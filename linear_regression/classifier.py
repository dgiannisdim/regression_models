import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.svm import SVR
import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe.csv')
selected_features_random_forest = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_rfe_percentage.csv')
selected_features_random_forest_percentage = pd.read_csv(r'C:\regression_models\linear_regression/selected_features_random_forest_percentage.csv')



#create model with sklearn
def sklearn_lr_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    for cf in selected_features_random_forest.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        X = data.loc[:, consumption_selected]
        X = sm.tools.add_constant(X)
        y = data[cf]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        clf = linear_model.LinearRegression()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        #print(cf, 'r2: ', r2_score(y_test, predictions))
        print(cf, 'r2: ', clf.score(X_train, y_train))


def sklearn_SVR_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    for cf in selected_features.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        X = data.loc[:, consumption_selected]
        #X = sm.tools.add_constant(X)
        y = data[cf]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        clf_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        clf_linear = SVR(kernel='linear', C=1e3)
        clf_poly = SVR(kernel='poly', C=1e3, degree=2)

        predictions_rbf = clf_rbf.fit(X_train, y_train).predict(X_train)
        predictions_linear = clf_linear.fit(X_train, y_train).predict(X_train)
        predictions_poly = clf_poly.fit(X_train, y_train).predict(X_train)
        res = clf_rbf.fit(X_train, y_train)

        print(cf, 'rbf score: ', clf_rbf.score(X_test, y_test))
        print(cf, 'linear score: ', clf_linear.score(X_test, y_test))
        print(cf, 'poly score: ', clf_poly.score(X_test, y_test))

        row = 270
        # prediction examples
        b = [data[f][row] for f in consumption_selected]
        #print('\n', res.predict(b), data.loc[row, cf])




#create model with OLS method of statsmodels
def stats_lr_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    for cf in selected_features.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        consumption_selected = consumption_selected[: 35]

        X = data.loc[:, consumption_selected]
        #X = sm.add_constant(X)
        y = data[cf]

        model = sm.OLS(y,X)
        res = model.fit()

        row = 270
        #prediction examples
        b = [data[f][row] for f in consumption_selected]
        print(res.predict(b) , data.loc[row, cf])

        print(cf, 'r2: ', res.rsquared_adj)
        #print(res.summary())
        #print(res.params)
        #print(cf, 'r2_adjusted: ', res.rsquared_adj)





def stats_lr_pred(data, data_percentage, selected_features, selected_features_percentage):
    data = pd.get_dummies(data, drop_first=True)
    data_percentage = pd.get_dummies(data_percentage, drop_first=True)

    consumption_selected = list(selected_features.loc[:, 'electricity_total'])
    consumption_selected = consumption_selected[: 25]

    X_electricity_total = data.loc[:, consumption_selected]
    #X = sm.add_constant(X)
    y_electricity_total = data['electricity_total']

    model_electricity_total = sm.OLS(y_electricity_total, X_electricity_total)
    res_electricity_total = model_electricity_total.fit()

    print('electricity_total_raw ', 'r2: ', res_electricity_total.rsquared_adj)


    row = 270
    b_electricity_total = [data[f][row] for f in consumption_selected]
    print(res_electricity_total.predict(b_electricity_total), data.loc[row, 'electricity_total'])
    print('\n')

    for cf in selected_features_percentage.columns:
        consumption_selected_percentage = list(selected_features_percentage.loc[:, cf])
        consumption_selected_percentage = consumption_selected_percentage[: 35]

        X = data_percentage.loc[:, consumption_selected_percentage]
        y = data_percentage[cf]

        model = sm.OLS(y,X)
        res = model.fit()

        #prediction examples
        b = [data_percentage[f][row] for f in consumption_selected_percentage]
        print(res.predict(b) , data_percentage.loc[row, cf])

        #turn percentage consumption to raw values
        print(cf, 'raw consumption predict: ', res.predict(b)*0.01*res_electricity_total.predict(b_electricity_total))
        print(cf, 'raw consumption: ', data.loc[row, cf])
        print(cf, 'r2: ', res.rsquared_adj)


#stats_lr_pred(data, data_percentage, selected_features_rfe, selected_features_rfe_percentage)



sklearn_SVR_eval(data, selected_features_rfe)


'''
#create model with formula method of statsmodels

mod = smf.ols(formula='electricity_cooking ~ oven + microwave + occupants + occupant_type'
              , data=data)
res = mod.fit()

print(res.summary())
'''


#create model with formula method of statsmodels (flexible)
def stats_formula_lr(data):
    y = 'electricity_total ~ '
    X = ''

    for f in data.columns[15:20 ]:
        X = X + f + ' + '

    X = X[:-3]
    formula = y + X
    formula = "'" + formula + "'"
    print(formula)

    mod = smf.ols(formula=formula, data=data)
    res = mod.fit()
    print(res.summary())





#print(model.predict([[0, 1, 1]]))

