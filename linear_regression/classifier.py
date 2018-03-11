import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
selected_features = pd.read_csv(r'C:\regression_models\linear_regression/selected_features2.csv')



#create model with sklearn
def sklearn_lr(data):
    data = pd.get_dummies(data, drop_first=True)
    consumption_features = list(data.columns)
    consumption_features = consumption_features[: 11]
    for cf in consumption_features:
        consumption_selected = list(selected_features.loc[:, cf])


        X = data.loc[:, consumption_selected]
        y = data[cf]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        clf = linear_model.LinearRegression()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        print(cf, 'r2: ', r2_score(y_test, predictions))





#create model with OLS method of statsmodels
def stats_lr(data):
    data = pd.get_dummies(data, drop_first=True)
    consumption_features = list(data.columns)
    consumption_features = consumption_features[: 11]
    for cf in consumption_features:
        consumption_selected = list(selected_features.loc[:, cf])

        consumption_selected = consumption_selected[:20]
        X = data.loc[:, consumption_selected]
        y = data[cf]

        model = sm.OLS(y,X)
        res = model.fit()
        print(cf, 'r2: ', res.rsquared)
        #print(cf, 'r2_adjusted: ', res.rsquared_adj)








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

