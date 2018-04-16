import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection




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

        r_squared_column = []
        r_squared_adjusted_column = []
        smape_column = []
        explained_variance_column = []
        root_mean_square_error_column = []

        for cf in selected_features.columns:
            consumption_selected = list(selected_features.loc[:, cf])

            consumption_selected = consumption_selected[: 25]

            X = data.loc[:, consumption_selected]
            # X = preprocessing.StandardScaler().fit_transform(X)
            # X = sm.add_constant(X)
            y = data[cf]

            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

            clf = linear_model.LinearRegression()
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            r2 = clf.score(X_train, y_train)
            adj_r2 = 1 - (1-clf.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)


            r_squared_column.append(r2)
            r_squared_adjusted_column.append(adj_r2)
            explained_variance_column.append(explained_variance_score(y_test, predictions))
            root_mean_square_error_column.append(np.sqrt(mean_squared_error(y_test, predictions)))
            smape_column.append(np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions))))



        df = pd.DataFrame({'consumption': selected_features.columns, 'r_squared': r_squared_column,
                           'adjusted_r_squared': r_squared_adjusted_column,
                           'smape': smape_column,
                           'root_mean_squared_error': root_mean_square_error_column,
                           'explained_variance_score': explained_variance_column},
                          columns=['consumption', 'r_squared', 'adjusted_r_squared', 'smape',
                                   'root_mean_squared_error', 'explained_variance_score'])

        return df

#print(sklearn_lr_eval(data, selected_features_rfe_percentage))



def sklearn_SVR_eval_params(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    r_squared_rbf_column = []
    r_squared_linear_column = []
    r_squared_poly_column = []


    for cf in selected_features.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        X = data.loc[:, consumption_selected]
        #X = sm.tools.add_constant(X)
        y = data[cf]

        def hyperparameters():
            parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
                           'gamma': [1e-2, 1e-1, 1, 1e1]},
                          {'kernel': ['poly'], 'C': [1, 10, 100, 1000],
                           'degree': [2, 3, 4]},
                          {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                          ]
            clf = model_selection.GridSearchCV(SVR(), parameters, cv=5)
            clf.fit(X, y)

            print(cf + ': ', clf.best_params_)

        if cf == 'electricity_total':
            hyperparameters()


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)


        clf_rbf = SVR(kernel='rbf', C=1000, gamma=0.01)
        clf_linear = SVR(kernel='linear', C=10)
        clf_poly = SVR(kernel='poly', C=100, degree=2)


        clf_rbf.fit(X_train, y_train)
        clf_linear.fit(X_train, y_train)
        clf_poly.fit(X_train, y_train)
        predictions = clf_rbf.fit(X_train, y_train).predict(X_test)


        #plot the model
        def plot_the_model():
            plt.scatter(y_test, predictions)
            plt.title(cf)
            plt.xlabel('True values')
            plt.ylabel('Predictions')
            plt.savefig('example ' + cf + '.png')
            plt.close()


        #create columns to compare r2 of different kernels
        r_squared_rbf_column.append(clf_rbf.score(X_train, y_train))
        r_squared_linear_column.append(clf_linear.score(X_train, y_train))
        r_squared_poly_column.append(clf_poly.score(X_train, y_train))


    df = pd.DataFrame({'consumption': selected_features.columns,
                       'r_squared_rbf': r_squared_rbf_column,
                       'r_squared_linear': r_squared_linear_column,
                       'r_squared_poly': r_squared_poly_column},
                      columns=['consumption', 'r_squared_rbf', 'r_squared_linear', 'r_squared_poly'])

    return df


#print(sklearn_SVR_eval_params(data, selected_features_rfe_percentage))


def sklearn_SVR_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    r_squared_column = []
    r_squared_adjusted_column = []
    smape_column = []
    explained_variance_column = []
    root_mean_square_error_column = []


    for cf in selected_features.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        X = data.loc[:, consumption_selected]
        #X = sm.tools.add_constant(X)
        y = data[cf]


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)


        clf = SVR(kernel='rbf', C=1000, gamma=0.01)

        clf.fit(X_train, y_train)
        predictions = clf.fit(X_train, y_train).predict(X_test)


        #calculate adjusted r2
        rsquared_adj = 1 - (1 - clf.score(X_train, y_train)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)

        #calculate sMAPE
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))


        #create columns to make dataframe with metrics for evaluation
        r_squared_column.append(clf.score(X_train, y_train))
        r_squared_adjusted_column.append(rsquared_adj)
        explained_variance_column.append(explained_variance_score(y_test, predictions))
        root_mean_square_error_column.append(np.sqrt(mean_squared_error(y_test, predictions)))
        smape_column.append(np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions))))


    df = pd.DataFrame({'consumption': selected_features.columns,
                       'r_squared': r_squared_column,
                       'adjusted_r_squared': r_squared_adjusted_column,
                       'smape': smape_column,
                       'root_mean_squared_error': root_mean_square_error_column,
                       'explained_variance_score': explained_variance_column},
                      columns=['consumption', 'r_squared', 'adjusted_r_squared', 'smape',
                               'root_mean_squared_error', 'explained_variance_score'])

    return df


#print(sklearn_SVR_eval(data, selected_features_rfe_percentage))

#create model with OLS method of statsmodels
def stats_lr_eval(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    r_squared_column = []
    r_squared_adjusted_column = []
    smape_column = []
    explained_variance_column = []
    root_mean_square_error_column = []

    for cf in selected_features.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        consumption_selected = consumption_selected[: 25]

        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data[cf]



        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        clf = sm.OLS(y,X)
        res = clf.fit()
        predictions = res.predict(X_test)

        row = 52
        #prediction examples
        #b = [data[f][row] for f in consumption_selected]
        #print(res.predict(b) , data.loc[row, cf])



        r_squared_column.append(res.rsquared)
        r_squared_adjusted_column.append(res.rsquared_adj)
        explained_variance_column.append(explained_variance_score(y_test, predictions))
        root_mean_square_error_column.append(np.sqrt(mean_squared_error(y_test, predictions)))
        smape_column.append(np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions))))

        #print(res.summary())
        #print(res.params)



    df = pd.DataFrame({'consumption' : selected_features.columns, 'r_squared' : r_squared_column,
                       'adjusted_r_squared' : r_squared_adjusted_column,
                       'smape' : smape_column,
                       'root_mean_squared_error' : root_mean_square_error_column,
                       'explained_variance_score' : explained_variance_column},
                      columns=['consumption', 'r_squared', 'adjusted_r_squared', 'smape',
                               'root_mean_squared_error', 'explained_variance_score'])

    return df

#print(stats_lr_eval(data, selected_features_rfe))
#stats_lr_eval(data, selected_features_rfe_percentage)



def gradient_boosting(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    r_squared_column = []
    r_squared_adjusted_column = []
    smape_column = []
    explained_variance_column = []
    root_mean_square_error_column = []

    for cf in selected_features.columns:
        consumption_selected = list(selected_features.loc[:, cf])

        consumption_selected = consumption_selected[: 25]

        X = data.loc[:, consumption_selected]
        # X = preprocessing.StandardScaler().fit_transform(X)
        # X = sm.add_constant(X)
        y = data[cf]

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)

        res = clf.fit(X_train, y_train)
        predictions = res.predict(X_test)

        rsquared_adj = 1 - (1 - r2_score(y_test, predictions)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1)
        r_squared_column.append(r2_score(y_test, predictions))
        r_squared_adjusted_column.append(rsquared_adj)
        explained_variance_column.append(explained_variance_score(y_test, predictions))
        root_mean_square_error_column.append(np.sqrt(mean_squared_error(y_test, predictions)))
        smape_column.append(np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions))))



    df = pd.DataFrame({'consumption': selected_features.columns, 'r_squared': r_squared_column,
                       'adjusted_r_squared': r_squared_adjusted_column,
                       'smape': smape_column,
                       'root_mean_squared_error': root_mean_square_error_column,
                       'explained_variance_score': explained_variance_column},
                      columns=['consumption', 'r_squared', 'adjusted_r_squared', 'smape',
                               'root_mean_squared_error', 'explained_variance_score'])

    print(df)


#gradient_boosting(data, selected_features_rfe_percentage)

#write model results to excel files
'''

sklearn_lr_eval(data, selected_features_rfe).to_excel('sklearn_lr_results.xlsx', index=False)
sklearn_lr_eval(data_percentage, selected_features_rfe_percentage).to_excel('sklearn_lr_percentage_results.xlsx', index=False)

stats_lr_eval(data, selected_features_rfe).to_excel('stats_lr_results.xlsx', index=False)
stats_lr_eval(data_percentage, selected_features_rfe_percentage).to_excel('stats_lr_percentage_results.xlsx', index=False)

gradient_boosting(data, selected_features_rfe).to_excel('gradient_boosting_results.xlsx', index=False)
gradient_boosting(data_percentage, selected_features_rfe_percentage).to_excel('gradient_boosting_percentage_results.xlsx', index=False)

sklearn_SVR_eval(data, selected_features_rfe).to_excel('sklearn_SVR_eval_results.xlsx', index=False)
sklearn_SVR_eval(data_percentage, selected_features_rfe_percentage).to_excel('sklearn_SVR_eval_percentage_results.xlsx', index=False)


'''




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

