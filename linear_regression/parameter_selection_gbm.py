import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection
from time import time



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




def gbm_parameters_loss(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    loss = ['ls', 'lad', 'huber', 'quantile']

    r2_column = []
    smape_column = []
    rmse_column = []

    for l in loss:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss=l)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'loss': loss, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['loss', 'R2', 'SMAPE', 'RMSE'])

    return df


#print(gbm_parameters_loss(data, selected_features_rfe))
#gbm_parameters_loss(data, selected_features_rfe).to_excel('gbm_parameters_loss.xlsx', index=False)




def gbm_parameters_learning_rate(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    learning_rate = [0.01, 0.1, 1, 2]
    learning_rate2 = np.arange(0.05, 0.51, 0.05)

    r2_column = []
    smape_column = []
    rmse_column = []

    for l in learning_rate2:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=l)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'learning_rate': learning_rate2, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['learning_rate', 'R2', 'SMAPE', 'RMSE'])

    return df


#print(gbm_parameters_learning_rate(data, selected_features_rfe))
#gbm_parameters_learning_rate(data, selected_features_rfe).to_excel('gbm_parameters_learning_rate.xlsx', index=False)
#gbm_parameters_learning_rate(data, selected_features_rfe).to_excel('gbm_parameters_learning_rate_detail.xlsx', index=False)




def gbm_parameters_n_estimators(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    n_estimators = 10 ** np.arange(0, 6)
    n_estimators2 = np.arange(100, 1001, 100)

    r2_column = []
    smape_column = []
    rmse_column = []
    time_column = []

    for n in n_estimators2:
        t0 = time()
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=n)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)
        time_column.append(time() - t0)

    df = pd.DataFrame({'n_estimators': n_estimators2, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,
                       'Time(s)': time_column},
                      columns=['n_estimators', 'R2', 'SMAPE', 'RMSE', 'Time(s)'])

    return df


#print(gbm_parameters_n_estimators(data, selected_features_rfe))
#gbm_parameters_n_estimators(data, selected_features_rfe).to_excel('gbm_parameters_n_estimators.xlsx', index=False)
#gbm_parameters_n_estimators(data, selected_features_rfe).to_excel('gbm_parameters_n_estimators_detail.xlsx', index=False)



def gbm_parameters_max_depth(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    max_depth = np.arange(1, 10)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_depth:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=m)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'max_depth': max_depth, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['max_depth', 'R2', 'SMAPE', 'RMSE'])

    return df


#print(gbm_parameters_max_depth(data, selected_features_rfe))
#gbm_parameters_max_depth(data, selected_features_rfe).to_excel('gbm_parameters_max_depth.xlsx', index=False)




def gbm_parameters_criterion(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    criterion = ['friedman_mse', 'mse', 'mae']

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in criterion:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion=c)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'criterion': criterion, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['criterion', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(gbm_parameters_criterion(data, selected_features_rfe))
#gbm_parameters_criterion(data, selected_features_rfe).to_excel('gbm_parameters_criterion.xlsx', index=False)



def gbm_parameters_min_samples_split(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_samples_split:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=m)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'min_samples_split': min_samples_split, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['min_samples_split', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(gbm_parameters_min_samples_split(data, selected_features_rfe))
#gbm_parameters_min_samples_split(data, selected_features_rfe).to_excel('gbm_parameters_min_samples_split.xlsx', index=False)



def gbm_parameters_min_samples_leaf(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    min_samples_leaf = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_samples_leaf:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=m)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'min_samples_leaf': min_samples_leaf, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['min_samples_leaf', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(gbm_parameters_min_samples_leaf(data, selected_features_rfe))
#gbm_parameters_min_samples_leaf(data, selected_features_rfe).to_excel('gbm_parameters_min_samples_leaf.xlsx', index=False)




def gbm_parameters_min_weight_fraction_leaf(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    min_weight_fraction_leaf = np.arange(0,0.6, 0.1)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in min_weight_fraction_leaf:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=m)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'min_weight_fraction_leaf': min_weight_fraction_leaf, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['min_weight_fraction_leaf', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(gbm_parameters_min_weight_fraction_leaf(data, selected_features_rfe))
#gbm_parameters_min_weight_fraction_leaf(data, selected_features_rfe).to_excel('gbm_parameters_min_weight_fraction_leaf.xlsx', index=False)




def gbm_parameters_subsample(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    subsample = np.arange(0.1, 1.1, 0.1)

    r2_column = []
    smape_column = []
    rmse_column = []

    for s in subsample:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0,
                                                 subsample=s)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'subsample': subsample, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['subsample', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(gbm_parameters_subsample(data, selected_features_rfe))
#gbm_parameters_subsample(data, selected_features_rfe).to_excel('gbm_parameters_subsample.xlsx', index=False)



def gbm_parameters_max_features(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    max_features = ['auto', 'sqrt', 'log2', None]
    max_features2 = np.arange(5, 30, 5)

    r2_column = []
    smape_column = []
    rmse_column = []

    for m in max_features2:
        X = data.loc[:, consumption_selected]
        #X = preprocessing.StandardScaler().fit_transform(X)
        #X = sm.add_constant(X)
        y = data['electricity_total']


        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                        random_state=0, test_size=0.25)

        clf = ensemble.GradientBoostingRegressor(random_state=0, loss='lad',
                                                 learning_rate=0.35,
                                                 n_estimators=400, max_depth=3,
                                                 criterion='friedman_mse',
                                                 min_samples_split=40,
                                                 min_samples_leaf=1,
                                                 min_weight_fraction_leaf=0,
                                                 subsample=1,
                                                 max_features=m)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)


        r2 = clf.score(X_test, y_test)
        smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)

    df = pd.DataFrame({'max_features': max_features2, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column,},
                      columns=['max_features', 'R2', 'SMAPE', 'RMSE'])

    return df

#print(gbm_parameters_max_features(data, selected_features_rfe))
#gbm_parameters_max_features(data, selected_features_rfe).to_excel('gbm_parameters_max_features_int.xlsx', index=False)

