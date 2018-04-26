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
import pylab



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



#create dataframe with metrics
def metrics_dataframe():
    pass

def stats_lr(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)

    r_squared_column = []
    r_squared_adjusted_column = []
    smape_column = []
    explained_variance_column = []
    root_mean_square_error_column = []


    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    X = data.loc[:, consumption_selected]
    # X = preprocessing.StandardScaler().fit_transform(X)
    # X = sm.add_constant(X)
    y = data['electricity_total']

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    clf = sm.OLS(y, X)
    res = clf.fit()
    predictions = res.predict(X_test)


    r_squared = res.rsquared
    r_squared_adjusted = res.rsquared_adj
    explained_variance = explained_variance_score(y_test, predictions)
    root_mean_square_error = np.sqrt(mean_squared_error(y_test, predictions))
    smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))


    df = pd.DataFrame({'consumption': 'stats_lr',
                       'r_squared': [r_squared],
                       'adjusted_r_squared': [r_squared_adjusted],
                       'smape': [smape],
                       'root_mean_squared_error': [root_mean_square_error],
                       'explained_variance_score': [explained_variance]},
                      columns=['consumption', 'r_squared', 'adjusted_r_squared', 'smape',
                               'root_mean_squared_error', 'explained_variance_score'])

    return df



#print(stats_lr(data, selected_features_rfe))
#stats_lr(data, selected_features_rfe).to_excel('test.xlsx', index=False)


def SVR_parameters(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    X = data.loc[:, consumption_selected]
    # X = preprocessing.StandardScaler().fit_transform(X)
    # X = sm.add_constant(X)
    y = data['electricity_total']

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    C_range = 10. ** np.arange(-3, 5)
    gamma_range = 10. ** np.arange(-4,4)
    degree_range = range(2, 5)
    kernel = ['rbf']
    coef = [0, 1]
    epsilon = [0, 0.01, 0.1, 0.5, 1, 2, 4]
    param_grid = dict(kernel=kernel, C=C_range, coef0=coef, epsilon=epsilon,
                      gamma=gamma_range, degree=degree_range)

    plt.scatter(X_train['occupants'], y_train)
    plt.show
    grid = model_selection.GridSearchCV(SVR(), param_grid=param_grid, cv=5)
    grid.fit(X, y)


    df = pd.DataFrame(grid.grid_scores_)
    best_par = pd.Series(grid.best_params_, name='best_parameters')
    df = df.append(best_par)
    print("The best classifier is: ", grid.best_params_)


    #create paramaeters plotand save
    def plot_paramaeters():
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(degree_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig('svr_parameters_rbf.png')
        plt.close()




    df.to_excel('svr_parameters_rbf2.xlsx', index=False)
    #plot_paramaeters()




SVR_parameters(data, selected_features_rfe)



def SVR_compare(data, selected_features):
    data = pd.get_dummies(data, drop_first=True)
    consumption_selected = list(selected_features.loc[:, 'electricity_total'])

    X = data.loc[:, consumption_selected]
    # X = preprocessing.StandardScaler().fit_transform(X)
    # X = sm.add_constant(X)
    y = data['electricity_total']

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)


    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 5],
                  'gamma': [1e-4, 1e4], 'epsilon': [0, 0.01, 0.1, 0.5, 1, 2, 4]}

    plt.scatter(X_train['occupants'], y_train)
    plt.show
    clf = model_selection.GridSearchCV(SVR(), parameters, cv=5)
    clf.fit(X, y)


    df = pd.DataFrame(grid.grid_scores_)
    best_par = pd.Series(grid.best_params_, name='best_parameters')
    df = df.append(best_par)
    print("The best classifier is: ", grid.best_params_)


    #create paramaeters plotand save
    def plot_paramaeters():
        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                             len(degree_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('degree')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(degree_range)), degree_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.savefig('svr_parameters_poly.png')
        plt.close()




    df.to_excel('svr_parameters_compare.xlsx', index=False)
    #plot_paramaeters()


#SVR_compare(data, selected_features_rfe)
