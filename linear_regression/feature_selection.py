import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.linear_model import LinearRegression, RandomizedLasso
from sklearn.feature_selection import SelectPercentile, SelectFromModel, SelectKBest, \
    f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier



#ignore some errors
np.seterr(divide='ignore', invalid='ignore')

#display settings in pandas
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


#read csv files
data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage.csv')
data = pd.get_dummies(data, drop_first=True)
data_percentage = pd.get_dummies(data, drop_first=True)



# importance of each feature
def importance_plot(X, y, title):
    forest = RandomForestRegressor(n_estimators=1000, random_state=0)
    forest.fit(X, y)
    features = X.columns

    imporatnces = forest.feature_importances_
    indices = np.argsort(imporatnces)

    plt.barh(range(len(indices)), imporatnces[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.title(title)
    plt.show()



#feature selection with random forest
def importance_df(X, y):
    forest = RandomForestRegressor(n_estimators=1000, random_state=0)
    forest.fit(X, y)
    features = X.columns

    imporatnces = forest.feature_importances_
    indices = np.argsort(imporatnces)

    all_features = pd.DataFrame({'features': features, 'importances': imporatnces})
    all_features = all_features.sort_values(by='importances', ascending=False)
    all_features = all_features.reset_index(drop=True)
    top_features = all_features.features[:30]

    return  top_features



##Univariate Feature Selection
#Select K-Best
def kbest(X, y):
    model = SelectKBest(k=30)
    selector = model.fit(X, y)
    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected


#can not run code if f_regression is on
def f_regression(X, y):

    model = SelectKBest(f_regression(X, y), k=30)
    selector = model.fit(X, y)

    print(selector.get_support(indices=True))



#Stability selection
def lasso(X, y):
    rlasso = RandomizedLasso(alpha=0.025)
    selector = rlasso.fit(X, y)

    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected


def select_from_model_extra_trees(X, y):
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)

    model = SelectFromModel(clf, threshold=0.001)
    selector = model.fit(X, y)

    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected



#RFE with SVR estimator
def rfe_selector_svr(X, y):
    estimator = SVR(kernel='linear')
    selector = RFE(estimator, 30, step=1)
    selector = selector.fit(X, y)
    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected



#RFE with Linear Regression estimator
def rfe_selector_lr(X, y):
    estimator = LinearRegression()
    selector = RFE(estimator, 30, step=1)
    selector = selector.fit(X, y)
    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected




#RFE with Random Forest estimator
def rfe_selector_random_forest(X, y):
    estimator = RandomForestRegressor()
    selector = RFE(estimator, 30, step=1)
    selector = selector.fit(X, y)
    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected





#select features with different methods and write to csv
def select_features(data):
    features = data.columns
    features = features[: 10]

    #create dataframe with most importand features for every consumption based only on random forests
    selected_features_random_forest = pd.DataFrame(np.nan, index=range(0, 30), columns=features)

    #create dataframe with most importand features for every consumption based on rfe methods
    selected_features_rfe = pd.DataFrame(np.nan, index=range(0, 30), columns=features)

    for f in features:
        # split dataset into independent and dependent variables
        X = data.loc[:, 'property_size':]
        y = data[f]
        y = y.astype('int')

        #random forest dataframe
        selected_features_random_forest[f] = importance_df(X, y)

        #importance_plot(X, y, f)


        ## rfe dataframe
        # how many times a feature was chosen
        list_of_all_indices = np.concatenate((kbest(X, y), rfe_selector_svr(X, y),
                                              rfe_selector_lr(X, y),
                                              rfe_selector_random_forest(X, y),
                                              lasso(X, y),
                                              select_from_model_extra_trees(X, y)),
                                              axis=0)
        list_of_all_indices = pd.Series(list_of_all_indices)


        importand_features = list_of_all_indices.value_counts().index.tolist()


        # create dataframe with most importand features of each consumption variable
        selected_features_rfe[f] = importand_features[0:30]



    selected_features_random_forest.to_csv('selected_features_random_forest_percentage.csv', index=False)


    selected_features_rfe.to_csv('selected_features_rfe_percentage.csv', index=False)

    return selected_features_rfe


select_features(data_percentage)
#df.to_excel('selected_features_percentage.xlsx', index=False)


