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
data_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged_percentage_new.csv')


#sort drop NaN values
def sort_and_drop(data):
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data


data = sort_and_drop(data)
data_new = sort_and_drop(data_new)
data_percentage = sort_and_drop(data_percentage)
data_percentage_new = sort_and_drop(data_percentage_new)

data = pd.get_dummies(data, drop_first=True)
data_percentage = pd.get_dummies(data_percentage, drop_first=True)
data_new = pd.get_dummies(data_new, drop_first=True)
data_percentage_new = pd.get_dummies(data_percentage_new, drop_first=True)



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
        #selected_features_random_forest[f] = importance_df(X, y)

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




    return selected_features_rfe


#print(select_features(data))

## write selected features to csv file
#old dataset
#select_features(data).to_csv('selected_features_rfe.csv', index=False)
#select_features(data_percentage).to_csv('selected_features_rfe_percentage.csv', index=False)



#select features with different methods and write to csv
def select_features_new(data):


    features = ['electricity_space_heating', 'electricity_water_heating',
       'electricity_electric_vehicle', 'electricity_pool_or_sauna']

    #create dataframe with most importand features for every consumption based on rfe methods
    selected_features_rfe = pd.DataFrame(np.nan, index=range(0, 30), columns=features)

    for f in features:
        # split dataset into independent and dependent variables
        X = data.loc[:, 'property_size':]
        y = data[f]
        y = y.astype('int')


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

        print('#################################################')
        print('#################################################')
        print('#################################################')
        print('\n')
        print('\n')




    return selected_features_rfe


#new dataset
#select_features_new(data_new).to_csv('selected_features_rfe_new2.csv', index=False)
#select_features_new(data_percentage_new).to_csv('selected_features_rfe_percentage_new2.csv', index=False)
#df.to_excel('selected_features_percentage.xlsx', index=False)


#pca
X = data.loc[:, 'property_size':]
y = data['electricity_total']
y = y.astype('int')

pca = PCA(n_components=20)
pca.fit(X)
print(pca.explained_variance_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()