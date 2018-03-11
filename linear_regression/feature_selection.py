import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, RandomizedLasso
from sklearn.feature_selection import SelectPercentile, SelectFromModel, SelectKBest, \
    f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


#ignore some errors
np.seterr(divide='ignore', invalid='ignore')

#display settings in pandas
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


#read csv files
data = pd.read_csv(r'C:\regression_models\linear_regression/final_dataset_merged.csv')
data= pd.get_dummies(data, drop_first=True)

#split the dataset into training and testing set
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)


##Univariate Feature Selection
#Select K-Best
def kbest(X, y):
    select = SelectKBest(k=30)
    selected_features = select.fit(X, y)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected



#Stability selection
def lasso(X, y):
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(X, y)

    print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),X.columns), reverse=True))




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

features = list(data.columns)
features = features[: 11]
selected_features = pd.DataFrame(np.nan, index=range(0, 30), columns=features)

for f in features:
    #split dataset into independent and dependent variables
    X = data.loc[: , 'property_size':]
    y = data[f]
    y=y.astype('int')
    print(rfe_selector_lr(X, y))

#RFE with Random Forest estimator
def rfe_selector_random_forest(X, y):
    estimator = RandomForestRegressor()
    selector = RFE(estimator, 30, step=1)
    selector = selector.fit(X, y)
    indices_selected = selector.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    return colnames_selected



#loop for feature selection for every type of consumption
features = list(data.columns)
features = features[: 11]
selected_features = pd.DataFrame(np.nan, index=range(0, 30), columns=features)

for f in features:
    #split dataset into independent and dependent variables
    X = data.loc[: , 'property_size':]
    y = data[f]
    y=y.astype('int')

    #how many times a feature was chosen
    list_of_all_indices = np.concatenate((kbest(X, y) , rfe_selector_svr(X, y),
                            rfe_selector_lr(X, y), rfe_selector_random_forest(X, y)), axis=0)
    list_of_all_indices = pd.Series(list_of_all_indices)

    importand_features = list_of_all_indices.value_counts().index.tolist()

    #create dataframe with most importand features of each consumption variable
    selected_features[f] = importand_features[0:30]


print(selected_features)
#selected_features.to_csv('selected_features2.csv', index=False)




'''
#feature selection


#linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print('Linear regression before feature selection: ', r2_score(y_test, predictions))



lm.fit(X_train_selected, y_train)
predictions_selected = lm.predict(X_test_selected)
print('Linear regression after feature selection: ',r2_score(y_test, predictions_selected))




#random forests
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train);

predictions = rf.predict(X_test)
print('Random forests before feature selection: ', r2_score(y_test, predictions))


rf.fit(X_train_selected, y_train)
predictions_selected = rf.predict(X_test_selected)
print('Random forests after feature selection: ',r2_score(y_test, predictions_selected))



#feature selection with Select From Model
select = SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=42), threshold='median')
select.fit(X_train, y_train)
X_train_s = select.transform(X_train)


X_test_s = select.transform(X_test)
score = LinearRegression().fit(X_train, y_train).score(X_test, y_test)
print(score)
score_s = LinearRegression().fit(X_train_s, y_train).score(X_test_s, y_test)
print(score_s)


#importance of each feature
def importance_plot(X, X_train, y_train):
    forest = RandomForestClassifier(n_estimators=1000, random_state=0)
    forest.fit(X_train, y_train)
    features = X.columns
    for name, importance in zip(features, forest.feature_importances_):
        print(name, "=", importance)

    imporatnces = forest.feature_importances_
    indices = np.argsort(imporatnces)

    plt.barh(range(len(indices)), imporatnces[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.show()

'''