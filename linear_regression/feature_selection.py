import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectPercentile, SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import sklearn.feature_selection
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

#data percentage
data = pd.read_csv(r'C:\regression_models\linear_regression/final.csv')
del data['Unnamed: 0']
data = data.dropna()


#data raw values
data = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data.reset_index(drop=True)
data = pd.get_dummies(data)



X = data.iloc[: , 14:]

y = data['electricity_total']
y=y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)


#feature selection
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)




#most important features
select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]




'''

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
'''


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

importance_plot(X, X_train, y_train)