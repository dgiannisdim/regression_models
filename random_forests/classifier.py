import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss

pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


features = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del features['Unnamed: 0']
features = features.dropna()
features = features.reset_index(drop=True)
features = pd.get_dummies(features)


print(features.head())

labels = np.array(features['electricity_laundry'])
features = features.iloc[:, 23 : 60]
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)



rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)



print(r2_score(test_labels, predictions))

