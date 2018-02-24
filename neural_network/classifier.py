import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data.reset_index(drop=True)
data = pd.get_dummies(data)

X = data.iloc[: , 14:35]
y = data['electricity_cooking']
y=y.astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(80,80,80))
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
print(r2_score(y_test, predictions))


'''
columns = ['property_type_PT.06', 'property_size_PS.04', 'property_age_PA.04',
           'property_ownership_OWN.02', 'occupants_OCC.05', 'occupant_type_OT.06',
           'space_heating_SH.07', 'water_heating_WH.05', 'stove_heating_STH.04',
           'grill_heating_GH.04', 'oven_heating_OH.02', 'photo_voltaic_SOL.02']
data = data.drop(columns, axis=1)
print(data.head())
print(data.shape)

labels = np.array(data['electricity_laundry'])
data = data.iloc[:, 23 : 60]
feature_list = list(data.columns)
data = np.array(data)





rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)



print(r2_score(test_labels, predictions))

'''