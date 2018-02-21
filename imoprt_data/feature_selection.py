import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import sklearn



data = pd.read_csv(r'C:\regression models\imoprt_data/final.csv')
del data['Unnamed: 0']
data = data.dropna()


X = data.iloc[:, [30,31]]
Y = data.iloc[:, 2]

lr = LinearRegression()
lr.fit(X, Y)




print(sklearn.metrics.r2_score)