import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf




pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression models\imoprt_data/final.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data.sort_values(by=['household_id', 'month'])
data = data.reset_index(drop=True)



'''
X = data[f]
X = sm.add_constant(X)
Y = data['electricity_refrigeration']

model = sm.OLS(Y,X)
res = model.fit()
print(res.summary())
'''
Y = 'electricity_cooking ~ '
X = ''
features = list(data.columns.values)
del features[0:25]
for f in features:
    X = X + f + ' + '

formula = Y + X
formula = formula[:-3]
formula = "'" + formula + "'"


mod = smf.ols(formula=formula, data=data)
res = mod.fit()
print(res.summary())


'''
mod = smf.ols(formula='electricity_cooking ~   oven + microwave + property_size'
                       '+ occupants + occupant_type'
              , data=data)
res = mod.fit()

#data = pd.get_dummies(data)

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
#print(sklearn.metrics.r2_score)
'''

#print(model.predict([[0, 1, 1]]))

