import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm



pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)


data = pd.read_csv(r'C:\regression_models\linear_regression/final_values.csv')
del data['Unnamed: 0']
data = data.dropna()
data = data.reset_index(drop=True)





data = pd.get_dummies(data)


#create model with OLS method of statsmodels
f = list(data.columns.values)
f = f[77 : 85]
X = data[f]
X = sm.add_constant(X)
Y = data['electricity_cooking']

model = sm.OLS(Y,X)
res = model.fit()
print(res.summary())




'''
#create model with formula method of statsmodels (flexible)
Y = 'electricity_cooking ~ '
X = ''
data = list(data.columns.values)
del data[0:25]
for f in data:
    X = X + f + ' + '

formula = Y + X
formula = formula[:-3]
formula = "'" + formula + "'"


mod = smf.ols(formula=formula, data=data)
res = mod.fit()
print(res.summary())





#create model with formula method of statsmodels
mod = smf.ols(formula='electricity_cooking ~ C(oven) + C(microwave) + C(occupants) + C(occupant_type)'
              , data=data)
res = mod.fit()

print(res.summary())


#create model with sklearn
#data = pd.get_dummies(data)

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
#print(sklearn.metrics.r2_score)

'''


#print(model.predict([[0, 1, 1]]))

