import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

# Fit regression models and calculate r2, smape and rmse
def calculate_metrics(clf, X_train, X_test, y_train, y_test):
    # Fit the model with train dataset
    clf.fit(X_train, y_train)

    # Predict consumption for the test dataset
    predictions = clf.predict(X_test)

    # Calculate r2
    SS_Residual = sum((y_test - predictions) ** 2)
    SS_Total = sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (float(SS_Residual)) / SS_Total

    # Calculate smape
    smape = np.mean(200 * abs(y_test - predictions) / (abs(y_test) + abs(predictions)))

    # Calculate rmse
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return r2, smape, rmse


# save the model
def save_model():
    model_name = consumption + '_model.pkl'
    model_pkl = open(model_name, 'wb')
    pickle.dump(res, model_pkl)