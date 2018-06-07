import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, cross_validation, svm, ensemble
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from basic_functions import calculate_metrics


pd.set_option('display.max_columns', 99)
pd.set_option('display.max_row', 999)

# Import old datasets
data = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged.csv')
data_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage.csv')
selected_features_rfe = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe.csv')
selected_features_rfe_percentage = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/selected_features_rfe_percentage.csv')


# Import new datasets
data_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_new.csv')
data_percentage_new = pd.read_csv(r'C:\regression_models\linear_regression\merged_datasets/final_dataset_merged_percentage_new.csv')

#drop Nan values
data = data.dropna()
data_percentage = data_percentage.dropna()
data_new = data_new.dropna()
data_percentage_new = data_percentage_new.dropna()

# One hot encoding for categorical features
data = pd.get_dummies(data, drop_first=True)

# Create a list with most important features related with electricity_total
consumption_selected = list(selected_features_rfe.loc[:, 'electricity_total'])

# X is the dataframe of selected features for electricity_total
X = data.loc[:, consumption_selected]

# y is the electricity_total column
y = data['electricity_total']

# Split the dataset into training (75%) and testing (25%) set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
                                                                     random_state=0, test_size=0.25)

# Create dataframe to compare different values of parameters
def create_dataframe(param_name, param, r2_column, smape_column, rmse_column):
    df = pd.DataFrame({param_name: param, 'R2': r2_column,
                       'SMAPE': smape_column, 'RMSE': rmse_column, },
                      columns=[param_name, 'R2', 'SMAPE', 'RMSE'])
    return df



# Create function to evaluate 'c_range' parameter
def SVR_parameters_c():
    c_range = 10. ** np.arange(-3, 7)
    c_range2 = []
    c_range2.append((1000))
    for i in range(5000, 55000, 5000):
        c_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in c_range:
        clf = SVR(kernel='rbf', C=c, gamma=0.01)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('C', c_range, r2_column, smape_column, rmse_column)

    return df


## Print final dataframe
#print(SVR_parameters_c())

## Write final dataframe to excel file
#SVR_parameters_c(data, selected_features_rfe).to_excel('svr_parameters_rbf_C.xlsx', index=False)
#SVR_parameters_c(data, selected_features_rfe).to_excel('svr_parameters_rbf_C_detail.xlsx', index=False)


# Create function to evaluate 'gamma_range' parameter
def SVR_parameters_gamma():
    gamma_range = 10. ** np.arange(-4, 4)
    gamma_range2 = []
    gamma_range2.append(0.001)
    for i in np.arange(0.005, 0.055, 0.005):
        gamma_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for gamma in gamma_range2:
        clf = SVR(kernel='rbf', C=5000, gamma=gamma)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('gamma', gamma_range2, r2_column, smape_column, rmse_column)

    return df

## Print final dataframe
#print(SVR_parameters_gamma())

## Write final dataframe to excel file
#SVR_parameters_gamma(data, selected_features_rfe).to_excel('svr_parameters_gamma.xlsx', index=False)
#SVR_parameters_gamma(data, selected_features_rfe).to_excel('svr_parameters_gamma_detail.xlsx', index=False)


def SVR_parameters_epsilon():
    epsilon = [0, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    epsilon2 = range(30, 130, 10)

    r2_column = []
    smape_column = []
    rmse_column = []

    for e in epsilon2:
        clf = SVR(kernel='rbf', C=5000, gamma=0.025, epsilon=e)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('epsilon', epsilon2, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_epsilon())
#SVR_parameters_epsilon(data, selected_features_rfe).to_excel('svr_parameters_rbf_epsilon_detail.xlsx', index=False)


def SVR_parameters_poly_c():
    c_range = 10. ** np.arange(-3, 7)
    c_range2 = []
    c_range2.append((1000))
    for i in range(5000, 55000, 5000):
        c_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in c_range:
        clf = SVR(kernel='poly', C=c, gamma=0.01)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('C', c_range, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_poly_c())
#SVR_parameters_poly_c(data, selected_features_rfe).to_excel('svr_parameters_poly_C.xlsx', index=False)


def SVR_parameters_poly_gamma():
    gamma_range = 10. ** np.arange(-4, 2)
    gamma_range2 = []
    gamma_range2.append(0.001)
    for i in np.arange(0.005, 0.055, 0.005):
        gamma_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for gamma in gamma_range2:
        clf = SVR(kernel='poly', C=5000, gamma=gamma)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('gamma', gamma_range2, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_poly_gamma())
#SVR_parameters_poly_gamma(data, selected_features_rfe).to_excel('SVR_parameters_poly_gamma.xlsx', index=False)
#SVR_parameters_poly_gamma(data, selected_features_rfe).to_excel('SVR_parameters_poly_gamma_detail.xlsx', index=False)


def SVR_parameters_poly_degree():
    degree_range = range(2, 6)

    r2_column = []
    smape_column = []
    rmse_column = []

    for d in degree_range:
        clf = SVR(kernel='poly', C=5000, gamma=0.01, degree=d)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('degree', degree_range, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_poly_degree())
#SVR_parameters_poly_degree(data, selected_features_rfe).to_excel('svr_parameters_poly_degree.xlsx', index=False)


def SVR_parameters_poly_epsilon():
    epsilon = [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    epsilon2 = range(30, 130, 10)

    r2_column = []
    smape_column = []
    rmse_column = []

    for e in epsilon:
        clf = SVR(kernel='poly', C=5000, gamma=0.01, epsilon=e)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('epsilon', epsilon, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_poly_epsilon())
#SVR_parameters_poly_epsilon(data, selected_features_rfe).to_excel('SVR_parameters_poly_epsilon.xlsx', index=False)


def SVR_parameters_linear_c():
    c_range = 10. ** np.arange(-3, 7)
    c_range2 = []
    c_range2.append((1))
    for i in range(5, 55, 5):
        c_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in c_range2:
        clf = SVR(kernel='linear', C=c, gamma=0.01)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('C', c_range2, r2_column, smape_column, rmse_column)

    return df



#print(SVR_parameters_linear_c())
#SVR_parameters_linear_c(data, selected_features_rfe).to_excel('SVR_parameters_linear_c.xlsx', index=False)
#SVR_parameters_linear_c(data, selected_features_rfe).to_excel('SVR_parameters_linear_c_detail.xlsx', index=False)


def SVR_parameters_linear_epsilon():
    epsilon = [0, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    epsilon2 = range(20, 65, 5)

    r2_column = []
    smape_column = []
    rmse_column = []

    for e in epsilon2:
        clf = SVR(kernel='linear', C=5000, gamma=0.01, epsilon=e)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('epsilon', epsilon2, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_linear_epsilon())
#SVR_parameters_linear_epsilon(data, selected_features_rfe).to_excel('SVR_parameters_linear_epsilon_detail.xlsx', index=False)



def SVR_parameters_sigmoid_c():
    c_range = 10. ** np.arange(-3, 7)
    c_range2 = []
    c_range2.append((10))
    for i in range(50, 550, 50):
        c_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for c in c_range2:
        clf = SVR(kernel='sigmoid', C=c, gamma=0.01)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('C', c_range2, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_sigmoid_c())
#SVR_parameters_sigmoid_c(data, selected_features_rfe).to_excel('SVR_parameters_sigmoid_c.xlsx', index=False)
#SVR_parameters_sigmoid_c(data, selected_features_rfe).to_excel('SVR_parameters_sigmoid_c_detail.xlsx', index=False)


def SVR_parameters_sigmoid_gamma():

    gamma_range = 10. ** np.arange(-4, 2)
    gamma_range2 = []
    gamma_range2.append(0.001)
    for i in np.arange(0.005, 0.055, 0.005):
        gamma_range2.append(i)

    r2_column = []
    smape_column = []
    rmse_column = []

    for gamma in gamma_range2:
        clf = SVR(kernel='sigmoid', C=200, gamma=gamma)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('gamma', gamma_range2, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_sigmoid_gamma())
#SVR_parameters_sigmoid_gamma(data, selected_features_rfe).to_excel('SVR_parameters_sigmoid_gamma.xlsx', index=False)
#SVR_parameters_sigmoid_gamma(data, selected_features_rfe).to_excel('SVR_parameters_sigmoid_gamma_detail.xlsx', index=False)


def SVR_parameters_sigmoid_epsilon():
    epsilon = [0, 0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    epsilon2 = range(1, 10, 1)

    r2_column = []
    smape_column = []
    rmse_column = []

    for e in epsilon2:
        clf = SVR(kernel='sigmoid', C=200, gamma=0.005, epsilon=e)

        r2, smape, rmse = calculate_metrics(clf, X_train, X_test, y_train, y_test)

        r2_column.append(r2)
        smape_column.append(smape)
        rmse_column.append(rmse)


    df = create_dataframe('epsilon', epsilon2, r2_column, smape_column, rmse_column)

    return df


#print(SVR_parameters_sigmoid_epsilon())
#SVR_parameters_sigmoid_epsilon(data, selected_features_rfe).to_excel('SVR_parameters_sigmoid_epsilon_detail.xlsx', index=False)


#create parameters plot and save
def plot_paramaeters():
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(degree_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.savefig('svr_parameters_rbf.png')
    plt.close()





