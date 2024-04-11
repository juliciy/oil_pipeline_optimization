
import numpy as np
from sklearn import  metrics
import math


def evaluate_test_with_constant_item(X_test, coef, y_test):
    y_predict = [np.dot(x, coef) for x in X_test]
    #mape = metrics.mean_absolute_percentage_error(y_test, y_predict)
    # print('mape', mape)
    print('coef ',coef)
    mse = metrics.mean_squared_error(y_test, y_predict)
    print('mse ', mse)

    rsquared = metrics.r2_score(y_test,y_predict)
    print('rsquared ',rsquared)

    return {'mse':mse,'rsquared':rsquared}

#主要是计算RMSE和R方
def evaluate_test_without_constant_item(X_test, coef, y_test):
    y_predict = [np.dot(np.insert(x,0,1.0), coef) for x in X_test]
    #mape = metrics.mean_absolute_percentage_error(y_test, y_predict)
    # print('mape', mape)
    print('coef ',coef)
    mse = metrics.mean_squared_error(y_test, y_predict)
    print('rmse ', math.sqrt(mse))

    rsquared = metrics.r2_score(y_test,y_predict)
    print('rsquared ',rsquared)

    return {'rmse':mse,'rsquared':rsquared}

def evaluate_linear_regression_rmse(X_test, coef, y_test):
    y_predict = [np.dot(np.insert(x,0,1.0), coef) for x in X_test]

    rmse = math.sqrt(metrics.mean_squared_error(y_test, y_predict))
    return rmse


def evaluate_linear_regression_rsquared(X, coef, y):
    y_predict = [np.dot(np.insert(x,0,1.0), coef) for x in X]

    rsquared = metrics.r2_score(y, y_predict)
    return rsquared





