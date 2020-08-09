import numpy as np


def r2_score(y_test, y_pred):
    pass


def mean_absolute_error(y_test, y_pred):
    return np.mean(np.absolute(y_test - y_pred))


def mean_squared_error(y_test, y_pred):
    return np.mean(np.square(y_test - y_pred))


def root_mean_squared_error(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)
