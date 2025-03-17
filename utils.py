import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller


def detect_outliers_iqr(column, tolerance=18):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - tolerance * IQR
    upper_bound = Q3 + tolerance * IQR
    return column[(column < lower_bound) | (column > upper_bound)]


def test_stationarity(series):
    result = adfuller(series)
    return result[1]  # p-value


def split_sequences_multi_output(data, n_steps, forecast_horizon, n_features_out):
    X, y = [], []
    for i in range(len(data) - n_steps - forecast_horizon + 1):
        X.append(data[i : i + n_steps])
        y.append(data[i + n_steps : i + n_steps + forecast_horizon, :n_features_out])
    return np.array(X), np.array(y)


def convert_dates_to_days_since_reference(date_str, reference_date="2020-01-01"):
    ref_date = pd.to_datetime(reference_date)
    date = pd.to_datetime(date_str)
    return (date - ref_date).days


def weighted_rmse(y_true, y_pred):
    weights = tf.where(y_true != 0, 1.0, 0.5)
    return tf.sqrt(tf.reduce_mean(weights * tf.square(y_true - y_pred)))
