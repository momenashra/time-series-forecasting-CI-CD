import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
from utils import detect_outliers_iqr, split_sequences_multi_output, convert_dates_to_days_since_reference, weighted_rmse

# Sample Data for Testing
df_sample = pd.DataFrame({
    'quantity': [10, 15, 14, 13, 12, 1000],  # Last value is an outlier
    'UnitePrice': [5, 5.5, 5.2, 5.1, 5.3, 100],
    'discount': [1, 1.2, 1.1, 1.3, 1.2, 50]
})

def test_detect_outliers_iqr():
    outliers = detect_outliers_iqr(df_sample['quantity'])
    assert 1000 in outliers.values, "Outlier detection failed!"

def test_stationarity():
    stationary_series = np.random.randn(100)  # Stationary series
    non_stationary_series = np.cumsum(np.random.randn(100))  # Non-stationary
    assert adfuller(stationary_series)[1] < 0.05, "Stationary series detected as non-stationary!"
    assert adfuller(non_stationary_series)[1] > 0.05, "Non-stationary series detected as stationary!"

def test_split_sequences_multi_output():
    data = np.array([[i] for i in range(20)])  # Fake time series data
    X, y = split_sequences_multi_output(data, n_steps=3, forecast_horizon=2, n_features_out=1)
    assert X.shape[0] > 0 and y.shape[0] > 0, "Sequence splitting failed!"

def test_convert_dates_to_days_since_reference():
    date_str = '2022-03-31'
    days = convert_dates_to_days_since_reference(date_str)
    assert isinstance(days, int), "Date conversion failed!"

def test_weighted_rmse():
    y_true = tf.constant([1, 2, 3, 0, 5], dtype=tf.float32)
    y_pred = tf.constant([1.1, 2.2, 2.8, 0.1, 4.9], dtype=tf.float32)
    loss = weighted_rmse(y_true, y_pred)
    assert loss.numpy() > 0, "RMSE calculation failed!"

if __name__ == "__main__":
    pytest.main()
