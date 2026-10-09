import pytest
import numpy as np
import pandas as pd
from src.preprocessing.cpcb_aqi import calculate_sub_index, compute_cpcb_aqi_row, aqi_category

def test_pm25_sub_index():
    assert calculate_sub_index('PM2.5', 15) == 25.0
    assert calculate_sub_index('PM2.5', 30) == 50.0
    assert calculate_sub_index('PM2.5', 60) == 100.0
    assert calculate_sub_index('PM2.5', 90) == 200.0
    assert calculate_sub_index('PM2.5', 120) == 300.0
    assert calculate_sub_index('PM2.5', 250) == 400.0

def test_pm10_sub_index():
    assert calculate_sub_index('PM10', 25) == 25.0
    assert calculate_sub_index('PM10', 50) == 50.0
    assert calculate_sub_index('PM10', 100) == 100.0
    assert calculate_sub_index('PM10', 250) == 200.0

def test_co_sub_index():
    assert calculate_sub_index('CO', 0.5) == 25.0
    assert calculate_sub_index('CO', 1.0) == 50.0
    assert calculate_sub_index('CO', 2.0) == 100.0
    assert calculate_sub_index('CO', 10.0) == 200.0

def test_minimum_criteria():
    # Only 2 pollutants -> NaN
    sub_indices_2 = {'PM2.5': 80.0, 'NO2': 60.0}
    val, dom, cnt = compute_cpcb_aqi_row(sub_indices_2)
    assert pd.isna(val)
    assert cnt == 2

    # 3 pollutants, but NO PM -> NaN
    sub_indices_no_pm = {'NO2': 60.0, 'SO2': 40.0, 'CO': 50.0}
    val, dom, cnt = compute_cpcb_aqi_row(sub_indices_no_pm)
    assert pd.isna(val)
    assert cnt == 3

    # 3 pollutants WITH PM -> Valid max
    sub_indices_valid = {'PM2.5': 120.0, 'NO2': 60.0, 'SO2': 40.0}
    val, dom, cnt = compute_cpcb_aqi_row(sub_indices_valid)
    assert val == 120.0
    assert dom == 'PM2.5'
    assert cnt == 3

def test_aqi_categories():
    assert aqi_category(35) == 'Good'
    assert aqi_category(75) == 'Satisfactory'
    assert aqi_category(150) == 'Moderate'
    assert aqi_category(250) == 'Poor'
    assert aqi_category(350) == 'Very Poor'
    assert aqi_category(450) == 'Severe'
    assert aqi_category(np.nan) == 'Unavailable'
