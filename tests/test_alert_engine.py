"""
Unit and Integration Tests for Step 12B Alert Engine.

Proves all 10 required conditions:
1. AQI threshold is 200.
2. AQI uses P(AQI > 200).
3. Rainfall threshold is 64.5 mm / 24h.
4. Hourly rainfall is NOT directly compared to 64.5.
5. Rainfall alert uses a 24-hour accumulation.
6. Temperature does NOT use 37°C alone.
7. Temperature requires joint condition: Tmax >= 37°C AND Tmax - normal >= 4.5°C.
8. Missing temperature normal data is never fabricated (returns UNAVAILABLE_MISSING_CLIMATOLOGY).
9. Official physical thresholds are strictly separated from 20%/60% probability decision rules.
10. No test observations are fabricated (unmonitored zones have strictly zero evidence).
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.alerts.alert_engine import (
    AQI_POOR_THRESHOLD,
    RAINFALL_24H_HEAVY_THRESHOLD,
    TEMP_COASTAL_HEATWAVE_TMAX,
    TEMP_COASTAL_HEATWAVE_DEPARTURE,
    PROB_WATCH_CUTOFF,
    PROB_WARNING_CUTOFF,
    evaluate_aqi_alert,
    evaluate_rainfall_alert_24h,
    evaluate_temperature_heatwave_alert,
    compute_24h_rainfall_accumulation,
    compute_gaussian_tail_probability,
    classify_probabilistic_alert
)


def test_1_aqi_threshold_is_200():
    """Condition 1: AQI threshold is exactly 200.0 (CPCB Poor category boundary)."""
    assert AQI_POOR_THRESHOLD == 200.0
    res = evaluate_aqi_alert(mu=150.0, var=100.0)
    assert res['threshold'] == 200.0


def test_2_aqi_uses_p_greater_than_200():
    """Condition 2: AQI evaluation computes exact tail probability P(AQI > 200)."""
    # Test symmetrical case: mu = 200.0 -> P(AQI > 200) must equal 0.50
    mu = 200.0
    var = 25.0
    res = evaluate_aqi_alert(mu=mu, var=var)
    assert pytest.approx(res['probability_crossing'], abs=1e-3) == 0.50
    assert res['alert_status'] == 'WATCH'  # 0.50 is between 0.20 and 0.60

    # Test high mean: mu = 250.0, std = 10.0 -> P(AQI > 200) ≈ 1.0 -> WARNING
    res_high = evaluate_aqi_alert(mu=250.0, var=100.0)
    assert res_high['probability_crossing'] > 0.99
    assert res_high['alert_status'] == 'WARNING'

    # Test clean air: mu = 50.0, std = 10.0 -> P(AQI > 200) ≈ 0.0 -> NORMAL
    res_low = evaluate_aqi_alert(mu=50.0, var=100.0)
    assert res_low['probability_crossing'] < 0.001
    assert res_low['alert_status'] == 'NORMAL'


def test_3_rainfall_threshold_is_64_point_5():
    """Condition 3: Rainfall threshold is 64.5 mm / 24h."""
    assert RAINFALL_24H_HEAVY_THRESHOLD == 64.5
    res = evaluate_rainfall_alert_24h(mu_accum_24=30.0, var_accum_24=100.0, is_monitored=True)
    assert res['threshold'] == 64.5


def test_4_and_5_hourly_rainfall_not_compared_to_64_point_5_and_uses_24h_accumulation():
    """
    Conditions 4 & 5: Hourly rainfall is NOT directly compared to 64.5 mm.
    Rainfall alert uses a 24-hour accumulation sum.
    """
    # Suppose an hourly rate is 10.0 mm/h.
    # Compared hourly, 10.0 < 64.5 would trivially register 0 probability.
    # But sustained 10.0 mm/h for 24h produces 240.0 mm accumulation!
    hourly_rate = 10.0
    
    # 2-state HMM test: state 0 (0 mm/h), state 1 (10 mm/h)
    means = np.array([0.0, 10.0])
    variances = np.array([0.1, 1.0])
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    gamma_sustained = np.array([0.0, 1.0])  # strictly in 10 mm/h rain state

    mu_accum, var_accum = compute_24h_rainfall_accumulation(gamma_sustained, transmat, means, variances)
    
    # Accumulation must be >> 10 mm (in fact ~100-200 mm)
    assert mu_accum > 64.5, f"Expected 24h accumulation to exceed 64.5 mm, got {mu_accum}"
    
    # Evaluating 24h accumulation correctly yields high exceedance probability
    res = evaluate_rainfall_alert_24h(mu_accum, var_accum, is_monitored=True)
    assert res['probability_crossing'] > 0.60
    assert res['alert_status'] == 'WARNING'


def test_6_and_7_temperature_does_not_use_37_alone_and_requires_joint_departure():
    """
    Conditions 6 & 7: Temperature does NOT use 37°C alone.
    Requires joint condition: Tmax >= 37°C AND Tmax - Normal_Tmax >= 4.5°C.
    """
    # Case A: Tmax = 38.0°C (>= 37.0°C), but Normal is 36.0°C (departure = +2.0°C < 4.5°C).
    # Standalone 37°C would wrongly flag heatwave. Joint criteria must NOT flag heatwave!
    tmax = 38.0
    var = 0.5  # tight distribution
    normal = 36.0
    res_a = evaluate_temperature_heatwave_alert(tmax_pred=tmax, tmax_var=var, normal_tmax=normal)
    # Effective threshold = max(37.0, 36.0 + 4.5) = 40.5°C
    assert res_a['threshold'] == 40.5
    # Since tmax = 38.0°C < 40.5°C, probability is small -> NORMAL
    assert res_a['alert_status'] == 'NORMAL'

    # Case B: Tmax = 41.5°C (>= 37.0°C) and Normal is 36.0°C (departure = +5.5°C >= 4.5°C).
    # Both conditions met!
    res_b = evaluate_temperature_heatwave_alert(tmax_pred=41.5, tmax_var=var, normal_tmax=normal)
    assert res_b['alert_status'] == 'WARNING'


def test_8_missing_temperature_normal_is_never_fabricated():
    """Condition 8: Missing temperature normal data is never fabricated."""
    res = evaluate_temperature_heatwave_alert(tmax_pred=40.0, tmax_var=1.0, normal_tmax=None)
    assert res['alert_status'] == 'UNAVAILABLE_MISSING_CLIMATOLOGY'
    assert np.isnan(res['probability_crossing'])
    assert 'Verified climatological normal' in res['reason']


def test_9_official_thresholds_separated_from_probability_decision_rules():
    """
    Condition 9: Official physical thresholds (CPCB/IMD) are separated from
    our model's 20% and 60% probability decision rules.
    """
    # Check physical constants
    assert AQI_POOR_THRESHOLD == 200.0
    assert RAINFALL_24H_HEAVY_THRESHOLD == 64.5
    assert TEMP_COASTAL_HEATWAVE_TMAX == 37.0
    assert TEMP_COASTAL_HEATWAVE_DEPARTURE == 4.5

    # Check decision rule cutoffs
    assert PROB_WATCH_CUTOFF == 0.20
    assert PROB_WARNING_CUTOFF == 0.60

    # Decision rule mapping
    assert classify_probabilistic_alert(0.19) == 'NORMAL'
    assert classify_probabilistic_alert(0.20) == 'WATCH'
    assert classify_probabilistic_alert(0.59) == 'WATCH'
    assert classify_probabilistic_alert(0.60) == 'WARNING'
    assert classify_probabilistic_alert(0.95) == 'WARNING'


def test_10_no_test_observations_are_fabricated():
    """Condition 10: No test observations are fabricated."""
    df_test = pd.read_csv('data/processed/test.csv', low_memory=False)
    
    # Assert temperature observations are genuinely unobserved (0 count, NaN throughout)
    assert int(df_test['temperature_observed'].sum()) == 0
    assert df_test['temperature'].isna().all()

    # Assert unmonitored rainfall zones receive strictly UNMONITORED_INSUFFICIENT_DATA
    res_unmon = evaluate_rainfall_alert_24h(mu_accum_24=0.0, var_accum_24=0.0, is_monitored=False)
    assert res_unmon['alert_status'] == 'UNMONITORED_INSUFFICIENT_DATA'
    assert np.isnan(res_unmon['probability_crossing'])
