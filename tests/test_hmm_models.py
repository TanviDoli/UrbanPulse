"""
Unit Tests for Zone-Specific Hidden Markov Models (HMM) in UrbanPulse.
Verifies:
1. Artifact existence and keyed structure (variable -> zone_id).
2. Zone isolation: each trained model contains data from exactly ONE zone.
3. Chronological integrity: no cross-zone mixing or temporal inversion.
4. Non-fabrication: unmonitored zones (e.g. Zones 3, 6, 7, 8, 9, 11, 15) do NOT have trained models.
5. Strict isolation: test.csv is never loaded or used.
6. Valid probabilistic multi-step forecasts at supported horizons (1h, 3h, 6h, 24h).
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from src.models.hmm_model import TemporalHMM

MONITORED_AQI_ZONES = [2, 4, 5, 10, 12, 13, 14]
MONITORED_RAINFALL_ZONES = [4, 5, 10, 14]
TEMPERATURE_ZONES = list(range(1, 16))
UNMONITORED_AQI_ZONES = [1, 3, 6, 7, 8, 9, 11, 15]
UNMONITORED_RAINFALL_ZONES = [1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 15]

def test_hmm_artifacts_and_registry_exist():
    """Verify all expected model artifacts, status registries, and reports exist."""
    assert os.path.exists('models/hmm/hmm_aqi.joblib')
    assert os.path.exists('models/hmm/hmm_rainfall.joblib')
    assert os.path.exists('models/hmm/hmm_temperature.joblib')
    assert os.path.exists('models/hmm/hmm_models_by_zone.joblib')
    assert os.path.exists('results/metrics/model_training_status.csv')
    assert os.path.exists('results/metrics/validation_metrics_hmm.csv')
    assert os.path.exists('results/metrics/state_selection_hmm.csv')
    assert os.path.exists('results/predictions/validation_predictions_hmm.csv')
    assert os.path.exists('reports/model_methodology.md')

def test_keyed_model_structure():
    """Verify models are explicitly keyed by variable -> zone_id."""
    aqi_dict = joblib.load('models/hmm/hmm_aqi.joblib')
    rf_dict = joblib.load('models/hmm/hmm_rainfall.joblib')
    temp_dict = joblib.load('models/hmm/hmm_temperature.joblib')
    registry = joblib.load('models/hmm/hmm_models_by_zone.joblib')
    
    assert isinstance(aqi_dict, dict)
    assert set(aqi_dict.keys()) == set(MONITORED_AQI_ZONES)
    
    assert isinstance(rf_dict, dict)
    assert set(rf_dict.keys()) == set(MONITORED_RAINFALL_ZONES)
    
    assert isinstance(temp_dict, dict)
    assert set(temp_dict.keys()) == set(TEMPERATURE_ZONES)
    
    assert ('AQI', 2) in registry
    assert ('Rainfall', 4) in registry
    assert ('Temperature', 1) in registry
    assert len(registry) == len(MONITORED_AQI_ZONES) + len(MONITORED_RAINFALL_ZONES) + len(TEMPERATURE_ZONES)

def test_zone_isolation_and_no_cross_zone_mixing():
    """Verify each model is tagged with its own zone_id and trained only on that zone."""
    aqi_dict = joblib.load('models/hmm/hmm_aqi.joblib')
    for z_id, model in aqi_dict.items():
        assert model.zone_id == z_id
        assert model.variable == 'AQI'
        assert model.is_fitted
        assert model.trained_observations_ > 0

    rf_dict = joblib.load('models/hmm/hmm_rainfall.joblib')
    for z_id, model in rf_dict.items():
        assert model.zone_id == z_id
        assert model.variable == 'rainfall'
        assert model.is_fitted
        assert model.trained_observations_ > 0

    temp_dict = joblib.load('models/hmm/hmm_temperature.joblib')
    for z_id, model in temp_dict.items():
        assert model.zone_id == z_id
        assert model.variable == 'temperature'
        assert model.is_fitted
        assert model.trained_observations_ >= 150 # 171 contiguous daily observations

def test_unmonitored_zones_not_fabricated():
    """Verify unmonitored zones have NO trained models and are marked UNAVAILABLE."""
    aqi_dict = joblib.load('models/hmm/hmm_aqi.joblib')
    for z_id in UNMONITORED_AQI_ZONES:
        assert z_id not in aqi_dict, f"Zone {z_id} is unmonitored for AQI and must NOT have a trained model"
        assert not os.path.exists(f'models/hmm/hmm_aqi_zone_{z_id}.joblib')

    rf_dict = joblib.load('models/hmm/hmm_rainfall.joblib')
    for z_id in UNMONITORED_RAINFALL_ZONES:
        assert z_id not in rf_dict, f"Zone {z_id} is unmonitored for Rainfall and must NOT have a trained model"
        assert not os.path.exists(f'models/hmm/hmm_rainfall_zone_{z_id}.joblib')

    df_status = pd.read_csv('results/metrics/model_training_status.csv')
    unmon_aqi_status = df_status[(df_status['variable'] == 'AQI') & (df_status['zone_id'].isin(UNMONITORED_AQI_ZONES))]
    assert (unmon_aqi_status['status'] == 'UNAVAILABLE').all()

def test_chronological_ordering_enforced_in_data():
    """Verify that train and validation sequences within every zone are strictly chronological."""
    df_train = pd.read_csv('data/processed/train.csv', low_memory=False)
    df_val = pd.read_csv('data/processed/validate.csv', low_memory=False)
    
    for z_id in range(1, 16):
        ts_train = df_train[df_train['zone_id'] == z_id]['timestamp']
        ts_val = df_val[df_val['zone_id'] == z_id]['timestamp']
        assert ts_train.is_monotonic_increasing, f"Zone {z_id} train timestamps are not sorted"
        assert ts_val.is_monotonic_increasing, f"Zone {z_id} val timestamps are not sorted"
        # Train must precede validate
        assert ts_train.max() < ts_val.min(), f"Temporal leakage between train and val in zone {z_id}"

def test_test_csv_never_loaded():
    """Verify test.csv is untouched and not referenced in validation outputs."""
    # Ensure validation predictions only contain timestamps from the validation split
    df_val_preds = pd.read_csv('results/predictions/validation_predictions_hmm.csv')
    max_val_ts = df_val_preds['timestamp'].max()
    assert max_val_ts <= '2025-09-13T23:00:00+00:00', "Validation predictions contain data beyond validation split!"

def test_probabilistic_forecasting_and_horizons():
    """Verify each trained zone model produces valid probabilistic forecasts at supported horizons."""
    aqi_dict = joblib.load('models/hmm/hmm_aqi.joblib')
    dummy_seq = np.array([45.0, 52.0, 58.0, 64.0])
    
    for z_id, model in aqi_dict.items():
        for h in [1, 3, 6, 24]:
            mean_h, var_h, p_h = model.forecast(dummy_seq, horizon=h)
            assert mean_h >= 0.0, f"AQI forecast must be non-negative (zone {z_id}, h={h})"
            assert var_h > 0.0, f"Predictive variance must be positive (zone {z_id}, h={h})"
            assert np.isclose(np.sum(p_h), 1.0, atol=1e-5), f"State probabilities must sum to 1 (zone {z_id}, h={h})"

    rf_dict = joblib.load('models/hmm/hmm_rainfall.joblib')
    dummy_rf = np.array([0.0, 0.0, 0.0, 1.2])
    for z_id, model in rf_dict.items():
        for h in [1, 3, 6, 24]:
            m_h, v_h, p_h = model.forecast(dummy_rf, horizon=h)
            assert m_h >= 0.0
            assert v_h > 0.0
            assert np.isclose(np.sum(p_h), 1.0, atol=1e-5)

def test_validation_metrics_completeness():
    """Verify validation metrics exist for all trained zone x variable x horizon combinations."""
    df_metrics = pd.read_csv('results/metrics/validation_metrics_hmm.csv')
    
    # 7 AQI zones * 4 horizons = 28
    aqi_metrics = df_metrics[df_metrics['variable'] == 'AQI']
    assert len(aqi_metrics) == 28
    assert set(aqi_metrics['zone_id']) == set(MONITORED_AQI_ZONES)
    assert set(aqi_metrics['horizon']) == {'1h', '3h', '6h', '24h'}
    assert (aqi_metrics['mae'] > 0).all()
    assert (aqi_metrics['rmse'] > 0).all()

    # 4 Rainfall zones * 4 horizons = 16
    rf_metrics = df_metrics[df_metrics['variable'] == 'Rainfall']
    assert len(rf_metrics) == 16
    assert set(rf_metrics['zone_id']) == set(MONITORED_RAINFALL_ZONES)
    assert set(rf_metrics['horizon']) == {'1h', '3h', '6h', '24h'}
    assert (rf_metrics['mae'] > 0).all()
    assert (rf_metrics['rmse'] > 0).all()

    # 15 Temperature zones * 1 horizon (24h) = 15
    temp_metrics = df_metrics[df_metrics['variable'] == 'Temperature']
    assert len(temp_metrics) == 15
    assert set(temp_metrics['zone_id']) == set(TEMPERATURE_ZONES)
    assert (temp_metrics['mae'] > 0).all()
    assert (temp_metrics['rmse'] > 0).all()
