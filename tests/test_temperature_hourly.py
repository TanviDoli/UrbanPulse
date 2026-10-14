"""
Hourly temperature forecasting (src/models/temp_model.py + train_temperature_hourly): forecast-time
features, split boundaries, train-only baselines, unrounded gate, saved artifact, live integration,
pipeline output and dashboard. No model is fitted here except on tiny synthetic data via lstsq.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collectors import live_collect
from src.forecasting.live_forecaster import LiveForecaster, RAIN_FAILED_EVAL
from src.models import train_and_validate_hmm as tv
from src.models import temp_model as tm
from dashboard import data_loader as dl

ARTIFACT = os.path.join(os.path.dirname(__file__), '..', tv.TEMP_HOURLY_ARTIFACT)
needs_artifact = pytest.mark.skipif(not os.path.exists(ARTIFACT), reason='hourly temperature artifact not saved')


def _diurnal(start, hours, base=29.0):
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    ist_h = idx.tz_convert('Asia/Kolkata').hour.to_numpy()
    return pd.Series(base + 3 * np.sin(2 * np.pi * (ist_h - 9) / 24), index=idx)


def test_features_use_only_values_known_at_the_origin():
    s = _diurnal('2025-06-01', 200)
    s.iloc[::7] += 0.5
    for h in (1, 3, 6, 24):
        f = tm.temp_features(s, h)
        t = 120
        s2 = s.copy()
        s2.iloc[t + 1:] = 99.0
        pd.testing.assert_frame_equal(f.iloc[:t + 1], tm.temp_features(s2, h).iloc[:t + 1])
        assert f['t_same_hour_yday'].iloc[t] == s.iloc[t + h - 24]           # same IST hour as the target, observed
    with pytest.raises(ValueError):
        tm.temp_features(s, 25)


def test_rows_respect_split_boundaries():
    start = pd.Timestamp(tv.RAIN_OM_SPLITS['validate'][0], tz='UTC') - pd.Timedelta(days=4)
    cells = {'cell_1': _diurnal(start, 24 * 8)}
    train_end = pd.Timestamp(tv.RAIN_OM_SPLITS['train'][1], tz='UTC')
    for h in tv.TEMP_HOURLY_HORIZONS:
        df = tv.temp_rows(cells, h)
        assert df[df['split'] == 'train'].index.max() == train_end - pd.Timedelta(hours=h)
        assert df[df['split'] == 'validate'].index.min() == train_end + pd.Timedelta(hours=1)
        i = cells['cell_1'].index.get_loc(df.index[0])
        assert df['y'].iloc[0] == cells['cell_1'].iloc[i + h]


def test_lstsq_model_recovers_persistence_and_baselines_are_train_only():
    start = pd.Timestamp(tv.RAIN_OM_SPLITS['validate'][0], tz='UTC') - pd.Timedelta(days=6)
    rng = np.random.default_rng(1)
    s = _diurnal(start, 24 * 12) + np.cumsum(rng.normal(0, 0.05, 24 * 12))
    df = tv.temp_rows({'cell_1': s}, 1)
    tr = df[df['split'] == 'train']
    model = tm.HourlyTempModel(1).fit(tr[tm.TEMP_FEATURES], tr['y'].to_numpy())
    assert np.isfinite(model.coef_).all() and model.resid_std_ >= 0
    m = tv.evaluate_temp(df, model.predict(df[tm.TEMP_FEATURES]), '1h')
    pooled = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL')]
    assert {'train', 'validate'} == set(pooled['split'])
    assert pooled[['skill_ci_low', 'skill_ci_high']].notna().all().all()
    # climatology baseline: validation targets must not leak into it
    y_leak = df['y'].to_numpy().copy()
    y_leak[(df['split'] == 'validate').to_numpy()] += 50
    lv = [['target_month', 'target_hour'], ['target_hour'], []]
    np.testing.assert_allclose(tv._train_lookup(df, df['y'].to_numpy(), lv), tv._train_lookup(df, y_leak, lv))


def test_gate_is_strict_and_unrounded():
    rows = []
    vals = {('1h', 'validate'): 1e-12, ('3h', 'test'): 0.0, ('6h', 'validate'): np.nan, ('24h', 'test'): -0.01}
    for h in tv.TEMP_HOURLY_HORIZONS:
        for split in ('validate', 'test'):
            rows.append({'horizon': f'{h}h', 'split': split, 'cell': 'ALL', 'season': 'ALL',
                         'skill': vals.get((f'{h}h', split), 0.2), 'best_baseline': 'persistence', 'rmse_model': 1.0})
    g = tv.temp_gate(pd.DataFrame(rows))
    assert g['1h']['passed'] is True and not any(g[k]['passed'] for k in ('3h', '6h', '24h'))


@needs_artifact
def test_saved_artifact_passes_only_1h_and_3h():
    import joblib
    b = joblib.load(ARTIFACT)
    assert {k: g['passed'] for k, g in b['gate'].items()} == {'1h': True, '3h': True, '6h': False, '24h': False}
    assert sorted(b['zone_to_cell']) == list(range(1, 16))
    assert b['gate']['1h']['validate_skill'] > 0 and b['gate']['1h']['test_skill'] > 0


@needs_artifact
def test_live_hourly_temperature_publishes_validated_horizons_only():
    fc = LiveForecaster()
    origin = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    temps = {z: _diurnal(origin - pd.Timedelta(hours=47), 48) for z in range(1, 16)}
    temps[7] = temps[7].copy()
    temps[7].iloc[-5] = np.nan                                     # incomplete input -> unavailable, not filled
    weather = {}
    for z in range(1, 16):                                         # 72 h frames for the promoted 3 h model
        t72 = _diurnal(origin - pd.Timedelta(hours=71), 72)
        weather[z] = pd.DataFrame({'temperature_2m_c': t72, 'relative_humidity_2m_pct': 70.0,
                                   'wind_speed_10m_kmh': 10.0, 'wind_direction_10m_deg': 90.0,
                                   'precipitation_mm': 0.0})
    weather[7] = weather[7].copy()                                 # zone 7 is its own cell's representative
    weather[7].iloc[-5, 0] = np.nan
    out = fc.run_temperature_hourly(temps, origin, weather_by_zone=weather)
    assert out['supported_horizons'] == ['1h', '3h'] and out['source'] == 'open_meteo_hourly_regression'
    for key, h in (('1h', 1), ('3h', 3)):
        for r in out['horizons'][key]:
            if r['zone_id'] == 7:
                assert r['availability'] == 'UNAVAILABLE' and r['temp_pred_c'] is None
                continue
            assert r['availability'] == 'MODEL_FORECAST' and 20 < r['temp_pred_c'] < 40 and r['pred_std'] > 0
            assert r['alert_status'] == 'NOT_EVALUATED_HEATWAVE_CRITERION' and r['exceedance_prob'] is None
            assert r['target_time_utc'] == (origin + pd.Timedelta(hours=h)).isoformat()
    for key in ('6h', '24h'):
        for r in out['horizons'][key]:
            assert r['alert_status'] == RAIN_FAILED_EVAL and r['temp_pred_c'] is None
            assert dl.get_risk_level(r['alert_status']) == 'Unavailable'


@needs_artifact
def test_pipeline_and_dashboard_show_hourly_temperature(tmp_path, monkeypatch):
    apptest = pytest.importorskip('streamlit.testing.v1')
    import streamlit as st
    from src.live import live_pipeline
    now = pd.Timestamp.now(tz='UTC').floor('h') + pd.Timedelta(minutes=5)
    t = pd.date_range(now.floor('h') - pd.Timedelta(days=35), now.floor('h'), freq='h')
    ist_h = t.tz_convert('Asia/Kolkata').hour.to_numpy()
    temp = 29 + 3 * np.sin(2 * np.pi * (ist_h - 9) / 24)
    pd.DataFrame([{'zone_id': z, 'time_utc': x, 'temperature_2m_c': v, 'precipitation_mm': 0.0,
                   'relative_humidity_2m_pct': 70, 'wind_speed_10m_kmh': 10, 'wind_direction_10m_deg': 90,
                   'grid_latitude': 13.0, 'grid_longitude': 80.2, 'source': 'test'}
                  for z in range(1, 16) for x, v in zip(t, temp)]).to_csv(tmp_path / 'om.csv', index=False)
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(tmp_path / 'none.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(tmp_path / 'om.csv'))
    monkeypatch.setattr(live_collect, 'load_status', lambda: {'open_meteo': {'ok': True, 'latest_record_utc': now.isoformat()}})
    out_path = tmp_path / 'pred.json'
    live_pipeline.run_once(fetch=False, now_utc=now, output_path=str(out_path), verbose=False)
    d = json.loads(out_path.read_text(encoding='utf-8'))
    assert d['meta']['supported_horizons']['temperature'] == ['1h', '3h']
    assert d['meta']['data_times']['temperature_forecast_origin_utc'] == now.floor('h').isoformat()
    assert 'hourly 2 m temperature' in d['meta']['temperature_limitation']
    v = dl.forecast_view(d, 3, 'temperature', '1h', now)
    assert v['forecast_freshness'] == 'CURRENT' and v['temp_pred_c'] is not None
    monkeypatch.setattr(dl, 'PREDICTIONS_PATH', str(out_path))
    st.cache_data.clear()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    monkeypatch.chdir(root)
    at = apptest.AppTest.from_file(os.path.join(root, 'dashboard', 'app.py'), default_timeout=120)
    at.run()
    for hz, text in (('1 Hour', '2 m air at'), ('3 Hours', '2 m air at'),
                     ('6 Hours', 'Temperature: the 6 hours horizon did not beat')):
        at.selectbox(key='horizon_dropdown').select(hz).run()
        assert not at.exception, [e.message for e in at.exception]
        assert text in ' '.join(m.value for m in at.markdown), hz
    st.cache_data.clear()
