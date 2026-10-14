"""
Promotion package for the frozen +6 h / +24 h temperature candidates (history-trained T4): the package holds the
unchanged frozen models, training-only (2017-2023) horizon-specific tables with the experiment's levels and
min_count, a complete zone mapping (zone 1 -> gc2 recorded), the 2024 gates; the live builder reproduces 2023
reference rows; the forecaster ignores the package unless explicitly enabled, and 1 h / 3 h output is unchanged.
Local files only; 2024 is never read.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import live_forecaster as lf
from src.forecasting.live_forecaster import LiveForecaster, RAIN_FAILED_EVAL
from src.models import package_history_models as ph
from src.models import temp_model as tm
from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ART = os.path.join(ROOT, ph.ARTIFACT)
packaged = pytest.mark.skipif(not os.path.exists(ART), reason='history temperature package not built')


@pytest.fixture(scope='module')
def bundle():
    return joblib.load(ART)


def _weather(origin, zones=range(1, 16)):
    idx = pd.date_range(origin - pd.Timedelta(hours=71), origin, freq='h')
    ist_h = idx.tz_convert('Asia/Kolkata').hour.to_numpy()
    return {z: pd.DataFrame({'temperature_2m_c': 29 + 3 * np.sin(2 * np.pi * (ist_h - 9) / 24) + 0.05 * z,
                             'relative_humidity_2m_pct': 70 - 10 * np.sin(2 * np.pi * (ist_h - 9) / 24),
                             'wind_speed_10m_kmh': 12.0, 'wind_direction_10m_deg': 120.0,
                             'precipitation_mm': 0.0}, index=idx) for z in zones}


# ── artifact, frozen models, gates ───────────────────────────────────────────
@packaged
def test_package_holds_the_unchanged_frozen_models_and_2024_gates(bundle):
    frozen = joblib.load(os.path.join(ROOT, ph.FROZEN_ARTIFACT))
    assert set(bundle['models']) == {'6h', '24h'}
    for key in ('6h', '24h'):
        m, f = bundle['models'][key], frozen['models'][('temperature', key)]
        assert isinstance(m, tm.HourlyTempModel) and m.horizon_h == int(key[:-1])
        assert m.features == tm.TEMP_FEATURES + tm.TEMP_CANDIDATE_EXTRA == f.features
        np.testing.assert_array_equal(m.coef_, f.coef_)                     # no refit, no retune
        assert m.resid_std_ == f.resid_std_
        g = bundle['gate'][key]
        assert g['passed'] is True and g['test_period'] == '2024' and g['test_skill'] > 0 and g['test_ci_week'][0] > 0
    lock = json.load(open(os.path.join(ROOT, ph.PROTOCOL_LOCK), encoding='utf-8'))
    meta = bundle['meta']
    assert meta['frozen_artifact_sha256'] == lock['sha256']['artifact_long_horizon_history'] == ph._sha256(ph.FROZEN_ARTIFACT)
    assert meta['protocol_sha256'] == lock['sha256']['protocol']
    assert meta['status'].startswith('PACKAGED FOR PROMOTION REVIEW')


@packaged
def test_tables_are_horizon_specific_training_only_with_the_experiment_conventions(bundle):
    meta = bundle['meta']
    assert tuple(meta['training_period']) == ph.TRAIN and meta['training_period'][1].startswith('2023-12-31')
    for key in ('6h', '24h'):
        enc = bundle['encodings'][key]
        assert set(enc) == {'clim_target', 'clim_origin'}
        assert enc['clim_target']['levels'] == tv.TEMP_CLIM_TARGET_LEVELS and enc['clim_origin']['levels'] == tv.TEMP_CLIM_ORIGIN_LEVELS
        assert all(t['min_count'] == tv.ENCODING_MIN_COUNT == 20 and t['per_cell'] for t in enc.values())
        n = sum(c for (_, c) in (lv[-1][()] for lv in enc['clim_target']['stats'].values()))   # [] level = all rows
        assert n == meta['table_rows_train'][key] and 8 * 61000 < n < 8 * 61400   # ~7 years x 8 cells, no other years
        assert sorted(enc['clim_target']['stats']) == [f'gc{i}' for i in range(1, 9)]
    # each horizon has its own tables: origin-time climatology differs (origins shift by h); the target-time table
    # covers the same target hours for both horizons, so it may coincide
    a, b = bundle['encodings']['6h']['clim_origin'], bundle['encodings']['24h']['clim_origin']
    assert a is not b and a['stats']['gc2'][0] != b['stats']['gc2'][0]


def test_table_building_ignores_non_training_rows():
    idx = pd.date_range('2022-11-01', periods=24 * 120, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    frames = {c: pd.DataFrame({'temperature_2m_c': 28 + rng.normal(0, 1, len(idx)), 'relative_humidity_2m_pct': 70.0,
                               'wind_speed_10m_kmh': 10.0, 'wind_direction_10m_deg': 90.0, 'precipitation_mm': 0.0},
                              index=idx) for c in ('gc1', 'gc2')}
    splits = {'train': ('2022-11-01 00:00', '2022-12-31 23:00'), 'test': ('2023-01-01 00:00', '2023-02-27 23:00')}
    df = ph.table_frame(frames, 6, splits)
    t1 = ph.build_tables(df)
    df2 = df.copy()
    df2.loc[df2['split'] == 'test', ['y', 't_now']] += 50.0
    assert ph.build_tables(df2) == t1


# ── zone mapping ─────────────────────────────────────────────────────────────
@packaged
def test_all_15_zones_mapped_and_zone1_follows_the_inland_live_cell(bundle):
    z2c, rep = bundle['zone_to_cell'], bundle['rep_zone']
    assert sorted(z2c) == list(range(1, 16))
    assert z2c[1] == 'gc2' and z2c[2] == 'gc2' and z2c[3] == 'gc2'
    note = bundle['meta']['zone1_mapping']
    assert note['cell'] == 'gc2' and 'inland' in note['note'] and 'gc1' in note['note']
    assert bundle['meta']['unused_cells'] == ['gc1'] and 'gc1' not in rep
    assert all(z2c[z] == c for c, z in rep.items()) and set(rep) == set(z2c.values())
    prod = joblib.load(os.path.join(ROOT, 'models', 'temperature', 'temp_hourly_openmeteo.joblib'))
    assert prod['zone_to_cell'][1] == 'cell_1'                              # production mapping left as it was


# ── live feature parity on 2023 reference rows ───────────────────────────────
@packaged
def test_live_builder_reproduces_2023_reference_rows(bundle):
    ref_path = os.path.join(ROOT, ph.REFERENCE_CSV)
    if not os.path.exists(ref_path):
        pytest.skip('reference rows not written')
    ref = pd.read_csv(ref_path, float_precision='round_trip')
    assert set(ref['horizon']) == {'6h', '24h'} and len(ref) >= 2 * 7 * 30
    ts = pd.to_datetime(ref['origin_utc'], utc=True)
    assert (ts.dt.year == 2023).all()
    dev, _, _ = ph.history_frames()
    res = ph.verify_live_parity(bundle, ref, dev)
    assert res['passed'] and res['tolerance'] == 1e-9
    for key in ('6h', '24h'):
        assert res[key]['rows'] == (ref['horizon'] == key).sum()
        assert res[key]['max_feature_diff'] <= 1e-9 and res[key]['max_prediction_diff'] <= 1e-9


# ── forecaster: optional, disabled by default; 1 h / 3 h unchanged ───────────
def test_history_package_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(lf.TEMP_HISTORY_ENV, raising=False)
    assert lf.temp_history_enabled() is False and LiveForecaster().temp_history is None
    assert lf.temp_history_enabled({lf.TEMP_HISTORY_ENV: '1'}) and not lf.temp_history_enabled({lf.TEMP_HISTORY_ENV: '0'})


@packaged
def test_enabled_package_serves_6h_24h_only_and_leaves_1h_3h_identical(monkeypatch):
    monkeypatch.delenv(lf.TEMP_HISTORY_ENV, raising=False)
    off, on = LiveForecaster(), LiveForecaster(enable_temp_history=True)
    origin = pd.Timestamp('2026-06-15 06:00', tz='UTC')
    weather = _weather(origin)
    temps = {z: w['temperature_2m_c'].iloc[-48:] for z, w in weather.items()}
    a = off.run_temperature_hourly(temps, origin, weather_by_zone=weather)
    b = on.run_temperature_hourly(temps, origin, weather_by_zone=weather)
    assert a['supported_horizons'] == ['1h', '3h'] and b['supported_horizons'] == ['1h', '3h', '6h', '24h']
    for key in ('1h', '3h'):
        assert a['horizons'][key] == b['horizons'][key]                     # byte-for-byte same records
    for key in ('6h', '24h'):
        assert all(r['alert_status'] == RAIN_FAILED_EVAL and r['temp_pred_c'] is None for r in a['horizons'][key])
        hb = on.temp_history
        m = hb['models'][key]
        for r in b['horizons'][key]:
            z = r['zone_id']
            cell = hb['zone_to_cell'][z]
            x = lf.temp_candidate_row(weather[hb['rep_zone'][cell]], m.horizon_h, cell, hb['encodings'][key])
            assert r['availability'] == 'MODEL_FORECAST' and r['grid_cell'] == cell
            assert r['temp_pred_c'] == round(float(m.predict(x[m.features])[0]), 2)
            assert r['evaluation']['passed'] is True and r['evaluation']['test_period'] == '2024'
            assert 'temp_hourly_history_6h_24h' in r['model_package']
        assert {r['zone_id']: r['grid_cell'] for r in b['horizons'][key]}[1] == 'gc2'
    no_w = on.run_temperature_hourly(temps, origin, weather_by_zone={})
    assert all(r['temp_pred_c'] is None for k in ('6h', '24h') for r in no_w['horizons'][k])   # never filled


@packaged
def test_package_is_separate_from_production_artifacts():
    assert os.path.normpath(ph.ARTIFACT) not in {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS}
    assert os.path.normpath(ph.ARTIFACT) != os.path.normpath(os.path.join('models', lf.TEMP_HOURLY_ARTIFACT))
    with pytest.raises(FileExistsError):
        ph.build_package(save=True, replace=False)                          # never silently overwritten
