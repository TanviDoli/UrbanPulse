"""
Controlled promotion of temperature 3 h and rainfall 3 h: promoted models load in the live forecaster,
their inputs are produced live in stored order and reproduce the experiment, the 1 h models and the
failed horizons are unchanged, AQI is untouched, and rollback restores the previous files byte-for-byte.
"""

import glob
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import live_forecaster as lf
from src.forecasting.live_forecaster import LiveForecaster, RAIN_FAILED_EVAL
from src.models import temp_model as tm
from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MANIFESTS = sorted(glob.glob(os.path.join(ROOT, 'models', 'backups', 'promotion_*', 'manifest.json')))
promoted = pytest.mark.skipif(not MANIFESTS, reason='3 h promotion not applied')


@pytest.fixture(scope='module')
def fc():
    return LiveForecaster()


def _weather(origin, zones=range(1, 16), rain_now=False):
    idx = pd.date_range(origin - pd.Timedelta(hours=71), origin, freq='h')
    ist_h = idx.tz_convert('Asia/Kolkata').hour.to_numpy()
    rain = np.where(np.arange(len(idx)) >= len(idx) - 2, 1.5, 0.0) if rain_now else np.zeros(len(idx))
    return {z: pd.DataFrame({'temperature_2m_c': 29 + 3 * np.sin(2 * np.pi * (ist_h - 9) / 24),
                             'relative_humidity_2m_pct': 70 - 10 * np.sin(2 * np.pi * (ist_h - 9) / 24),
                             'wind_speed_10m_kmh': 12.0, 'wind_direction_10m_deg': 120.0,
                             'precipitation_mm': rain}, index=idx) for z in zones}


@promoted
def test_promoted_models_load_with_their_stored_features_and_tables(fc):
    t, r = fc.temp_hourly, fc.rain_twopart
    assert t['models']['3h'].features == tm.TEMP_FEATURES + tm.TEMP_CANDIDATE_EXTRA
    assert r['models']['3h'].feature_names[-len(tv.RAIN_CANDIDATE_EXTRA):] == tv.RAIN_CANDIDATE_EXTRA
    assert set(t['encodings']) == {'3h'} and set(r['encodings']) == {'3h'}                # only 3 h promoted
    assert set(t['encodings']['3h']) == {'clim_target', 'clim_origin'} and set(r['encodings']['3h']) == {'pclim', 'pcond'}
    for enc in (*t['encodings']['3h'].values(), *r['encodings']['3h'].values()):
        assert enc['min_count'] == tv.ENCODING_MIN_COUNT and enc['levels'][-1] == []        # fallback rule stored
    assert t['gate']['3h']['passed'] and r['gate']['3h']['probability_passed'] and r['gate']['3h']['amount_passed']
    assert not any(t['gate'][k]['passed'] for k in ('6h', '24h'))
    assert not any(r['gate'][k]['probability_passed'] for k in ('6h', '24h', '24h_total'))
    assert t['promotion']['keys'] == ['3h'] and r['promotion']['keys'] == ['3h']


@promoted
def test_one_hour_models_are_identical_to_the_pre_promotion_backup(fc):
    man = json.load(open(MANIFESTS[-1], encoding='utf-8'))
    by_name = {os.path.basename(e['original']): e['backup'] for e in man['files']}
    old_t = joblib.load(os.path.join(ROOT, by_name['temp_hourly_openmeteo.joblib']))
    old_r = joblib.load(os.path.join(ROOT, by_name['rain_twopart_openmeteo.joblib']))
    for key in ('1h', '6h', '24h'):
        assert np.array_equal(old_t['models'][key].coef_, fc.temp_hourly['models'][key].coef_)
        assert old_t['gate'][key]['passed'] == fc.temp_hourly['gate'][key]['passed']
    for key in ('1h', '6h', '24h', '24h_total'):
        assert np.array_equal(old_r['models'][key].coef_, fc.rain_twopart['models'][key].coef_)
        assert old_r['gate'][key]['probability_passed'] == fc.rain_twopart['gate'][key]['probability_passed']


@promoted
def test_live_inputs_reproduce_the_experiment_within_tolerance(fc):
    res = tv.verify_3h_live_compat(fc.temp_hourly, fc.rain_twopart)
    assert res['passed'] and res['tolerance'] == 1e-9
    assert res['temperature']['rows'] >= 100 and res['rainfall']['rows'] >= 100
    for v in ('temperature', 'rainfall'):
        assert res[v]['max_feature_diff'] <= 1e-9 and res[v]['max_prediction_diff'] <= 1e-9


@promoted
def test_live_forecast_rows_use_complete_ordered_features_and_keep_unsupported_horizons_unavailable(fc):
    origin = pd.Timestamp('2026-06-15 06:00', tz='UTC')
    weather = _weather(origin, rain_now=True)
    temps = {z: w['temperature_2m_c'].iloc[-48:] for z, w in weather.items()}
    precip = {z: w['precipitation_mm'] for z, w in weather.items()}
    t_out = fc.run_temperature_hourly(temps, origin, weather_by_zone=weather)
    r_out = fc.run_rainfall(precip, origin, weather_by_zone=weather)
    assert t_out['supported_horizons'] == ['1h', '3h'] and r_out['supported_horizons'] == ['1h', '3h']
    # inputs present, in the model's stored order, and the published value is that model's prediction
    t3, r3 = fc.temp_hourly['models']['3h'], fc.rain_twopart['models']['3h']
    for z in (1, 4, 9, 15):
        cell_t = fc.temp_hourly['zone_to_cell'][z]
        x = lf.temp_candidate_row(weather[fc.temp_hourly['rep_zone'][cell_t]], 3, cell_t, fc.temp_hourly['encodings']['3h'])
        assert list(x[t3.features].columns) == t3.features and not x[t3.features].isna().any(axis=None)
        row = {r['zone_id']: r for r in t_out['horizons']['3h']}[z]
        assert row['availability'] == 'MODEL_FORECAST' and row['temp_pred_c'] == round(float(t3.predict(x[t3.features])[0]), 2)
        cell_r = fc.rain_twopart['zone_to_cell'][z]
        xr = lf.rain_candidate_row(precip[z], weather[fc.rain_twopart['rep_zone'][cell_r]], 3, cell_r,
                                   fc.rain_twopart['cells'], fc.rain_twopart['encodings']['3h'])
        assert list(xr[r3.feature_names].columns) == r3.feature_names and not xr[r3.feature_names].isna().any(axis=None)
        rr = {r['zone_id']: r for r in r_out['horizons']['3h']}[z]
        assert rr['availability'] == 'MODEL_FORECAST' and 0 <= rr['prob_rain'] <= 1 and rr['mrf_pred'] >= 0
    for key in ('6h', '24h'):
        assert all(r['alert_status'] == RAIN_FAILED_EVAL and r['temp_pred_c'] is None for r in t_out['horizons'][key])
        assert all(r['alert_status'] == RAIN_FAILED_EVAL and r['prob_rain'] is None and r['accum_24h_mm'] is None
                   for r in r_out['horizons'][key])
    # missing weather for the promoted model -> unavailable, never filled
    no_w = fc.run_temperature_hourly(temps, origin, weather_by_zone={})
    assert no_w['supported_horizons'] == ['1h'] and all(r['temp_pred_c'] is None for r in no_w['horizons']['3h'])


def test_aqi_is_untouched_by_the_promotion(fc):
    assert not any('hmm_aqi' in p or 'hmm_mrf' in p for p in tv.PROMOTION_FILES)
    for m in MANIFESTS:
        assert not any('hmm_aqi' in e['original'] or 'hmm_mrf' in e['original']
                       for e in json.load(open(m, encoding='utf-8'))['files'])
    try:
        rc = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'models/hmm', 'models/hmm_mrf'], cwd=ROOT).returncode
    except FileNotFoundError:
        pytest.skip('git not available')
    assert rc == 0                                                   # committed AQI/MRF artifacts unchanged
    assert sorted(fc.aqi_hmm) == [2, 4, 5, 10, 12, 13, 14] and fc.mrf['aqi'].beta == 0.0001


def test_backup_and_rollback_restore_files_byte_for_byte(tmp_path):
    originals = {}
    for name, content in (('a.joblib', b'\x00old-bundle-bytes\xff'), ('b.csv', b'h,x\n3h,1\n')):
        p = tmp_path / name
        p.write_bytes(content)
        originals[str(p)] = content
    manifest = tv.backup_files(list(originals), root=str(tmp_path / 'backups'), label='test')
    for p in originals:                                              # "promotion" overwrites atomically
        tv._atomic_replace(lambda t: open(t, 'wb').write(b'new-content'), p)
    restored = tv.rollback_from_manifest(manifest)
    assert sorted(restored) == sorted(originals)
    for p, content in originals.items():
        assert open(p, 'rb').read() == content and not os.path.exists(p + '.tmp')
    man = json.load(open(manifest, encoding='utf-8'))
    with open(man['files'][0]['backup'], 'r+b') as f:                # corrupted backup is refused
        f.write(b'X')
    with pytest.raises(RuntimeError):
        tv.rollback_from_manifest(manifest)


@promoted
def test_real_promotion_backups_are_intact_and_restorable(tmp_path):
    man = json.load(open(MANIFESTS[-1], encoding='utf-8'))
    assert {os.path.normpath(e['original']) for e in man['files']} == {os.path.normpath(p) for p in tv.PROMOTION_FILES}
    sandbox = []
    for e in man['files']:
        assert tv._sha256(os.path.join(ROOT, e['backup'])) == e['sha256']             # backups intact
        target = tmp_path / os.path.basename(e['original'])
        target.write_bytes(b'promoted-version')                                      # stand-in for production
        sandbox.append({'original': str(target), 'backup': os.path.join(ROOT, e['backup']), 'sha256': e['sha256']})
    sb = tmp_path / 'manifest.json'
    sb.write_text(json.dumps({'files': sandbox}), encoding='utf-8')
    tv.rollback_from_manifest(str(sb))                               # real backups restore byte-for-byte
    for e in sandbox:
        assert tv._sha256(e['original']) == e['sha256']
