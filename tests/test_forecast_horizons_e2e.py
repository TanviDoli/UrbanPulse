"""
Rainfall/temperature horizons end to end (offline replay of the cached Open-Meteo data): the forecaster produces
1/3/6/24 h, the pipeline serialises them, the dashboard loader selects each horizon with correct target times,
freshness is judged at view time (EXPIRED once the target has passed), the optional 6/24 h packages are off by
default with an explicit reason, and no synthetic values appear. Writes only to tmp_path.
"""

import hashlib
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import freshness as fr
from src.forecasting import live_forecaster as lf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OM = os.path.join(ROOT, 'data', 'live', 'raw', 'open_meteo_recent.csv')
PROTECTED = [os.path.join(ROOT, 'data', 'live', f) for f in ('current_predictions.json', 'demo_predictions.json')] + \
    [os.path.join(ROOT, 'data', 'live', 'raw', f) for f in ('open_meteo_recent.csv', 'openaq_recent.csv', 'collection_status.json')]
need_cache = pytest.mark.skipif(not os.path.exists(OM), reason='no cached Open-Meteo data')


def _sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _origin():
    om = pd.read_csv(OM, usecols=['zone_id', 'time_utc', 'precipitation_mm'])
    om['time_utc'] = pd.to_datetime(om['time_utc'], utc=True)
    n = om.dropna().groupby('time_utc')['zone_id'].nunique()
    return n[n == 15].index.max()


def _run(tmp_path, monkeypatch, enable: bool):
    from src.live import live_pipeline as lp
    for env in (lf.TEMP_HISTORY_ENV, lf.RAIN_HISTORY_ENV):
        if enable:
            monkeypatch.setenv(env, '1')
        else:
            monkeypatch.delenv(env, raising=False)
    origin = _origin()
    out = tmp_path / ('on.json' if enable else 'off.json')
    lp.run_once(fetch=False, now_utc=origin + pd.Timedelta(minutes=30), output_path=str(out), verbose=False)
    return json.load(open(out, encoding='utf-8')), origin


@need_cache
def test_all_four_horizons_survive_pipeline_and_dashboard_selection(tmp_path, monkeypatch):
    from dashboard import data_loader as dl
    before = {p: _sha(p) for p in PROTECTED if os.path.exists(p)}
    d, origin = _run(tmp_path, monkeypatch, enable=True)
    view_now = origin + pd.Timedelta(minutes=30)
    for var, field in (('rainfall', 'prob_rain'), ('temperature', 'temp_pred_c')):
        assert d['meta']['supported_horizons'][var] == ['1h', '3h', '6h', '24h']
        for h in (1, 3, 6, 24):
            for z in range(1, 16):
                fv = dl.forecast_view(d, z, var, f'{h}h', view_now)
                assert fv['availability'] == 'MODEL_FORECAST' and fv[field] is not None
                t = pd.Timestamp(fv['target_time_utc'])
                assert t == origin + pd.Timedelta(hours=h) and fv['forecast_freshness'] == fr.CURRENT
                assert t.tz_convert('Asia/Kolkata').utcoffset() == pd.Timedelta(hours=5, minutes=30)
            later = dl.forecast_view(d, 1, var, f'{h}h', origin + pd.Timedelta(hours=h, minutes=1))
            assert later['forecast_freshness'] == fr.EXPIRED                # aged at view time, value kept
    tot = {r['zone_id']: r for r in d['predictions']['rainfall']['24h']}
    assert all(r['accum_24h_mm'] is not None and r['accum_quantity'] == 'total_next_24h_mm' for r in tot.values())
    assert 'synthetic' not in json.dumps(d).lower()                         # live output never synthetic
    assert {p: _sha(p) for p in before} == before                           # protected files untouched


@need_cache
def test_without_the_optional_packages_6h_24h_are_explicitly_unavailable(tmp_path, monkeypatch):
    d, _ = _run(tmp_path, monkeypatch, enable=False)
    for var in ('rainfall', 'temperature'):
        assert d['meta']['supported_horizons'][var] == ['1h', '3h']
        for hk in ('6h', '24h'):
            rows = d['predictions'][var][hk]
            assert len(rows) == 15 and all(r['availability'] == 'UNAVAILABLE' for r in rows)
            assert all(r['alert_status'] == lf.RAIN_FAILED_EVAL and r.get('reason') for r in rows)
    assert d['predictions']['AQI'] == {}                                    # AQI path unchanged (no fresh station data)
