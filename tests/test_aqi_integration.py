"""
AQI data flow: WeatherAPI parsing (mocked), OpenAQ freshness diagnostics (fake client), CPCB scale, and the
real HMM+MRF path for 1 h / 3 h / 6 h / 24 h from fresh station input through the pipeline JSON to the dashboard.
No network: OpenAQ input is a temporary CSV built from the cached sensor metadata; outputs go to tmp_path.
"""

import io
import json
import os
import sys
import urllib.error

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collectors import live_collect
from src.preprocessing import live_inputs as li
from src.preprocessing.cpcb_aqi import calculate_sub_index

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'data', 'live', 'raw', 'openaq_recent.csv')
TYPICAL = {'PM2.5': 45.0, 'PM10': 90.0, 'NO2': 25.0, 'SO2': 10.0, 'CO': 0.8, 'Ozone': 30.0, 'NH3': 20.0}


# ── WeatherAPI client (legacy; not used by the v3 pipeline) ──────────────────
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _wa(monkeypatch, payload=None, error=None):
    from src.live import weatherapi_client as wa

    def fake(req, timeout=15):
        if error:
            raise error
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(wa.urllib.request, 'urlopen', fake)
    return wa.WeatherAPIClient(api_key='SECRET-KEY-123', request_delay_s=0)


def test_weatherapi_parses_air_quality_and_keeps_epa_index_as_metadata(monkeypatch):
    payload = {'location': {}, 'current': {'last_updated_epoch': 1791800000, 'temp_c': 30.1, 'precip_mm': 0.0,
                                           'humidity': 70, 'condition': {'text': 'Clear'},
                                           'air_quality': {'co': 400.5, 'no2': 12.0, 'o3': 60.0, 'so2': 5.0,
                                                           'pm2_5': 35.2, 'pm10': 70.1, 'us-epa-index': 2,
                                                           'gb-defra-index': 3}}}
    obs = _wa(monkeypatch, payload).fetch_zone(2)
    assert obs['fetch_ok'] and obs['air_quality_available'] and obs['pm2_5_ugm3'] == 35.2
    assert obs['weatherapi_us_epa_index'] == 2                         # ordinal 1-6 category, kept as metadata
    assert not any(k.lower() in ('aqi', 'cpcb_aqi', 'aqi_value') for k in obs)   # never presented as Indian AQI
    assert 'SECRET-KEY-123' not in json.dumps(obs, default=str)


def test_weatherapi_missing_air_quality_and_http_errors(monkeypatch):
    obs = _wa(monkeypatch, {'current': {'temp_c': 30.0}}).fetch_zone(3)
    assert obs['fetch_ok'] and obs['air_quality_available'] is False and obs['pm2_5_ugm3'] is None
    err = urllib.error.HTTPError('https://x?key=SECRET-KEY-123', 401, 'Unauthorized', {},
                                 io.BytesIO(b'{"error":{"code":2006,"message":"API key SECRET-KEY-123 is invalid."}}'))
    bad = _wa(monkeypatch, error=err).fetch_zone(3)
    assert bad['fetch_ok'] is False and bad['error_msg'].startswith('HTTP 401') and 'SECRET-KEY-123' not in bad['error_msg']


def test_weatherapi_requires_key(monkeypatch):
    from src.live import weatherapi_client as wa
    monkeypatch.delenv('WEATHERAPI_KEY', raising=False)
    with pytest.raises(PermissionError):
        wa.WeatherAPIClient(api_key=None)


# ── OpenAQ freshness diagnostics ─────────────────────────────────────────────
class FakeOpenAQ:
    def __init__(self, last_utc, record_hours):
        self.last, self.hours, self.request_log = last_utc, record_hours, []

    def list_sensors(self, location_id):
        self.request_log.append(('sensors', location_id))
        return [{'id': location_id * 10 + 1, 'parameter': {'name': 'pm25', 'units': 'µg/m³'},
                 'datetimeLast': {'utc': self.last}}]

    def sensor_hours(self, sensor_id, dfrom, dto):
        self.request_log.append(('hours', sensor_id))
        return [{'period': {'datetimeFrom': {'utc': t}, 'datetimeTo': {'utc': t}}, 'value': 40.0,
                 'parameter': {'name': 'pm25', 'units': 'µg/m³'}, 'coverage': {'observedCount': 4}} for t in self.hours]


def test_openaq_status_records_upstream_datetime_last(monkeypatch):
    monkeypatch.setattr(live_collect, 'load_validated_sensors', lambda: None)
    old = '2026-10-07T14:30:00Z'
    df, status = live_collect.collect_openaq(FakeOpenAQ(old, ['2026-10-07T13:30:00Z']),
                                             now_utc=pd.Timestamp('2026-10-12 02:00', tz='UTC'))
    assert status['ok'] and status['latest_record_utc'].startswith('2026-10-07T13:30')
    assert pd.Timestamp(status['latest_sensor_datetime_last_utc']) == pd.Timestamp(old)
    assert all(s['datetime_last_utc'] for st in status['stations'] for s in st['sensors'].values())


def test_stale_or_short_station_records_give_no_current_aqi():
    from src import freshness as fr
    rows = _synthetic_openaq(pd.Timestamp('2026-10-07 13:30', tz='UTC'), hours=13)
    st = li.build_aqi_inputs(rows, pd.Timestamp('2026-10-07 14:00', tz='UTC'))
    assert all(s['latest_valid_utc'] is None for s in st.values())     # < 16 valid hours: no 24 h CPCB mean
    rows = _synthetic_openaq(pd.Timestamp('2026-10-07 13:30', tz='UTC'), hours=30)
    st = li.build_aqi_inputs(rows, pd.Timestamp('2026-10-07 14:00', tz='UTC'))
    t = next(s['latest_valid_utc'] for s in st.values() if s['role'] == 'primary')
    assert t is not None and fr.is_observation_current(t, pd.Timestamp('2026-10-07 14:00', tz='UTC'))
    assert not fr.is_observation_current(t, pd.Timestamp('2026-10-12 02:00', tz='UTC'))   # 3 h rule -> stale


def test_cpcb_scale_breakpoints():
    assert calculate_sub_index('PM2.5', 30.0) == pytest.approx(50.0)
    assert calculate_sub_index('PM2.5', 60.0) == pytest.approx(100.0)
    assert calculate_sub_index('PM10', 100.0) == pytest.approx(100.0)
    assert calculate_sub_index('PM2.5', 90.0) == pytest.approx(200.0)


# ── real HMM + MRF path, four horizons, pipeline JSON, dashboard ─────────────
def _synthetic_openaq(end_utc, hours=48):
    """OpenAQ-format hourly rows for the cached (unit-validated) sensors, ending at end_utc (hour starts :30)."""
    meta = pd.read_csv(CACHE)[['location_id', 'station_name', 'zone_id', 'station_role', 'sensor_id',
                               'cpcb_pollutant', 'units', 'parameter']].drop_duplicates()
    times = pd.date_range(end=end_utc, periods=hours, freq='h')
    k = np.arange(hours)
    rows = []
    for _, m in meta.iterrows():
        base = TYPICAL[m['cpcb_pollutant']] * (1 + 0.05 * (int(m['zone_id']) % 4))
        vals = base * (1 + 0.15 * np.sin(2 * np.pi * k / 24))
        for t, v in zip(times, vals):
            rows.append({**m.to_dict(), 'hour_start_utc': t.strftime('%Y-%m-%dT%H:%M:%SZ'),
                         'hour_end_utc': (t + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                         'value': float(v), 'observed_count': 4, 'expected_count': 4, 'percent_coverage': 100.0,
                         'has_flags': False})
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def replay(tmp_path_factory):
    if not os.path.exists(CACHE):
        pytest.skip('no cached OpenAQ sensor metadata')
    from src.live import live_pipeline as lp
    tmp = tmp_path_factory.mktemp('aqi')
    now = pd.Timestamp('2026-10-10 06:10', tz='UTC')
    oaq = tmp / 'openaq.csv'
    _synthetic_openaq(pd.Timestamp('2026-10-10 05:30', tz='UTC')).to_csv(oaq, index=False)
    mp = pytest.MonkeyPatch()
    mp.setattr(live_collect, 'OPENAQ_FILE', str(oaq))
    out = tmp / 'pred.json'
    try:
        lp.run_once(fetch=False, now_utc=now, output_path=str(out), verbose=False)
    finally:
        mp.undo()
    return json.load(open(out, encoding='utf-8')), now, str(out)


def test_four_horizons_are_generated_independently(replay):
    d, now, _ = replay
    assert d['meta']['supported_horizons']['AQI'] == ['1h', '3h', '6h', '24h']
    origin = pd.Timestamp(d['meta']['data_times']['aqi_forecast_origin_utc'])
    vals = {}
    for h in (1, 3, 6, 24):
        rows = {r['zone_id']: r for r in d['predictions']['AQI'][f'{h}h']}
        r2 = rows[2]
        assert r2['availability'] == 'MODEL_FORECAST' and r2['basis'].startswith('HMM+MRF')
        assert pd.Timestamp(r2['target_time_utc']) == origin + pd.Timedelta(hours=h)
        assert 0 <= r2['mrf_pred'] <= 500 and r2['category']
        vals[h] = r2['mrf_pred']
        assert rows[1]['availability'] == 'MODEL_ESTIMATE'                 # no station: MRF from neighbour, labelled
        assert rows[15]['availability'] in ('UNAVAILABLE', 'MODEL_ESTIMATE')
    assert len(set(vals.values())) > 1                                       # horizons are not copies of each other
    zm = {z['zone_id']: z for z in d['zone_meta']}
    assert zm[2]['aqi_state'] == 'MEASURED' and 0 <= zm[2]['cpcb_aqi_computed'] <= 500
    assert 'epa' not in json.dumps(d).lower()                                # no US-EPA index anywhere


def test_dashboard_selects_each_horizon_record(replay):
    from dashboard import data_loader as dl
    d, now, _ = replay
    seen = set()
    for hk in ('1h', '3h', '6h', '24h'):
        fv = dl.forecast_view(d, 2, 'AQI', hk, now)
        assert fv['target_time_utc'] == {r['zone_id']: r for r in d['predictions']['AQI'][hk]}[2]['target_time_utc']
        seen.add(fv['target_time_utc'])
    assert len(seen) == 4


def test_missing_station_input_leaves_every_horizon_unavailable():
    from src.forecasting.live_forecaster import LiveForecaster
    out = LiveForecaster().run_aqi({})
    assert out['origin_utc'] is None and not out.get('horizons')


def test_dashboard_renders_with_aqi_forecasts(replay, monkeypatch):
    apptest = pytest.importorskip('streamlit.testing.v1')
    _, _, path = replay
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv('URBANPULSE_PREDICTIONS_PATH', path)
    at = apptest.AppTest.from_file(os.path.join(ROOT, 'dashboard', 'app.py'), default_timeout=120)
    at.session_state['selected_zone'] = 2
    at.session_state['zone_dropdown'] = 2
    at.run()
    assert [e.message for e in at.exception] == []
    md = ' '.join(m.value for m in at.markdown)
    assert 'All Forecast Horizons' in md
