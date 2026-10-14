"""
Tests for the final OpenAQ + Open-Meteo pipeline: data validation, zone mapping,
missing-data handling, API-failure handling and prediction-output validation.
All network access is mocked.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collectors import open_meteo, live_collect
from src.collectors.openaq_client import OpenAQClient, OpenAQError
from src.preprocessing.cpcb_aqi import aqi_from_hourly_concentrations
from src.preprocessing.live_inputs import station_hourly_concentrations, build_aqi_inputs
from src.forecasting.live_forecaster import LiveForecaster
from src.alerts.alert_engine import evaluate_aqi_alert, evaluate_rainfall_alert_24h, UNAVAILABLE_NO_DATA
from src.audit.openaq_audit import load_zone_polygons, zone_for_point
from dashboard import data_loader as dl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ── fixtures / helpers ──────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, status, payload=None, headers=None, text=''):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, text

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def openaq_rows(location_id, zone_id, role, start, hours, values, pollutant, sensor_id, observed=4):
    t = pd.date_range(start, periods=hours, freq='h', tz='UTC') + pd.Timedelta(minutes=30)
    vals = values if hasattr(values, '__len__') else [values] * hours
    return pd.DataFrame({'hour_start_utc': t.strftime('%Y-%m-%dT%H:%M:%SZ'), 'value': vals,
                         'observed_count': observed, 'location_id': location_id, 'zone_id': zone_id,
                         'station_name': f'st{location_id}', 'station_role': role,
                         'sensor_id': sensor_id, 'cpcb_pollutant': pollutant})


def station_frame(location_id, zone_id, start, hours, pm25=40.0, pm10=80.0, o3=30.0, role='primary', sid0=0):
    return pd.concat([openaq_rows(location_id, zone_id, role, start, hours, pm25, 'PM2.5', sid0 + 1),
                      openaq_rows(location_id, zone_id, role, start, hours, pm10, 'PM10', sid0 + 2),
                      openaq_rows(location_id, zone_id, role, start, hours, o3, 'Ozone', sid0 + 3)],
                     ignore_index=True)


@pytest.fixture(scope='module')
def forecaster():
    return LiveForecaster()


# ── data validation ─────────────────────────────────────────────────────────
def test_open_meteo_format_detected_from_content_not_extension(tmp_path):
    js = tmp_path / 'weather.csv'          # JSON content behind a .csv extension
    js.write_text(json.dumps([{'latitude': 13.18, 'longitude': 80.24, 'elevation': 3, 'utc_offset_seconds': 0,
                               'timezone': 'GMT', 'hourly_units': {'temperature_2m': '°C'},
                               'hourly': {'time': ['2025-01-01T00:00'], 'temperature_2m': [25.0],
                                          'precipitation': [0.0]}}]), encoding='utf-8')
    assert open_meteo.detect_format(str(js)) == 'json'
    d = open_meteo.load_local(str(js))
    assert d['format'] == 'json' and d['hourly']['temperature_2m_c'].iloc[0] == 25.0
    assert (d['hourly']['provenance'] == 'reanalysis').all()


def test_local_open_meteo_file_is_two_block_csv_and_complete():
    d = open_meteo.load_local()
    assert d['format'] == 'csv'
    h = d['hourly']
    assert h['location_id'].nunique() == 15
    assert not h.duplicated(['location_id', 'time_utc']).any()
    assert str(h['time_utc'].dt.tz) == 'UTC'
    assert (h['provenance'] == 'reanalysis').all()      # never labelled as measured


def test_daily_mean_requires_complete_days():
    t = pd.date_range('2025-01-01 18:30', periods=30, freq='h', tz='UTC')   # IST midnight start
    h = pd.DataFrame({'zone_id': 1, 'time_utc': t, 'temperature_2m_c': 30.0})
    d = open_meteo.daily_mean_temperature(h)
    assert d['tmean_c'].notna().sum() == 1          # first IST day complete (24 h)
    assert d['tmean_c'].isna().sum() == 1           # partial second day stays NaN, not filled


def test_shared_aqi_function_matches_training_methodology():
    df = pd.read_csv(os.path.join(ROOT, 'data/processed/clean_aqi_stations/perungudi-2024-25.csv'),
                     index_col=0, parse_dates=True)
    pol = ['PM2.5', 'PM10', 'NO2', 'SO2', 'NH3', 'CO', 'Ozone']
    h = df[pol].resample('1h').agg(lambda s: s.mean() if s.count() >= 2 else np.nan).iloc[:600]
    a = aqi_from_hourly_concentrations(h)['AQI'].values
    m = pd.read_csv(os.path.join(ROOT, 'data/processed/master_table.csv'), usecols=['zone_id', 'AQI'])
    m = m[m['zone_id'] == 14]['AQI'].values[:600]
    assert np.array_equal(np.isnan(a), np.isnan(m))
    assert np.nanmax(np.abs(a - m)) < 1e-6


# ── zone mapping ────────────────────────────────────────────────────────────
def test_open_meteo_locations_map_to_15_distinct_zones():
    zones = [z for z, _ in open_meteo.LOCATION_TO_ZONE.values()]
    assert sorted(zones) == list(range(1, 16))


def test_zone_request_points_belong_to_their_zone():
    # Zone 12 (Alandur) is concave: its centroid lies 0.2 km outside the polygon, in a gap that
    # belongs to no other zone. Accept "inside own zone" or "<= 0.5 km from it, inside no other zone".
    polys = load_zone_polygons()
    for z, (lat, lon) in open_meteo.ZONE_CENTROIDS.items():
        r = zone_for_point(lat, lon, polys)
        assert r['pip_zone_id'] == z or (r['pip_zone_id'] is None and r['nearest_zone_id'] == z
                                         and r['distance_to_nearest_zone_km'] <= 0.5)


def test_live_station_registry_matches_station_zone_mapping():
    mp = pd.read_csv(os.path.join(ROOT, 'data/processed/station_zone_mapping.csv'))
    by_name = dict(zip(mp['station_name'], mp['zone_id']))
    for st in live_collect.STATIONS:
        assert by_name[st['station_name']] == st['zone_id']
    primaries = [s['zone_id'] for s in live_collect.STATIONS if s['role'] == 'primary']
    assert len(primaries) == len(set(primaries))       # one primary station per zone


def test_sensor_selection_respects_unit_validation():
    sensors = [
        {'id': 1, 'parameter': {'name': 'no2', 'units': 'µg/m³'}, 'datetimeLast': {'utc': '2026-10-07T00:00:00Z'}},
        {'id': 2, 'parameter': {'name': 'no2', 'units': 'ppb'}, 'datetimeLast': {'utc': '2026-10-07T00:00:00Z'}},
        {'id': 3, 'parameter': {'name': 'wind_speed', 'units': 'm/s'}, 'datetimeLast': {'utc': '2026-10-07T00:00:00Z'}},
    ]
    assert live_collect.select_sensors(sensors)['NO2']['sensor_id'] == 1
    assert live_collect.select_sensors(sensors, validated={2})['NO2']['sensor_id'] == 2
    assert 'NO2' not in live_collect.select_sensors(sensors, validated={99})


# ── missing data ────────────────────────────────────────────────────────────
def test_invalid_hours_are_missing_not_zero():
    idx = pd.date_range('2026-10-01 00:30', periods=3, freq='h', tz='UTC')
    rec = pd.DataFrame({'hour_start_utc': idx.strftime('%Y-%m-%dT%H:%M:%SZ'), 'value': [10.0, -5.0, 12.0],
                        'observed_count': [4, 4, 1], 'sensor_id': 7, 'cpcb_pollutant': 'PM2.5'})
    out = station_hourly_concentrations(rec, idx, validated={7})
    assert out['PM2.5'].iloc[0] == 10.0
    assert np.isnan(out['PM2.5'].iloc[1])           # negative -> invalid
    assert np.isnan(out['PM2.5'].iloc[2])           # 1 of 4 sub-hourly values -> invalid
    assert station_hourly_concentrations(rec, idx, validated={8}).empty or \
        'PM2.5' not in station_hourly_concentrations(rec, idx, validated={8})


def test_aqi_requires_three_pollutants_including_pm():
    st = openaq_rows(1, 2, 'primary', '2026-10-01', 48, 40.0, 'PM2.5', 1)    # PM2.5 only
    res = build_aqi_inputs(st, pd.Timestamp('2026-10-03', tz='UTC'), validated={1})
    assert res[1]['latest_aqi'] is None and res[1]['latest_category'] == 'Unavailable'


def test_missing_forecast_is_not_reported_as_normal():
    assert evaluate_aqi_alert(np.nan, np.nan)['alert_status'] == UNAVAILABLE_NO_DATA
    assert evaluate_rainfall_alert_24h(np.nan, np.nan, is_monitored=True)['alert_status'] == UNAVAILABLE_NO_DATA
    assert dl.get_risk_level('SOMETHING_UNKNOWN') == 'Unavailable'
    empty = {'predictions': {'AQI': {}, 'rainfall': {}}}
    assert dl.get_dominant_risk_for_zone(empty, 3) == 'Unavailable'


# ── API failure handling ────────────────────────────────────────────────────
def test_openaq_auth_failure_raises_without_exposing_key():
    secret = 'SECRET-KEY-123'
    c = OpenAQClient(api_key=secret, session=FakeSession([FakeResp(401, text='unauthorized')]), max_retries=2)
    with pytest.raises(OpenAQError) as e:
        c.list_sensors(1)
    assert secret not in str(e.value)
    assert c.request_log[-1]['error'] and secret not in json.dumps(c.request_log)


def test_openaq_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda s: None)
    sess = FakeSession([FakeResp(429, headers={'X-Ratelimit-Reset': '1'}),
                        FakeResp(200, {'results': [{'id': 5}]}, headers={'X-Ratelimit-Remaining': '50'})])
    c = OpenAQClient(api_key='k', session=sess)
    assert c.list_sensors(1) == [{'id': 5}]
    assert len(sess.calls) == 2


def test_openaq_network_error_retried_then_fails(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda s: None)
    sess = FakeSession([requests.ConnectionError('down')] * 3)
    c = OpenAQClient(api_key='k', session=sess, max_retries=3)
    with pytest.raises(OpenAQError):
        c.sensor_hours(1, '2026-10-01T00:00:00Z', '2026-10-02T00:00:00Z')


def test_collection_failure_keeps_previous_data(tmp_path, monkeypatch):
    monkeypatch.setattr(live_collect, 'RAW_DIR', str(tmp_path))
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(tmp_path / 'openaq_recent.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(tmp_path / 'open_meteo_recent.csv'))
    monkeypatch.setattr(live_collect, 'STATUS_FILE', str(tmp_path / 'status.json'))
    (tmp_path / 'openaq_recent.csv').write_text('previous,data\n1,2\n')
    (tmp_path / 'status.json').write_text(json.dumps({'openaq': {'last_success_utc': '2026-10-08T00:00:00+00:00'}}))
    monkeypatch.setattr(live_collect, 'collect_openaq',
                        lambda client, now: (pd.DataFrame(), {'ok': False, 'error': 'HTTP 503'}))

    def om_fail(now):
        return pd.DataFrame(), {'ok': False, 'error': 'ConnectionError: down'}
    monkeypatch.setattr(live_collect, 'collect_open_meteo', om_fail)
    st = live_collect.run_collection(now_utc=pd.Timestamp('2026-10-09T08:00', tz='UTC'), verbose=False)
    assert (tmp_path / 'openaq_recent.csv').read_text() == 'previous,data\n1,2\n'
    assert st['openaq']['ok'] is False and st['openaq']['last_success_utc'] == '2026-10-08T00:00:00+00:00'
    assert st['open_meteo']['ok'] is False and st['open_meteo']['last_success_utc'] is None


def test_open_meteo_rejects_unexpected_units(monkeypatch):
    payload = [{'latitude': 13, 'longitude': 80, 'hourly_units': {'temperature_2m': '°F'},
                'hourly': {'time': ['2026-10-01T00:00'], 'temperature_2m': [80.0]}}] * 15
    sess = FakeSession([FakeResp(200, payload)])
    sess.responses[0].raise_for_status = lambda: None
    with pytest.raises(ValueError):
        open_meteo.fetch_archive('2026-10-01', '2026-10-01', session=sess)


# ── prediction output validation ────────────────────────────────────────────
def test_no_measured_input_gives_no_aqi_forecast(forecaster):
    out = forecaster.run_aqi({})
    assert out['origin_utc'] is None and not out['zones']


def test_aqi_forecast_availability_rules(forecaster):
    end = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    frames = [station_frame(2586, 2, '2026-10-01', 96, sid0=0),          # Manali: current
              station_frame(12046, 14, '2026-10-02', 36, sid0=10)]       # Perungudi: last data 36 h before origin
    stations = build_aqi_inputs(pd.concat(frames), end, validated=set(range(1, 20)))
    out = forecaster.run_aqi(stations)
    by_zone = {r['zone_id']: r for r in out['horizons']['1h']}
    assert by_zone[2]['availability'] == 'MODEL_FORECAST' and by_zone[2]['mrf_pred'] is not None
    assert by_zone[14]['alert_status'] == 'UNAVAILABLE_STALE_INPUT' and by_zone[14]['mrf_pred'] is None
    # zone 3 (Madhavaram) borders Manali -> estimate; zone 15 does not border zone 2 -> unavailable
    assert by_zone[3]['availability'] == 'MODEL_ESTIMATE'
    assert by_zone[15]['availability'] == 'UNAVAILABLE' and by_zone[15]['exceedance_prob'] is None
    for h, rows in out['horizons'].items():
        assert len(rows) == 15
        for r in rows:
            if r['mrf_pred'] is not None:
                assert 0 <= r['exceedance_prob'] <= 1 and r['mrf_var'] > 0


def test_temperature_output_is_daily_mean_and_heatwave_not_evaluated(forecaster):
    days = pd.date_range('2026-09-01', periods=35, freq='D').date
    daily = pd.DataFrame([{'zone_id': z, 'date_ist': d, 'tmean_c': 29.0} for z in range(1, 16) for d in days])
    rows = forecaster.run_temperature(daily)['horizons']['24h']
    assert len(rows) == 15
    for r in rows:
        assert 'next_day_tmax_pred_c' not in r
        assert 20 < r['next_day_tmean_pred_c'] < 40
        assert r['alert_status'] == 'NOT_EVALUATED_MODEL_IS_DAILY_MEAN' and r['exceedance_prob'] is None


def test_rainfall_only_for_trained_zones():
    gauge_fc = LiveForecaster()
    gauge_fc.rain_om = gauge_fc.rain_twopart = None   # exercise the gauge-trained path explicitly
    origin = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    idx = pd.date_range(origin - pd.Timedelta(hours=71), origin, freq='h')
    precip = {z: pd.Series(0.0, index=idx) for z in (4, 5, 10, 14)}
    out = gauge_fc.run_rainfall(precip, origin)
    rows24 = {r['zone_id']: r for r in out['horizons']['24h']}
    assert rows24[4]['accum_24h_mm'] is not None and rows24[4]['alert_status'] in ('NORMAL', 'WATCH', 'WARNING')
    assert rows24[1]['alert_status'] == 'NOT_SUPPORTED_NO_MODEL' and rows24[1]['mrf_pred'] is None
    assert all(r['alert_status'] == 'NOT_APPLICABLE_HOURLY' for r in out['horizons']['1h'] if r['zone_id'] == 4)


def test_end_to_end_output_is_strict_json(tmp_path, monkeypatch):
    from src.live import live_pipeline
    raw = tmp_path / 'raw'
    raw.mkdir()
    now = pd.Timestamp('2026-10-05 01:00', tz='UTC')
    station_frame(2586, 2, '2026-10-01', 96).to_csv(raw / 'openaq.csv', index=False)
    t = pd.date_range(pd.Timestamp('2026-09-01', tz='UTC'), now, freq='h')
    om = pd.DataFrame([{'zone_id': z, 'time_utc': x, 'temperature_2m_c': 29.0, 'precipitation_mm': 0.0,
                        'relative_humidity_2m_pct': 70, 'wind_speed_10m_kmh': 10, 'wind_direction_10m_deg': 90,
                        'grid_latitude': 13.0, 'grid_longitude': 80.2 + (z % 8) / 100, 'source': 'test'}
                       for z in range(1, 16) for x in t])
    om.to_csv(raw / 'om.csv', index=False)
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(raw / 'openaq.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(raw / 'om.csv'))
    monkeypatch.setattr(live_collect, 'load_status', lambda: {
        'openaq': {'ok': True, 'latest_record_utc': '2026-10-04T23:30:00+00:00'},
        'open_meteo': {'ok': True, 'latest_record_utc': now.isoformat()}})
    monkeypatch.setattr(live_pipeline, 'load_validated_sensors', lambda: {1, 2, 3})
    out_path = tmp_path / 'pred.json'
    live_pipeline.run_once(fetch=False, now_utc=now, output_path=str(out_path), verbose=False)
    text = out_path.read_text(encoding='utf-8')
    assert 'NaN' not in text
    d = json.loads(text)
    assert len(d['zone_meta']) == 15
    fc_t = LiveForecaster()
    expected_t = ([k for k, g in fc_t.temp_hourly['gate'].items() if g['passed']]   # hourly model, passing horizons
                  if fc_t.temp_hourly is not None else ['24h'])                     # else the daily-mean HMM
    assert d['meta']['supported_horizons']['temperature'] == expected_t
    zm = {z['zone_id']: z for z in d['zone_meta']}
    assert zm[2]['aqi_state'] == 'MEASURED' and zm[2]['cpcb_aqi_computed'] is not None
    assert zm[3]['cpcb_aqi_computed'] is None                     # no station -> no observed value
    assert zm[3]['weather_provenance'].startswith('Open-Meteo reanalysis')
    for a in d['alerts']:
        assert a['severity'] in ('HIGH', 'MODERATE', 'LOW', 'UNAVAILABLE', 'UNMONITORED')


def test_rainfall_disabled_policy_is_explicit(forecaster):
    from src.forecasting.live_forecaster import RAINFALL_FORECAST_ENABLED
    assert RAINFALL_FORECAST_ENABLED is False            # documented decision (final_data_audit.md)
    out = forecaster.run_rainfall({}, None, enabled=False)
    assert out['n_active'] == 0 and out['enabled'] is False
    for rows in out['horizons'].values():
        assert len(rows) == 15
        for r in rows:
            assert r['mrf_pred'] is None and r['exceedance_prob'] is None
            assert r['alert_status'] in ('DISABLED_INPUT_NOT_VALIDATED', 'NOT_SUPPORTED_NO_MODEL')
            assert dl.get_risk_level(r['alert_status']) == 'Unavailable'


def test_aqi_not_extended_past_last_measurement():
    st = station_frame(2586, 2, '2026-10-01', 48)                 # data ends 2026-10-02 23:30 UTC
    res = build_aqi_inputs(st, pd.Timestamp('2026-10-03 12:00', tz='UTC'), validated={1, 2, 3})
    assert res[2586]['latest_valid_utc'] == pd.Timestamp('2026-10-02 23:30', tz='UTC')
    assert res[2586]['aqi'].loc[pd.Timestamp('2026-10-03 03:30', tz='UTC')] != res[2586]['aqi'].loc[
        pd.Timestamp('2026-10-03 03:30', tz='UTC')]                # NaN, although the 24 h window still has data
