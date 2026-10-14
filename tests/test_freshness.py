"""
Freshness, forecast-time and alert-status rules (src/freshness.py) as applied by the live
pipeline and the dashboard data loader. No network, no writes outside tmp_path; the committed
data/live/current_predictions.json is only read.
"""

import hashlib
import importlib
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import freshness as fr
from src.alerts.alert_backend import build_alert_records
from src.collectors import live_collect
from dashboard import data_loader as dl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LIVE_JSON = os.path.join(ROOT, 'data', 'live', 'current_predictions.json')
T0 = pd.Timestamp('2026-10-07 13:30', tz='UTC')


# ── R1 observation freshness ─────────────────────────────────────────────────
def test_observation_current_up_to_max_age_then_stale():
    assert fr.AQI_OBS_MAX_AGE_HOURS == 3.0
    assert fr.is_observation_current(T0, T0 + pd.Timedelta(hours=3))
    assert not fr.is_observation_current(T0, T0 + pd.Timedelta(hours=3, minutes=1))
    assert not fr.is_observation_current(T0, T0 + pd.Timedelta(hours=43))
    assert not fr.is_observation_current(None, T0)              # missing time is never current
    assert fr.age_hours('2026-10-07T13:30:00+00:00', T0 + pd.Timedelta(hours=43.1)) == pytest.approx(43.1)


def test_max_age_is_configurable_via_environment(monkeypatch):
    monkeypatch.setenv('URBANPULSE_AQI_MAX_AGE_HOURS', '6')
    try:
        importlib.reload(fr)
        assert fr.AQI_OBS_MAX_AGE_HOURS == 6.0
        assert fr.is_observation_current(T0, T0 + pd.Timedelta(hours=5))
        monkeypatch.setenv('URBANPULSE_AQI_MAX_AGE_HOURS', 'not-a-number')
        importlib.reload(fr)
        assert fr.AQI_OBS_MAX_AGE_HOURS == 3.0                  # invalid value falls back to default
    finally:
        monkeypatch.delenv('URBANPULSE_AQI_MAX_AGE_HOURS', raising=False)
        importlib.reload(fr)


# ── R2 forecast timing ───────────────────────────────────────────────────────
def test_forecast_freshness_states():
    tgt = T0 + pd.Timedelta(hours=24)
    assert fr.forecast_freshness(tgt, T0, T0 + pd.Timedelta(hours=1)) == fr.CURRENT
    assert fr.forecast_freshness(tgt, T0, T0 + pd.Timedelta(hours=5)) == fr.STALE_INPUT
    assert fr.forecast_freshness(tgt, T0, tgt) == fr.EXPIRED     # target reached = no longer upcoming
    assert fr.forecast_freshness(tgt, T0, tgt + pd.Timedelta(hours=19)) == fr.EXPIRED
    assert fr.forecast_freshness(None, T0, T0) == fr.UNKNOWN
    assert fr.forecast_freshness(tgt, None, T0) == fr.UNKNOWN


# ── R3 display risk ──────────────────────────────────────────────────────────
@pytest.mark.parametrize('status,availability,freshness,level,evaluated,stale_warning', [
    ('NORMAL', 'MODEL_FORECAST', fr.CURRENT, 'Low', True, False),
    ('WARNING', 'MODEL_FORECAST', fr.CURRENT, 'High', True, False),
    ('NORMAL', 'MODEL_ESTIMATE', fr.CURRENT, 'Estimate', True, False),
    ('WATCH', 'MODEL_ESTIMATE', fr.CURRENT, 'Moderate', True, False),
    ('NORMAL', 'MODEL_FORECAST', fr.EXPIRED, 'Stale', False, False),
    ('NORMAL', 'MODEL_FORECAST', fr.STALE_INPUT, 'Stale', False, False),
    ('WARNING', 'MODEL_FORECAST', fr.EXPIRED, 'Stale warning', False, True),
    ('WATCH', 'MODEL_ESTIMATE', fr.STALE_INPUT, 'Stale warning', False, True),
    ('NORMAL', 'MODEL_FORECAST', fr.UNKNOWN, 'Stale', False, False),
    ('UNAVAILABLE_NO_DATA', 'UNAVAILABLE', fr.CURRENT, 'Unavailable', False, False),
    ('NOT_SUPPORTED_NO_MODEL', 'UNAVAILABLE', fr.UNKNOWN, 'Unavailable', False, False),
])
def test_display_risk_never_green_without_current_measured_input(status, availability, freshness,
                                                                  level, evaluated, stale_warning):
    r = fr.display_risk(status, availability, freshness)
    assert (r['level'], r['evaluated'], r['stale_warning']) == (level, evaluated, stale_warning)
    if r['level'] == 'Low':
        assert availability == 'MODEL_FORECAST' and freshness == fr.CURRENT


def test_estimate_input_time_is_origin_forecast_uses_zone_measurement():
    zin = {'2': {'last_measured_utc': '2026-10-07T10:30:00+00:00'}}
    assert fr.aqi_forecast_input_time(2, 'MODEL_FORECAST', zin, T0) == '2026-10-07T10:30:00+00:00'
    assert fr.aqi_forecast_input_time(3, 'MODEL_ESTIMATE', zin, T0) == T0


# ── R4 pipeline output ───────────────────────────────────────────────────────
def test_pipeline_status_overdue_is_not_zero_minutes():
    meta = {'generated_at_utc': '2026-10-09T08:33:05+00:00', 'next_update_utc': '2026-10-09T09:33:05+00:00'}
    s = fr.pipeline_status(meta, pd.Timestamp('2026-10-11T06:33:05', tz='UTC'))
    assert s['state'] == fr.OVERDUE and s['minutes_to_next'] is None
    assert s['overdue_minutes'] == 45 * 60
    assert s['output_age_hours'] == pytest.approx(46.0)
    assert fr.pipeline_status(meta, pd.Timestamp('2026-10-09T09:00:05', tz='UTC'))['minutes_to_next'] == 33
    # within the grace period it is still on schedule (0 min), not overdue
    assert fr.pipeline_status(meta, pd.Timestamp('2026-10-09T09:40:05', tz='UTC'))['state'] == fr.ON_SCHEDULE
    assert fr.pipeline_status({'next_update_utc': None}, T0)['state'] == fr.NO_SCHEDULE
    assert dl.get_countdown_minutes({'meta': meta}, now=pd.Timestamp('2026-10-11T06:33:05', tz='UTC')) is None


# ── dashboard loader on synthetic output ─────────────────────────────────────
def _synthetic(statuses):
    """statuses: {zone: (availability, alert_status)}; origin T0; horizons 1h and 24h."""
    preds = {'AQI': {}}
    for h in (1, 24):
        rows = []
        for z in range(1, 16):
            av, stt = statuses.get(z, ('UNAVAILABLE', 'UNAVAILABLE_NO_DATA'))
            rows.append({'zone_id': z, 'availability': av, 'alert_status': stt,
                         'mrf_pred': None if av == 'UNAVAILABLE' else 150.0 + z,
                         'mrf_std': 60.0, 'exceedance_prob': None if av == 'UNAVAILABLE' else 0.5,
                         'is_monitored': av == 'MODEL_FORECAST',
                         'target_time_utc': (T0 + pd.Timedelta(hours=h)).isoformat()})
        preds['AQI'][f'{h}h'] = rows
    zone_meta = [{'zone_id': z, 'zone_label': f'Z{z}',
                  'aqi_state': 'MEASURED' if statuses.get(z, ('',))[0] == 'MODEL_FORECAST' else 'NO_STATION',
                  'aqi_observed_at_utc': T0.isoformat() if statuses.get(z, ('',))[0] == 'MODEL_FORECAST' else None,
                  'last_measured_aqi': 120.0 if statuses.get(z, ('',))[0] == 'MODEL_FORECAST' else None,
                  'cpcb_aqi_computed': 120.0 if statuses.get(z, ('',))[0] == 'MODEL_FORECAST' else None}
                 for z in range(1, 16)]
    zin = {str(z): {'last_measured_utc': T0.isoformat()} for z, (av, _) in statuses.items() if av == 'MODEL_FORECAST'}
    return {'meta': {'timestamp_utc': T0.isoformat(), 'data_times': {'aqi_forecast_origin_utc': T0.isoformat()}},
            'predictions': preds, 'zone_meta': zone_meta, 'aqi_zone_inputs': zin,
            'alerts': build_alert_records(preds, zone_meta, T0.isoformat())}


STATUSES = {2: ('MODEL_FORECAST', 'NORMAL'), 3: ('MODEL_ESTIMATE', 'NORMAL'),
            4: ('MODEL_ESTIMATE', 'WARNING'), 5: ('MODEL_FORECAST', 'WATCH')}


def test_current_data_levels_estimates_and_active_alerts():
    d = _synthetic(STATUSES)
    now = T0 + pd.Timedelta(minutes=30)
    risks = dl.get_zone_risk_map(d, now)
    assert risks[2] == 'Low'                                    # current forecast from measured input
    assert risks[3] == 'Estimate'                               # estimate without elevated risk: not green
    assert risks[4] == 'High'                                   # estimate WARNING still surfaces
    assert risks[5] == 'Moderate'
    assert risks[6] == 'Unavailable'                            # distinct from estimate
    active = dl.get_alerts(d, active_only=True, now=now)
    assert {(a['zone_id'], a['horizon']) for a in active} == {(4, '1h'), (4, '24h'), (5, '1h'), (5, '24h')}
    assert all(a['forecast_freshness'] == fr.CURRENT for a in active)
    assert any(a['availability'] == 'MODEL_ESTIMATE' and 'MRF spatial estimate' in a['message'] for a in active)
    assert dl.get_stale_warnings(d, now) == []
    b = dl.evaluation_breakdown(d, now)
    assert b == {'evaluated': 3, 'estimate_uncertain': 1, 'stale': 0, 'unavailable': 11, 'total': 15}
    v = dl.forecast_view(d, 2, 'AQI', '24h', now)
    assert v['target_time_utc'] == (T0 + pd.Timedelta(hours=24)).isoformat()


def test_partly_expired_horizons_only_count_current_ones():
    d = _synthetic(STATUSES)
    now = T0 + pd.Timedelta(hours=2)                            # 1h target passed, 24h still current
    assert dl.forecast_view(d, 5, 'AQI', '1h', now)['forecast_freshness'] == fr.EXPIRED
    assert dl.forecast_view(d, 5, 'AQI', '24h', now)['forecast_freshness'] == fr.CURRENT
    assert {(a['zone_id'], a['horizon']) for a in dl.get_alerts(d, True, now)} == {(4, '24h'), (5, '24h')}
    assert {(a['zone_id'], a['horizon']) for a in dl.get_stale_warnings(d, now)} == {(4, '1h'), (5, '1h')}


def test_stale_input_and_expired_forecasts_are_never_green_or_active():
    d = _synthetic(STATUSES)
    for now in (T0 + pd.Timedelta(hours=5), T0 + pd.Timedelta(hours=30)):   # stale input / all expired
        risks = dl.get_zone_risk_map(d, now)
        assert 'Low' not in risks.values() and 'Estimate' not in risks.values()
        assert risks[2] == 'Stale' and risks[3] == 'Stale'
        assert risks[4] == 'Stale warning' and risks[5] == 'Stale warning'
        assert dl.get_alerts(d, active_only=True, now=now) == []
        assert {a['zone_id'] for a in dl.get_stale_warnings(d, now)} == {4, 5}
        assert dl.evaluation_breakdown(d, now) == {'evaluated': 0, 'estimate_uncertain': 0, 'stale': 4,
                                                    'unavailable': 11, 'total': 15}
        obs = dl.aqi_observation(d, 2, now)
        assert obs['state'] == 'MEASURED_STALE' and obs['value'] == 120.0   # value kept, never "current"
        stats = dl.get_summary_stats(d, now)
        assert stats['avg_aqi'] is None and stats['monitored_aqi'] == 0 and stats['stale_aqi'] == 2
        v = dl.forecast_view(d, 5, 'AQI', '1h', now)
        assert v['mrf_pred'] == 155.0 and v['target_time_utc'] == (T0 + pd.Timedelta(hours=1)).isoformat()


# ── committed live output (read-only) ────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(LIVE_JSON), reason='no live output file')
def test_committed_live_output_viewed_later_is_stale_not_green():
    raw = open(LIVE_JSON, 'rb').read()
    d = json.loads(raw)
    now = fr.to_utc(d['meta']['generated_at_utc']) + pd.Timedelta(hours=48)
    for z in range(1, 16):
        obs = dl.aqi_observation(d, z, now)
        assert obs['state'] != 'MEASURED'
        zm = dl.get_zone_meta_map(d)[z]
        if zm.get('last_measured_aqi') is not None:
            assert obs['state'] == 'MEASURED_STALE' and obs['value'] == zm['last_measured_aqi']
    assert set(dl.get_zone_risk_map(d, now).values()) <= {'Stale', 'Stale warning', 'Unavailable'}
    assert dl.get_alerts(d, active_only=True, now=now) == []
    b = dl.evaluation_breakdown(d, now)
    assert b['evaluated'] == 0 and b['stale'] + b['unavailable'] == 15
    for h, rows in d['predictions']['AQI'].items():          # model values and targets are untouched
        for r in rows:
            v = dl.forecast_view(d, r['zone_id'], 'AQI', h, now)
            assert v['target_time_utc'] == r['target_time_utc'] and v['mrf_pred'] == r['mrf_pred']
    assert dl.get_pipeline_status(d, now)['state'] in (fr.OVERDUE, fr.NO_SCHEDULE)
    assert hashlib.sha256(open(LIVE_JSON, 'rb').read()).digest() == hashlib.sha256(raw).digest()


# ── pipeline writes freshness-correct output ─────────────────────────────────
def _station_rows(start, hours, pm25):
    t = pd.date_range(start, periods=hours, freq='h', tz='UTC') + pd.Timedelta(minutes=30)
    frames = []
    for sid, pol, val in ((1, 'PM2.5', pm25), (2, 'PM10', 80.0), (3, 'Ozone', 30.0)):
        frames.append(pd.DataFrame({'hour_start_utc': t.strftime('%Y-%m-%dT%H:%M:%SZ'), 'value': val,
                                    'observed_count': 4, 'location_id': 2586, 'zone_id': 2,
                                    'station_name': 'st2586', 'station_role': 'primary',
                                    'sensor_id': sid, 'cpcb_pollutant': pol}))
    return pd.concat(frames, ignore_index=True)


def _run_pipeline(tmp_path, monkeypatch, now, scheduled=False):
    from src.live import live_pipeline
    raw = tmp_path / 'raw'
    raw.mkdir(exist_ok=True)
    _station_rows('2026-10-01', 96, pm25=150.0).to_csv(raw / 'openaq.csv', index=False)  # ends 2026-10-04 23:30
    t = pd.date_range(pd.Timestamp('2026-09-01', tz='UTC'), now.floor('h'), freq='h')
    pd.DataFrame([{'zone_id': z, 'time_utc': x, 'temperature_2m_c': 29.0, 'precipitation_mm': 0.0,
                   'relative_humidity_2m_pct': 70, 'wind_speed_10m_kmh': 10, 'wind_direction_10m_deg': 90,
                   'grid_latitude': 13.0, 'grid_longitude': 80.2 + (z % 8) / 100, 'source': 'test'}
                  for z in range(1, 16) for x in t]).to_csv(raw / 'om.csv', index=False)
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(raw / 'openaq.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(raw / 'om.csv'))
    monkeypatch.setattr(live_collect, 'load_status', lambda: {
        'openaq': {'ok': True, 'latest_record_utc': '2026-10-04T23:30:00+00:00'},
        'open_meteo': {'ok': True, 'latest_record_utc': now.isoformat()}})
    monkeypatch.setattr(live_pipeline, 'load_validated_sensors', lambda: {1, 2, 3})
    out = tmp_path / 'pred.json'
    live_pipeline.run_once(fetch=False, now_utc=now, output_path=str(out), verbose=False, scheduled=scheduled)
    return json.loads(out.read_text(encoding='utf-8'))


def test_pipeline_marks_old_observation_stale_and_splits_stale_warnings(tmp_path, monkeypatch):
    d = _run_pipeline(tmp_path, monkeypatch, pd.Timestamp('2026-10-05 06:00', tz='UTC'))   # data 6.5 h old
    zm2 = {z['zone_id']: z for z in d['zone_meta']}[2]
    assert zm2['aqi_state'] == 'MEASURED_STALE'
    assert zm2['cpcb_aqi_computed'] is None and zm2['aqi_category'] == 'Unavailable'
    assert zm2['last_measured_aqi'] is not None and zm2['aqi_age_hours'] == pytest.approx(6.5)
    assert zm2['aqi_observed_at_utc'] == '2026-10-04T23:30:00+00:00'      # real timestamp kept
    assert d['active_alerts'] == []
    elevated = [a for a in d['alerts'] if a['alert_status'] in ('WATCH', 'WARNING')]
    assert len(d['stale_warnings']) == len(elevated)
    assert all(a['forecast_freshness'] in (fr.EXPIRED, fr.STALE_INPUT) for a in d['stale_warnings'])
    aqi_1h = {r['zone_id']: r for r in d['predictions']['AQI']['1h']}
    assert aqi_1h[2]['target_time_utc'] == '2026-10-05T00:30:00+00:00'     # origin + 1 h, not moved forward
    m = d['meta']
    assert m['run_mode'] == 'once' and m['next_update_utc'] is None
    assert m['freshness_rules']['aqi_observation_max_age_hours'] == fr.AQI_OBS_MAX_AGE_HOURS


def test_pipeline_current_data_gives_measured_and_active_alerts(tmp_path, monkeypatch):
    d = _run_pipeline(tmp_path, monkeypatch, pd.Timestamp('2026-10-05 00:00', tz='UTC'), scheduled=True)
    zm2 = {z['zone_id']: z for z in d['zone_meta']}[2]
    assert zm2['aqi_state'] == 'MEASURED' and zm2['cpcb_aqi_computed'] is not None
    assert all(a['forecast_freshness'] == fr.CURRENT for a in d['active_alerts'])
    elevated = [a for a in d['alerts'] if a['alert_status'] in ('WATCH', 'WARNING')]
    assert len(d['active_alerts']) + len(d['stale_warnings']) == len(elevated)
    assert d['meta']['run_mode'] == 'scheduler' and d['meta']['next_update_utc'] is not None
