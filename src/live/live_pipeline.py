"""
UrbanPulse Live Pipeline Orchestrator — v3 (OpenAQ + Open-Meteo).

    collect     src/collectors/live_collect.py      -> data/live/raw/*.csv + collection_status.json
    preprocess  src/preprocessing/live_inputs.py    -> station AQI series, weather inputs
    infer       src/forecasting/live_forecaster.py  -> frozen HMM + MRF predictions
    alerts      src/alerts/alert_backend.py
    present     dashboard/app.py reads data/live/current_predictions.json (never calls APIs/models)

The previous v2 orchestrator used WeatherAPI *modelled* air-quality concentrations as if they
were CPCB AQI observations and fed an instantaneous temperature into a daily-mean model; it is
preserved in backups/2026-10-09_pre_final_audit/src/live/live_pipeline.py.

Usage:
    python -m src.live.live_pipeline --once              # collect + infer once
    python -m src.live.live_pipeline --once --no-fetch   # infer from the last collected files
    python -m src.live.live_pipeline                     # hourly scheduler
"""

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.collectors import live_collect
from src.collectors.open_meteo import daily_mean_temperature
from src.preprocessing.live_inputs import build_aqi_inputs, weather_current, hourly_series, load_validated_sensors
from src.forecasting.live_forecaster import (LiveForecaster, AQI_HMM_ZONES, RAIN_HMM_ZONES, RAIN_CONTEXT_HOURS,
                                             RAINFALL_FORECAST_ENABLED)
from src.alerts.alert_backend import build_alert_records, filter_active_alerts, format_alert_summary

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'live',
                                           'current_predictions.json'))
REFRESH_INTERVAL_SECONDS = 60 * 60          # both sources update hourly at best
STALE_AFTER_HOURS = {'openaq': 3.0, 'open_meteo': 6.0}

ZONE_LABELS = {1: 'Thiruvottiyur', 2: 'Manali', 3: 'Madhavaram', 4: 'Tondiarpet', 5: 'Royapuram',
               6: 'Thiru-Vi-Ka Nagar', 7: 'Ambattur', 8: 'Anna Nagar', 9: 'Teynampet', 10: 'Kodambakkam',
               11: 'Valasaravakkam', 12: 'Alandur', 13: 'Adyar', 14: 'Perungudi', 15: 'Sholinganallur'}

LIMITATIONS = [
    'AQI values are computed by UrbanPulse with the CPCB NAQI sub-index method (24 h / 8 h rolling '
    'means, >=3 pollutants incl. PM) from OpenAQ hourly station data; they are not official CPCB bulletins.',
    'Only 7 zones have an AQI HMM; other zones show an MRF spatial estimate (not a measurement) when '
    'an adjacent zone has measured input, otherwise unavailable.',
    'OpenAQ CPCB data typically lags real time by 1-2 days; AQI forecast target times are relative to '
    'the latest measured hour, which may already be in the past.',
    'Weather is Open-Meteo reanalysis/model analysis, not station measurement; the 15 zones map to only '
    '8 distinct model grid cells, so several zones share identical weather.',
    'Rainfall forecasts and heavy-rain alerts are disabled: the only rainfall HMMs (zones 4, 5, 10, 14) were '
    'trained on CAAQMS gauges, no live gauge feed exists, and reanalysis input failed validation. Current '
    'rainfall is shown from Open-Meteo reanalysis for all zones.',
    'Temperature forecast is the next-day DAILY MEAN; no sub-daily temperature forecast and no Tmax '
    'forecast exist, so the IMD heatwave criterion is not evaluated. The frozen temperature HMM (171 '
    'training days, Jan-Jun 2024) has low skill: next-day MAE ~1.0 °C vs ~0.5 °C typical day-to-day change.',
]


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _source_state(status: Dict[str, Any], name: str, now: pd.Timestamp) -> Dict[str, Any]:
    st = dict(status.get(name) or {})
    latest = st.get('latest_record_utc')
    age = (now - pd.Timestamp(latest)) / pd.Timedelta(hours=1) if latest else None
    st['data_age_hours'] = round(age, 1) if age is not None else None
    st['stale'] = age is None or age > STALE_AFTER_HOURS[name]
    return st


def _clean(o):
    """Recursively replace NaN/inf with None so the JSON is strict and NaN is never shown as a value."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def _iso(t) -> Optional[str]:
    return None if t is None or (isinstance(t, float) and np.isnan(t)) else pd.Timestamp(t).isoformat()


def run_once(fetch: bool = True, forecaster: Optional[LiveForecaster] = None,
             now_utc: Optional[pd.Timestamp] = None, output_path: str = OUTPUT_PATH,
             verbose: bool = True) -> Dict[str, Any]:
    now = now_utc or pd.Timestamp.now(tz='UTC')
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n{'=' * 65}\nUrbanPulse live pipeline v3 — {now.isoformat()}\n{'=' * 65}")

    # 1. Collect
    if fetch:
        log('Step 1: collecting OpenAQ + Open-Meteo ...')
        status = live_collect.run_collection(now_utc=now, verbose=verbose)
    else:
        log('Step 1: --no-fetch, using last collected files')
        status = live_collect.load_status()
    src_openaq = _source_state(status, 'openaq', now)
    src_om = _source_state(status, 'open_meteo', now)

    # 2. Preprocess
    log('Step 2: preprocessing ...')
    oaq = _read_csv(live_collect.OPENAQ_FILE)
    om = _read_csv(live_collect.OPEN_METEO_FILE)
    if len(om):
        om['time_utc'] = pd.to_datetime(om['time_utc'], utc=True)
        om = om[om['time_utc'] <= now]
    stations = build_aqi_inputs(oaq, now, validated=load_validated_sensors()) if len(oaq) else {}
    weather = weather_current(om)
    if not stations and not weather:
        raise RuntimeError('No usable input data from any source; previous output left unchanged.')

    # 3. Infer
    log('Step 3: HMM + MRF inference ...')
    fc = forecaster or LiveForecaster()
    aqi = fc.run_aqi(stations)
    rain_origin = None
    precip = {}
    if len(om):
        complete = om.dropna(subset=['precipitation_mm']).groupby('time_utc')['zone_id'].nunique()
        complete = complete[complete == 15]
        if len(complete):
            rain_origin = complete.index.max()
            precip = {z: hourly_series(om, z, 'precipitation_mm', rain_origin, RAIN_CONTEXT_HOURS)
                      for z in RAIN_HMM_ZONES}
    rain = fc.run_rainfall(precip, rain_origin, enabled=RAINFALL_FORECAST_ENABLED)
    temp = fc.run_temperature(daily_mean_temperature(om) if len(om) else
                              pd.DataFrame(columns=['zone_id', 'date_ist', 'tmean_c']))

    # zone_meta: current conditions with provenance
    shared = {}
    for z, w in weather.items():
        shared.setdefault(w.get('grid_cell'), []).append(z)
    zone_meta = []
    zin = aqi.get('zone_inputs', {})
    for z in range(1, 16):
        prim = next((s for s in stations.values() if s['zone_id'] == z and s['role'] == 'primary'), None)
        secondary = [s for s in stations.values() if s['zone_id'] == z and s['role'] == 'secondary']
        w = weather.get(z, {})
        aqi_age = ((now - prim['latest_valid_utc']) / pd.Timedelta(hours=1)
                   if prim and prim['latest_valid_utc'] is not None else None)
        if prim is None:
            aqi_state = 'NO_STATION' if z not in AQI_HMM_ZONES and z != 1 else 'NO_LIVE_DATA'
        elif prim['latest_valid_utc'] is None:
            aqi_state = 'NO_LIVE_DATA'
        elif zin.get(z, {}).get('reason') == 'UNAVAILABLE_STALE_INPUT':
            aqi_state = 'MEASURED_STALE'
        else:
            aqi_state = 'MEASURED'
        zone_meta.append({
            'zone_id': z, 'zone_label': ZONE_LABELS[z],
            # air quality (measured)
            'aqi_state': aqi_state,
            'cpcb_aqi_computed': prim['latest_aqi'] if prim and aqi_state == 'MEASURED' else None,
            'last_measured_aqi': prim['latest_aqi'] if prim else None,
            'aqi_category': prim['latest_category'] if prim and aqi_state == 'MEASURED' else 'Unavailable',
            'aqi_observed_at_utc': _iso(prim['latest_valid_utc']) if prim else None,
            'aqi_age_hours': round(aqi_age, 1) if aqi_age is not None else None,
            'aqi_station': prim['station_name'] if prim else None,
            'dominant_pollutant': prim['latest_dominant'] if prim else None,
            'aqi_pollutants_used': prim['latest_pollutants_used'] if prim else None,
            'aqi_source': 'OpenAQ (CPCB/TNPCB CAAQMS station, measured)' if prim else None,
            'has_aqi_model': z in AQI_HMM_ZONES,
            'secondary_stations': [{'station_name': s['station_name'], 'aqi': s['latest_aqi'],
                                    'observed_at_utc': _iso(s['latest_valid_utc'])} for s in secondary],
            # weather (reanalysis)
            'temp_c_current': w.get('temperature_c'),
            'humidity_pct': w.get('relative_humidity_pct'),
            'rainfall_mm': w.get('precip_1h_mm'),
            'precip_24h_mm': w.get('precip_24h_mm'),
            'wind_speed_kmh': w.get('wind_speed_kmh'),
            'weather_valid_at_utc': _iso(w.get('valid_at_utc')),
            'weather_provenance': 'Open-Meteo reanalysis/model analysis (not a station measurement)' if w else None,
            'weather_grid_cell': w.get('grid_cell'),
            'weather_shared_with_zones': [x for x in shared.get(w.get('grid_cell'), []) if x != z] if w else [],
            'has_rainfall_model': z in RAIN_HMM_ZONES,
            'rainfall_forecast_enabled': bool(RAINFALL_FORECAST_ENABLED and z in RAIN_HMM_ZONES),
            'is_rainfall_monitored': False,      # no live rain gauge for any zone
        })

    predictions = {'AQI': aqi.get('horizons', {}), 'rainfall': rain['horizons'], 'temperature': temp['horizons']}
    supported = {
        'AQI': list(aqi.get('horizons', {}).keys()),
        'rainfall': list(rain['horizons'].keys()) if rain['n_active'] else [],
        'temperature': ['24h'] if any(r['availability'] != 'UNAVAILABLE' for r in temp['horizons']['24h']) else [],
    }
    data_time = aqi.get('origin_utc')

    # 4. Alerts
    alerts = build_alert_records(predictions, zone_meta, _iso(data_time) or now.isoformat())
    active = filter_active_alerts(alerts)
    log(format_alert_summary(alerts))

    gen = pd.Timestamp.now(tz='UTC')
    output = {
        'meta': {
            'pipeline_version': '3.0',
            'timestamp_utc': _iso(data_time),
            'generated_at_utc': gen.isoformat(),
            'last_successful_update_utc': gen.isoformat(),
            'next_update_utc': (gen + pd.Timedelta(seconds=REFRESH_INTERVAL_SECONDS)).isoformat(),
            'refresh_interval_seconds': REFRESH_INTERVAL_SECONDS,
            'data_times': {
                'aqi_forecast_origin_utc': _iso(data_time),
                'rainfall_forecast_origin_utc': _iso(rain_origin),
                'temperature_last_input_day_ist': temp['last_input_day_ist'],
            },
            'sources': {'openaq': src_openaq, 'open_meteo': src_om},
            'aqi_active_zones': aqi.get('n_active', 0),
            'rainfall_active_zones': rain['n_active'],
            'rainfall_forecast_enabled': rain.get('enabled', False),
            'supported_horizons': supported,
            'model_artifacts': fc.artifacts_summary(),
            'temperature_limitation': LIMITATIONS[-1],
            'limitations': LIMITATIONS,
        },
        'predictions': predictions,
        'zone_meta': zone_meta,
        'aqi_zone_inputs': aqi.get('zone_inputs', {}),
        'alerts': alerts,
        'active_alerts': active,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_clean(output), f, indent=2, default=str, allow_nan=False)
    os.replace(tmp, output_path)
    log(f'Step 5: written -> {output_path}')
    return output


def run_scheduler(interval_seconds: int = REFRESH_INTERVAL_SECONDS):
    fc = LiveForecaster()
    while True:
        try:
            run_once(fetch=True, forecaster=fc)
        except Exception as e:
            # Previous output is left in place; the dashboard shows its age.
            print(f'[ERROR] pipeline cycle failed: {e}')
            traceback.print_exc()
        time.sleep(interval_seconds)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='UrbanPulse live pipeline v3')
    ap.add_argument('--once', action='store_true', help='Run one cycle and exit')
    ap.add_argument('--no-fetch', action='store_true', help='Skip collection; use last collected files')
    ap.add_argument('--interval', type=int, default=REFRESH_INTERVAL_SECONDS)
    args = ap.parse_args()
    if args.once:
        run_once(fetch=not args.no_fetch)
    else:
        run_scheduler(args.interval)
