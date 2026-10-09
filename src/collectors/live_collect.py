"""
Live data collection (component 1 of 4: collect -> preprocess -> infer -> present).

Fetches recent data from the two sources validated in the final data audit:
  - OpenAQ v3: measured hourly pollutant concentrations for the CPCB/TNPCB stations
    that map to GCC zones (see STATIONS).
  - Open-Meteo Historical Weather API: reanalysis weather for the 15 zone centroids.

Writes (atomically) under data/live/raw/:
  openaq_recent.csv, open_meteo_recent.csv, collection_status.json

Failure policy: if a source fails, its previous file is KEPT (not overwritten with
partial/empty data) and collection_status.json records the error plus the time of the
last successful fetch, so downstream stages can mark the data stale.

Run:  python -m src.collectors.live_collect
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.collectors.openaq_client import OpenAQClient, OpenAQError, flatten_hour_record
from src.collectors import open_meteo
from src.preprocessing.live_inputs import load_validated_sensors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'live', 'raw')
OPENAQ_FILE = os.path.join(RAW_DIR, 'openaq_recent.csv')
OPEN_METEO_FILE = os.path.join(RAW_DIR, 'open_meteo_recent.csv')
STATUS_FILE = os.path.join(RAW_DIR, 'collection_status.json')

OPENAQ_HOURS_BACK = 120        # 24 h CPCB rolling window + 72 h HMM context + slack
OPEN_METEO_DAYS_BACK = 35      # 30 complete days for the daily temperature HMM context

# OpenAQ location -> GCC zone. Mapping is by station identity (name), reconciled against
# data/processed/station_zone_mapping.csv; see reports/final_data_audit.md §3 for why OpenAQ
# coordinates are NOT used directly (e.g. Alandur Bus Depot is reported ~11 km outside GCC).
STATIONS: List[Dict[str, Any]] = [
    {'location_id': 2586, 'station_name': 'Manali, Chennai - CPCB', 'zone_id': 2, 'role': 'primary'},
    {'location_id': 10780, 'station_name': 'Manali Village, Chennai - TNPCB', 'zone_id': 2, 'role': 'secondary'},
    {'location_id': 11579, 'station_name': 'Kodungaiyur, Chennai - TNPCB', 'zone_id': 4, 'role': 'primary'},
    {'location_id': 11578, 'station_name': 'Royapuram, Chennai - TNPCB', 'zone_id': 5, 'role': 'primary'},
    {'location_id': 11581, 'station_name': 'Arumbakkam, Chennai - TNPCB', 'zone_id': 10, 'role': 'primary'},
    {'location_id': 3409319, 'station_name': 'Alandur Bus Depot, Chennai - CPCB', 'zone_id': 12, 'role': 'primary'},
    {'location_id': 5655, 'station_name': 'Velachery Res. Area, Chennai - CPCB', 'zone_id': 13, 'role': 'primary'},
    {'location_id': 12046, 'station_name': 'Perungudi, Chennai - TNPCB', 'zone_id': 14, 'role': 'primary'},
]

# OpenAQ parameter name -> CPCB pollutant name used by src.preprocessing.cpcb_aqi
OPENAQ_TO_CPCB = {'pm25': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO2', 'so2': 'SO2',
                  'co': 'CO', 'o3': 'Ozone', 'nh3': 'NH3'}
# Units that need no ppb conversion (mass concentration)
MASS_UNITS = {'µg/m³', 'ug/m3', 'µg/m3', 'mg/m³', 'mg/m3'}


def select_sensors(sensors: List[Dict[str, Any]], recent_within_hours: float = 72.0,
                   validated: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """
    Choose one sensor per CPCB pollutant for a location.
    If `validated` (sensor ids that passed the unit audit) is given, only those are eligible.
    Rule: the sensor with the most recent `datetimeLast`; among sensors whose latest data
    is within `recent_within_hours` of that most recent one, prefer a mass-concentration
    unit (no ppb conversion needed).
    """
    by_pol: Dict[str, List[Dict[str, Any]]] = {}
    for s in sensors:
        pname = (s.get('parameter') or {}).get('name')
        if pname not in OPENAQ_TO_CPCB:
            continue
        if validated is not None and int(s['id']) not in validated:
            continue
        last = (s.get('datetimeLast') or {}).get('utc')
        if not last:
            continue
        by_pol.setdefault(OPENAQ_TO_CPCB[pname], []).append({
            'sensor_id': s['id'], 'parameter': pname,
            'units': (s.get('parameter') or {}).get('units'),
            'last_utc': pd.Timestamp(last)})
    chosen = {}
    for pol, cands in by_pol.items():
        newest = max(c['last_utc'] for c in cands)
        recent = [c for c in cands if (newest - c['last_utc']) <= pd.Timedelta(hours=recent_within_hours)]
        recent.sort(key=lambda c: (c['units'] in MASS_UNITS, c['last_utc']), reverse=True)
        chosen[pol] = recent[0]
    return chosen


def collect_openaq(client: Optional[OpenAQClient] = None, now_utc: Optional[pd.Timestamp] = None,
                   hours_back: int = OPENAQ_HOURS_BACK) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now_utc = now_utc or pd.Timestamp.now(tz='UTC')
    status: Dict[str, Any] = {'source': 'openaq', 'attempted_at_utc': now_utc.isoformat(),
                              'failures': [], 'stations': []}
    try:
        client = client or OpenAQClient()
    except PermissionError as e:
        status.update({'ok': False, 'error': str(e)})
        return pd.DataFrame(), status
    validated = load_validated_sensors()
    status['unit_validation_applied'] = validated is not None
    dfrom = (now_utc - pd.Timedelta(hours=hours_back)).strftime('%Y-%m-%dT%H:%M:%SZ')
    dto = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    frames = []
    for st in STATIONS:
        st_status = {'location_id': st['location_id'], 'station_name': st['station_name'], 'sensors': {}}
        try:
            chosen = select_sensors(client.list_sensors(st['location_id']), validated=validated)
        except OpenAQError as e:
            status['failures'].append({'location_id': st['location_id'], 'sensor_id': None, 'error': str(e)})
            status['stations'].append(st_status)
            continue
        for pol, s in chosen.items():
            try:
                recs = client.sensor_hours(s['sensor_id'], dfrom, dto)
            except OpenAQError as e:
                status['failures'].append({'location_id': st['location_id'], 'sensor_id': s['sensor_id'],
                                           'error': str(e)})
                continue
            df = pd.DataFrame([flatten_hour_record(r) for r in recs])
            if df.empty:
                st_status['sensors'][pol] = {'sensor_id': s['sensor_id'], 'units': s['units'], 'records': 0}
                continue
            df['location_id'] = st['location_id']
            df['station_name'] = st['station_name']
            df['zone_id'] = st['zone_id']
            df['station_role'] = st['role']
            df['sensor_id'] = s['sensor_id']
            df['cpcb_pollutant'] = pol
            frames.append(df)
            st_status['sensors'][pol] = {'sensor_id': s['sensor_id'], 'units': s['units'], 'records': len(df)}
        status['stations'].append(st_status)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    status['n_requests'] = len(client.request_log)
    status['n_records'] = int(len(data))
    status['latest_record_utc'] = (pd.to_datetime(data['hour_start_utc'], utc=True).max().isoformat()
                                   if len(data) else None)
    # Partial station failures are tolerated; total failure is not.
    status['ok'] = len(data) > 0
    if not status['ok']:
        status['error'] = 'No OpenAQ records retrieved'
    return data, status


def collect_open_meteo(now_utc: Optional[pd.Timestamp] = None,
                       days_back: int = OPEN_METEO_DAYS_BACK) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    now_utc = now_utc or pd.Timestamp.now(tz='UTC')
    status: Dict[str, Any] = {'source': 'open_meteo', 'attempted_at_utc': now_utc.isoformat()}
    start = (now_utc - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
    end = now_utc.strftime('%Y-%m-%d')
    try:
        df = open_meteo.fetch_archive(start, end, now_utc=now_utc)
    except Exception as e:  # network, HTTP, schema/unit validation
        status.update({'ok': False, 'error': f'{type(e).__name__}: {str(e)[:200]}'})
        return pd.DataFrame(), status
    complete = df.dropna(subset=['temperature_2m_c', 'precipitation_mm'])
    status.update({'ok': len(complete) > 0, 'n_records': int(len(df)),
                   'latest_record_utc': complete['time_utc'].max().isoformat() if len(complete) else None})
    if not status['ok']:
        status['error'] = 'Open-Meteo returned no non-null hourly values'
    return df, status


def _atomic_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_status() -> Dict[str, Any]:
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def run_collection(now_utc: Optional[pd.Timestamp] = None, openaq_client: Optional[OpenAQClient] = None,
                   verbose: bool = True) -> Dict[str, Any]:
    os.makedirs(RAW_DIR, exist_ok=True)
    now_utc = now_utc or pd.Timestamp.now(tz='UTC')
    prev = load_status()
    result = {}
    for name, fn, path in [('openaq', lambda: collect_openaq(openaq_client, now_utc), OPENAQ_FILE),
                           ('open_meteo', lambda: collect_open_meteo(now_utc), OPEN_METEO_FILE)]:
        data, st = fn()
        prev_success = (prev.get(name) or {}).get('last_success_utc')
        if st.get('ok'):
            _atomic_csv(data, path)
            st['last_success_utc'] = now_utc.isoformat()
        else:
            st['last_success_utc'] = prev_success
            st['kept_previous_file'] = os.path.exists(path)
        result[name] = st
        if verbose:
            msg = 'ok' if st.get('ok') else f"FAILED ({st.get('error')}) - keeping previous data"
            print(f"  [collect] {name}: {msg}; records={st.get('n_records', 0)}; "
                  f"latest={st.get('latest_record_utc')}")
    result['collected_at_utc'] = now_utc.isoformat()
    tmp = STATUS_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp, STATUS_FILE)
    return result


if __name__ == '__main__':
    run_collection()
