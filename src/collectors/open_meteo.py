"""
Open-Meteo weather loader / collector for UrbanPulse (data collection layer only).

PROVENANCE: Open-Meteo Historical Weather API values are reanalysis / numerical-model
derived (ERA5 family + ECMWF IFS analysis), NOT measured station observations. Every
frame returned here carries `source='open_meteo'` and `provenance='reanalysis'`.

Local file: data/external/open_meteo_chennai_hourly_2025_2026.csv
  The format is detected from the CONTENT (JSON vs CSV), never from the extension.
  The CSV export is two blank-line-separated blocks:
    1) location block: location_id, latitude, longitude, elevation, utc_offset_seconds, timezone, ...
    2) hourly block:   location_id, time, temperature_2m (°C), relative_humidity_2m (%), ...

Location -> zone mapping (verified in reports/open_meteo_audit/):
  location_id i was requested at the centroid of GeoJSON feature i (feature order of
  chennai_zones_2025.geojson). Evidence: mean distance from returned grid cell to that
  centroid is 3.5 km, versus 9.5 km under the zone-number ordering. Open-Meteo snaps
  requests to model grid cells, so several zones share one cell (8 distinct cells for
  15 zones) — those zones receive identical weather series.
"""

import io
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOCAL_FILE = os.path.join(ROOT, 'data', 'external', 'open_meteo_chennai_hourly_2025_2026.csv')
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'

HOURLY_VARS = ['temperature_2m', 'relative_humidity_2m', 'precipitation',
               'wind_speed_10m', 'wind_direction_10m']

# Canonical column names (unit in the name) for the normalised hourly frame.
CANONICAL = {
    'temperature_2m': 'temperature_2m_c',
    'relative_humidity_2m': 'relative_humidity_2m_pct',
    'precipitation': 'precipitation_mm',
    'wind_speed_10m': 'wind_speed_10m_kmh',
    'wind_direction_10m': 'wind_direction_10m_deg',
}
EXPECTED_UNITS = {
    'temperature_2m': '°C', 'relative_humidity_2m': '%', 'precipitation': 'mm',
    'wind_speed_10m': 'km/h', 'wind_direction_10m': '°',
}

# GeoJSON feature order -> (zone_id, zone_label); location_id == feature index.
LOCATION_TO_ZONE: Dict[int, Tuple[int, str]] = {
    0: (1, 'Thiruvottiyur'), 1: (2, 'Manali'), 2: (7, 'Ambattur'), 3: (6, 'Thiru-Vi-Ka Nagar'),
    4: (10, 'Kodambakkam'), 5: (12, 'Alandur'), 6: (11, 'Valasaravakkam'), 7: (4, 'Tondiarpet'),
    8: (5, 'Royapuram'), 9: (9, 'Teynampet'), 10: (8, 'Anna Nagar'), 11: (13, 'Adyar'),
    12: (15, 'Sholinganallur'), 13: (14, 'Perungudi'), 14: (3, 'Madhavaram'),
}

# Zone centroids (Shapely centroid of each GCC 2025 polygon) — the request coordinates.
ZONE_CENTROIDS: Dict[int, Tuple[float, float]] = {
    1: (13.1785, 80.3015), 2: (13.1877, 80.2658), 3: (13.1498, 80.2170), 4: (13.1291, 80.2681),
    5: (13.0888, 80.2796), 6: (13.1098, 80.2384), 7: (13.1082, 80.1712), 8: (13.0876, 80.2216),
    9: (13.0479, 80.2558), 10: (13.0450, 80.2101), 11: (13.0485, 80.1690), 12: (13.0008, 80.1797),
    13: (13.0008, 80.2382), 14: (12.9511, 80.2200), 15: (12.9023, 80.2362),
}


def detect_format(path: str) -> str:
    """Return 'json' or 'csv' based on file content (first non-whitespace character)."""
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        head = f.read(4096).lstrip()
    if not head:
        raise ValueError(f'{path} is empty')
    if head[0] in '{[':
        return 'json'
    return 'csv'


def _normalise_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map 'temperature_2m (°C)' style headers to canonical names; return units seen."""
    rename, units = {}, {}
    for col in df.columns:
        base = col.split(' (')[0].strip()
        if base in CANONICAL:
            rename[col] = CANONICAL[base]
            units[base] = col[len(base):].strip(' ()') if '(' in col else ''
    return df.rename(columns=rename), units


def load_local(path: str = LOCAL_FILE) -> Dict[str, object]:
    """
    Load the local Open-Meteo export. Returns dict with:
      locations: DataFrame (location_id, latitude, longitude, elevation, zone_id, zone_label, ...)
      hourly:    DataFrame (zone_id, location_id, time_utc, <canonical vars>, source, provenance)
      units:     dict of units as written in the file
      format:    'csv' | 'json'
    """
    fmt = detect_format(path)
    if fmt == 'json':
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
        items = payload if isinstance(payload, list) else [payload]
        loc_rows, frames, units = [], [], {}
        for i, it in enumerate(items):
            loc_rows.append({'location_id': i, 'latitude': it.get('latitude'), 'longitude': it.get('longitude'),
                             'elevation': it.get('elevation'), 'utc_offset_seconds': it.get('utc_offset_seconds'),
                             'timezone': it.get('timezone')})
            h = pd.DataFrame(it.get('hourly', {}))
            h.insert(0, 'location_id', i)
            frames.append(h)
            units.update({k: v for k, v in (it.get('hourly_units') or {}).items() if k in CANONICAL})
        locations = pd.DataFrame(loc_rows)
        hourly = pd.concat(frames, ignore_index=True).rename(columns=CANONICAL)
    else:
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            text = f.read()
        blocks = [b for b in text.replace('\r\n', '\n').split('\n\n') if b.strip()]
        if len(blocks) == 1:
            locations = pd.DataFrame()
            hourly = pd.read_csv(io.StringIO(blocks[0]))
        else:
            locations = pd.read_csv(io.StringIO(blocks[0]))
            hourly = pd.read_csv(io.StringIO(blocks[1]))
        hourly, units = _normalise_columns(hourly)

    tz = None
    if len(locations) and 'utc_offset_seconds' in locations:
        offsets = set(locations['utc_offset_seconds'].dropna().astype(int))
        if offsets != {0}:
            tz = offsets
    if tz:
        raise ValueError(f'Unexpected non-UTC offsets in Open-Meteo file: {tz}')
    hourly['time_utc'] = pd.to_datetime(hourly['time'], utc=True)
    hourly = hourly.drop(columns=['time'])
    if len(locations):
        locations['zone_id'] = locations['location_id'].map(lambda i: LOCATION_TO_ZONE[int(i)][0])
        locations['zone_label'] = locations['location_id'].map(lambda i: LOCATION_TO_ZONE[int(i)][1])
    hourly['zone_id'] = hourly['location_id'].map(lambda i: LOCATION_TO_ZONE[int(i)][0])
    hourly['source'] = 'open_meteo_local_file'
    hourly['provenance'] = 'reanalysis'
    return {'locations': locations, 'hourly': hourly, 'units': units, 'format': fmt}


def fetch_archive(start_date: str, end_date: str, now_utc: Optional[pd.Timestamp] = None,
                  timeout_s: float = 60.0, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Fetch hourly weather for all 15 zone centroids from the Open-Meteo Historical
    Weather (archive) API — the same source and variables as the local file.

    Rows timestamped after `now_utc` are DROPPED: the archive endpoint fills the most
    recent hours with model forecast values, which are not usable as past conditions.
    Raises requests.RequestException / ValueError on failure (caller handles).
    """
    now_utc = now_utc or pd.Timestamp.now(tz='UTC')
    zones = sorted(ZONE_CENTROIDS)
    params = {
        'latitude': ','.join(str(ZONE_CENTROIDS[z][0]) for z in zones),
        'longitude': ','.join(str(ZONE_CENTROIDS[z][1]) for z in zones),
        'start_date': start_date, 'end_date': end_date,
        'hourly': ','.join(HOURLY_VARS), 'timezone': 'GMT',
    }
    sess = session or requests.Session()
    resp = sess.get(ARCHIVE_URL, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    items = payload if isinstance(payload, list) else [payload]
    if len(items) != len(zones):
        raise ValueError(f'Open-Meteo returned {len(items)} locations, expected {len(zones)}')
    frames = []
    for z, it in zip(zones, items):
        units = it.get('hourly_units') or {}
        for var, unit in EXPECTED_UNITS.items():
            if units.get(var) not in (None, unit):
                raise ValueError(f'Unexpected unit for {var}: {units.get(var)} (expected {unit})')
        h = pd.DataFrame(it['hourly']).rename(columns=CANONICAL)
        h['time_utc'] = pd.to_datetime(h.pop('time'), utc=True)
        h['zone_id'] = z
        h['grid_latitude'] = it.get('latitude')
        h['grid_longitude'] = it.get('longitude')
        frames.append(h)
    df = pd.concat(frames, ignore_index=True)
    df = df[df['time_utc'] <= now_utc.floor('h')]
    df['source'] = 'open_meteo_archive_api'
    df['provenance'] = 'reanalysis'
    return df


def daily_mean_temperature(hourly: pd.DataFrame, min_hours: int = 24) -> pd.DataFrame:
    """
    Daily mean 2 m temperature per zone and IST calendar day, from complete days only
    (default: all 24 hourly values present). Incomplete days are returned as NaN, never filled.
    """
    h = hourly[['zone_id', 'time_utc', 'temperature_2m_c']].copy()
    h['date_ist'] = h['time_utc'].dt.tz_convert('Asia/Kolkata').dt.date
    g = h.groupby(['zone_id', 'date_ist'])['temperature_2m_c']
    out = g.agg(['mean', 'count']).reset_index()
    out.loc[out['count'] < min_hours, 'mean'] = np.nan
    return out.rename(columns={'mean': 'tmean_c', 'count': 'n_hours'})
