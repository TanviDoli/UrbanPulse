"""
Live preprocessing (component 2 of 4: collect -> PREPROCESS -> infer -> present).

Turns the raw files written by src/collectors/live_collect.py into model-ready inputs.
No model is called here and nothing is fabricated: missing hours stay NaN.

AIR QUALITY (measured, OpenAQ):
  - Only sensors that passed the unit validation in reports/unit_validation.csv are used.
    That audit showed OpenAQ's unit labels for CPCB gas sensors can be wrong (e.g. CO and
    NO2 labelled 'ppb' while the values are CPCB-native mg/m³ / µg/m³), so labels are not trusted.
  - An hour is valid only if >= 2 of the 4 sub-hourly measurements were present (same rule as
    the training pipeline). Values < 0 or above the existing physical bounds in src/live/schemas.py
    (e.g. PM2.5 > 1000 µg/m³) are invalid.
  - Hourly CPCB AQI is computed with src.preprocessing.cpcb_aqi.aqi_from_hourly_concentrations,
    i.e. exactly the methodology used to build the HMM training data.

WEATHER (reanalysis, Open-Meteo): passed through with provenance='reanalysis'.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.preprocessing.cpcb_aqi import aqi_from_hourly_concentrations, aqi_category
from src.live.schemas import PHYSICAL_BOUNDS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UNIT_VALIDATION_FILE = os.path.join(ROOT, 'reports', 'unit_validation.csv')
MIN_SUBHOUR_COUNT = 2


def load_validated_sensors(path: str = UNIT_VALIDATION_FILE) -> Optional[set]:
    """Sensor ids whose values were verified to be in CPCB-native units. None if no table."""
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    return set(v.loc[v['verdict'] == 'native_cpcb_units', 'sensor_id'].astype(int))


def station_hourly_concentrations(records: pd.DataFrame, hour_index: pd.DatetimeIndex,
                                  validated: Optional[set]) -> pd.DataFrame:
    """
    records: OpenAQ hour rows for ONE station (columns hour_start_utc, value, observed_count,
             sensor_id, cpcb_pollutant). Returns a frame indexed by hour_index (UTC hour starts)
             with one column per CPCB pollutant, NaN where invalid/missing.
    """
    out = pd.DataFrame(index=hour_index)
    if records.empty:
        return out
    r = records.copy()
    r['t'] = pd.to_datetime(r['hour_start_utc'], utc=True)
    r['value'] = pd.to_numeric(r['value'], errors='coerce')
    ok = r['value'].notna() & (r['value'] >= 0)
    if 'observed_count' in r:
        oc = pd.to_numeric(r['observed_count'], errors='coerce')
        ok &= oc.isna() | (oc >= MIN_SUBHOUR_COUNT)
    if validated is not None:
        ok &= r['sensor_id'].astype(int).isin(validated)
    r = r[ok]
    for pol, g in r.groupby('cpcb_pollutant'):
        lo, hi = PHYSICAL_BOUNDS.get(pol, (0.0, np.inf))   # existing project bounds (schemas.py)
        g = g[(g['value'] >= lo) & (g['value'] <= hi)]
        s = g.drop_duplicates('t').set_index('t')['value']
        out[pol] = s.reindex(hour_index)
    return out


def build_aqi_inputs(openaq: pd.DataFrame, end_hour_utc: pd.Timestamp, hours: int = 96,
                     validated: Optional[set] = None) -> Dict[int, Dict[str, Any]]:
    """
    Per-station hourly AQI over the window ending at end_hour_utc (inclusive).
    OpenAQ hours start at :30 UTC (IST hour boundaries), so the grid is aligned to that.

    Returns {location_id: {zone_id, station_name, role, aqi (Series), dominant (Series),
             pollutants_used (Series), latest_valid_utc, latest_aqi, ...}}
    """
    if validated is None:
        validated = load_validated_sensors()
    end = end_hour_utc.floor('h') + pd.Timedelta(minutes=30)
    if end > end_hour_utc:
        end -= pd.Timedelta(hours=1)
    idx = pd.date_range(end - pd.Timedelta(hours=hours - 1), end, freq='h')
    stations: Dict[int, Dict[str, Any]] = {}
    if openaq.empty:
        return stations
    for lid, g in openaq.groupby('location_id'):
        conc = station_hourly_concentrations(g, idx, validated)
        aqi = aqi_from_hourly_concentrations(conc)
        # The 24 h rolling window (min 16 h) keeps yielding an "AQI" for up to 8 h after a station
        # stops reporting. Live, an hour counts as measured only if the station reported in that hour.
        no_obs = conc.notna().sum(axis=1) == 0 if len(conc.columns) else pd.Series(True, index=idx)
        aqi.loc[no_obs.reindex(aqi.index, fill_value=True).values,
                ['AQI', 'dominant_pollutant', 'pollutants_used']] = [np.nan, None, None]
        valid = aqi['AQI'].dropna()
        latest_t = valid.index.max() if len(valid) else None
        stations[int(lid)] = {
            'location_id': int(lid),
            'zone_id': int(g['zone_id'].iloc[0]),
            'station_name': g['station_name'].iloc[0],
            'role': g['station_role'].iloc[0],
            'aqi': aqi['AQI'],
            'dominant': aqi['dominant_pollutant'],
            'pollutants_used': aqi['pollutants_used'],
            'latest_valid_utc': latest_t,
            'latest_aqi': float(valid.iloc[-1]) if len(valid) else None,
            'latest_category': aqi_category(float(valid.iloc[-1])) if len(valid) else 'Unavailable',
            'latest_dominant': aqi['dominant_pollutant'].get(latest_t) if latest_t is not None else None,
            'latest_pollutants_used': aqi['pollutants_used'].get(latest_t) if latest_t is not None else None,
            'pollutants_received': sorted(conc.columns[conc.notna().any()].tolist()),
        }
    return stations


def weather_current(om: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Latest complete hourly reanalysis values per zone, plus trailing 24 h precipitation."""
    out: Dict[int, Dict[str, Any]] = {}
    if om.empty:
        return out
    om = om.copy()
    om['time_utc'] = pd.to_datetime(om['time_utc'], utc=True)
    for z, g in om.groupby('zone_id'):
        g = g.dropna(subset=['temperature_2m_c', 'precipitation_mm']).sort_values('time_utc')
        if g.empty:
            continue
        last = g.iloc[-1]
        t = last['time_utc']
        w24 = g[(g['time_utc'] > t - pd.Timedelta(hours=24)) & (g['time_utc'] <= t)]
        out[int(z)] = {
            'valid_at_utc': t,
            'temperature_c': float(last['temperature_2m_c']),
            'relative_humidity_pct': _f(last.get('relative_humidity_2m_pct')),
            'precip_1h_mm': float(last['precipitation_mm']),
            'precip_24h_mm': float(w24['precipitation_mm'].sum()) if len(w24) == 24 else None,
            'wind_speed_kmh': _f(last.get('wind_speed_10m_kmh')),
            'wind_direction_deg': _f(last.get('wind_direction_10m_deg')),
            'grid_cell': (f"{last['grid_latitude']:.4f},{last['grid_longitude']:.4f}"
                          if 'grid_latitude' in g and pd.notna(last.get('grid_latitude')) else None),
            'provenance': 'reanalysis',
            'source': last.get('source', 'open_meteo'),
        }
    return out


def hourly_series(om: pd.DataFrame, zone_id: int, column: str, end_utc: pd.Timestamp,
                  hours: int) -> pd.Series:
    """Gap-preserving hourly series for one zone ending at end_utc (inclusive)."""
    idx = pd.date_range(end_utc - pd.Timedelta(hours=hours - 1), end_utc, freq='h')
    g = om[om['zone_id'] == zone_id].copy()
    g['time_utc'] = pd.to_datetime(g['time_utc'], utc=True)
    s = g.drop_duplicates('time_utc').set_index('time_utc')[column]
    return s.reindex(idx)


def _f(v) -> Optional[float]:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None
