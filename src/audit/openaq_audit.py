"""
Phase C — OpenAQ audit for Chennai (2025-01-01 .. 2026-10-08, IST hours).

No pre-existing OpenAQ audit script was found in the project (checked 2026-10-09),
so this script was written against live API responses.

Steps
  1. Inventory OpenAQ locations inside the GCC bounding box and map each location's
     reported coordinates to the 2025 GCC zone polygons (point-in-polygon).
  2. Inventory every sensor (parameter, units, metadata first/last timestamps).
  3. Download hourly aggregates (/v3/sensors/{id}/hours) month by month for every
     sensor whose metadata period overlaps the audit window. Raw months are cached
     under data/external/openaq/hours/ so the run is resumable and reproducible.
  4. Compute coverage FROM THE DOWNLOADED RECORDS (sensor metadata coverage is
     reported alongside but never treated as proof of observations).

Outputs (reports/openaq_audit/):
  locations.csv, sensors.csv, sensor_coverage.csv, monthly_coverage_sensor.csv,
  monthly_coverage_location.csv, failed_requests.csv, summary.md

Run:  python -m src.audit.openaq_audit [--refresh-current-month]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.collectors.openaq_client import OpenAQClient, OpenAQError, flatten_hour_record

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'reports', 'openaq_audit')
CACHE_DIR = os.path.join(ROOT, 'data', 'external', 'openaq', 'hours')
GEOJSON = os.path.join(ROOT, 'data', 'raw', 'chennai_zones_2025.geojson')

IST = 'Asia/Kolkata'
# Audit window expressed in IST calendar hours (Chennai local time).
PERIOD_START_IST = pd.Timestamp('2025-01-01 00:00', tz=IST)
PERIOD_END_IST = pd.Timestamp('2026-10-08 23:00', tz=IST)          # last hour start, inclusive
EXPECTED_HOURS = int((PERIOD_END_IST - PERIOD_START_IST) / pd.Timedelta(hours=1)) + 1

POLLUTANTS = {'pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'nh3', 'no', 'nox'}

GEOJSON_NAME_TO_ZONE = {
    'THIRUVOTTRIYUR': (1, 'Thiruvottiyur'), 'MANALI': (2, 'Manali'), 'MADHAVARAM': (3, 'Madhavaram'),
    'TONDIARPET': (4, 'Tondiarpet'), 'ROYAPURAM': (5, 'Royapuram'), 'THIRU-VIKA-NAGAR': (6, 'Thiru-Vi-Ka Nagar'),
    'AMBATTUR': (7, 'Ambattur'), 'ANNA NAGAR': (8, 'Anna Nagar'), 'TEYNAMPET': (9, 'Teynampet'),
    'KODAMBAKKAM': (10, 'Kodambakkam'), 'VALASARVAKKAM': (11, 'Valasaravakkam'), 'ALANDUR': (12, 'Alandur'),
    'ADYAR': (13, 'Adyar'), 'PERUNKUDI': (14, 'Perungudi'), 'SHOZHANGANALLUR': (15, 'Sholinganallur'),
}


def load_zone_polygons():
    with open(GEOJSON, encoding='utf-8') as f:
        g = json.load(f)
    return [(GEOJSON_NAME_TO_ZONE[ft['properties']['name_of_th'].strip().upper()], shape(ft['geometry']))
            for ft in g['features']]


def zone_for_point(lat: float, lon: float, polys) -> Dict:
    p = Point(lon, lat)
    for (zid, zlabel), poly in polys:
        if poly.contains(p):
            return {'pip_zone_id': zid, 'pip_zone_label': zlabel, 'distance_to_nearest_zone_km': 0.0}
    (zid, zlabel), poly = min(polys, key=lambda t: t[1].distance(p))
    return {'pip_zone_id': None, 'pip_zone_label': None,
            'nearest_zone_id': zid, 'nearest_zone_label': zlabel,
            'distance_to_nearest_zone_km': round(poly.distance(p) * 111.0, 2)}


def month_windows():
    """IST calendar months covering the audit window -> list of (label, from_utc, to_utc)."""
    out = []
    m = PERIOD_START_IST
    end_excl = PERIOD_END_IST + pd.Timedelta(hours=1)
    while m < end_excl:
        nxt = min(m + pd.offsets.MonthBegin(1), end_excl)
        out.append((m.strftime('%Y-%m'),
                    m.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
                    nxt.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')))
        m = nxt
    return out


def fetch_sensor(client: OpenAQClient, sensor_id: int, refresh_last: bool, failures: List[Dict]) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    windows = month_windows()
    for i, (label, dfrom, dto) in enumerate(windows):
        path = os.path.join(CACHE_DIR, f'sensor_{sensor_id}_{label}.csv')
        is_last = i == len(windows) - 1
        if os.path.exists(path) and not (is_last and refresh_last):
            frames.append(pd.read_csv(path))
            continue
        try:
            recs = client.sensor_hours(sensor_id, dfrom, dto)
        except OpenAQError as e:
            failures.append({'sensor_id': sensor_id, 'month': label, 'error': str(e)})
            continue
        df = pd.DataFrame([flatten_hour_record(r) for r in recs],
                          columns=['hour_start_utc', 'hour_end_utc', 'value', 'parameter', 'units',
                                   'observed_count', 'expected_count', 'percent_coverage', 'has_flags'])
        df.to_csv(path + '.tmp', index=False)
        os.replace(path + '.tmp', path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_ist_hour(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(IST).dt.floor('h')


def sensor_stats(df: pd.DataFrame, parameter: str) -> Dict:
    if df.empty:
        return ({'total_records': 0, 'valid_records': 0, 'duplicate_hours': 0, 'null_values': 0,
                 'negative_values': 0, 'sentinel_values': 0, 'flagged_records': 0,
                 'missing_hours_full_period': EXPECTED_HOURS, 'coverage_full_period_pct': 0.0},
                pd.DataFrame(columns=['hour_ist', 'value']))
    df = df.copy()
    df['hour_ist'] = to_ist_hour(df['hour_start_utc'])
    in_win = (df['hour_ist'] >= PERIOD_START_IST) & (df['hour_ist'] <= PERIOD_END_IST)
    df = df[in_win]
    total = len(df)
    dup_hours = int(df.duplicated('hour_ist', keep='first').sum())
    v = pd.to_numeric(df['value'], errors='coerce')
    is_pollutant = parameter in POLLUTANTS
    negative = int((v < 0).sum())
    sentinel = int(v.isin([-999, -9999, 999, 9999, 99999]).sum())
    null = int(v.isna().sum())
    valid_mask = v.notna() & ~v.isin([-999, -9999, 999, 9999, 99999])
    if is_pollutant:
        valid_mask &= v >= 0
    valid = df[valid_mask].drop_duplicates('hour_ist')
    n_valid = len(valid)
    out = {
        'total_records': total, 'valid_records': n_valid, 'duplicate_hours': dup_hours,
        'null_values': null, 'negative_values': negative, 'sentinel_values': sentinel,
        'flagged_records': int((df['has_flags'] == True).sum()),  # noqa: E712
        'minute_offsets_utc': ','.join(sorted(set(pd.to_datetime(df['hour_start_utc'], utc=True).dt.minute.astype(str)))),
        'units_seen': ','.join(sorted(set(df['units'].dropna().astype(str)))),
    }
    if n_valid:
        first, last = valid['hour_ist'].min(), valid['hour_ist'].max()
        active_hours = int((last - first) / pd.Timedelta(hours=1)) + 1
        out.update({
            'first_valid_ist': first.isoformat(), 'last_valid_ist': last.isoformat(),
            'missing_hours_full_period': EXPECTED_HOURS - n_valid,
            'coverage_full_period_pct': round(100.0 * n_valid / EXPECTED_HOURS, 2),
            'active_period_hours': active_hours,
            'missing_hours_active_period': active_hours - n_valid,
            'coverage_active_period_pct': round(100.0 * n_valid / active_hours, 2),
            'value_min': float(v[valid_mask].min()), 'value_median': float(v[valid_mask].median()),
            'value_p99': float(v[valid_mask].quantile(0.99)), 'value_max': float(v[valid_mask].max()),
            'mean_subhour_observed_count': round(float(pd.to_numeric(valid['observed_count'], errors='coerce').mean()), 2),
        })
    else:
        out.update({'missing_hours_full_period': EXPECTED_HOURS, 'coverage_full_period_pct': 0.0})
    return out, valid[['hour_ist']].assign(value=v[valid_mask].loc[valid.index].values)


def monthly_counts(valid_hours: pd.Series) -> pd.Series:
    months = valid_hours.dt.strftime('%Y-%m')
    return months.value_counts()


def expected_hours_by_month() -> pd.Series:
    rng = pd.date_range(PERIOD_START_IST, PERIOD_END_IST, freq='h')
    return pd.Series(rng.strftime('%Y-%m')).value_counts().sort_index()


def run(refresh_current_month: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    client = OpenAQClient()
    polys = load_zone_polygons()
    failures: List[Dict] = []
    run_time = datetime.now(timezone.utc).isoformat()
    print(f"[OpenAQ audit] window {PERIOD_START_IST} .. {PERIOD_END_IST} IST ({EXPECTED_HOURS} hours)")

    # 1. Locations
    locations = client.list_locations()
    loc_rows = []
    for L in locations:
        lat, lon = L['coordinates']['latitude'], L['coordinates']['longitude']
        row = {
            'location_id': L['id'], 'name': L['name'],
            'provider': (L.get('provider') or {}).get('name'),
            'owner': (L.get('owner') or {}).get('name'),
            'is_mobile': L.get('isMobile'), 'is_monitor': L.get('isMonitor'),
            'latitude': lat, 'longitude': lon,
            'metadata_first_utc': (L.get('datetimeFirst') or {}).get('utc'),
            'metadata_last_utc': (L.get('datetimeLast') or {}).get('utc'),
            'n_sensors': len(L.get('sensors') or []),
        }
        row.update(zone_for_point(lat, lon, polys))
        loc_rows.append(row)
    loc_df = pd.DataFrame(loc_rows).sort_values('location_id')
    print(f"  locations found in bbox: {len(loc_df)}")

    # 2. Sensors
    sensor_rows = []
    for L in locations:
        try:
            sensors = client.list_sensors(L['id'])
        except OpenAQError as e:
            failures.append({'sensor_id': None, 'month': None, 'error': f"location {L['id']} sensors: {e}"})
            sensors = L.get('sensors') or []
        for s in sensors:
            cov = s.get('coverage') or {}
            sensor_rows.append({
                'location_id': L['id'], 'location_name': L['name'], 'sensor_id': s['id'],
                'parameter': (s.get('parameter') or {}).get('name'),
                'units': (s.get('parameter') or {}).get('units'),
                'metadata_first_utc': (s.get('datetimeFirst') or {}).get('utc'),
                'metadata_last_utc': (s.get('datetimeLast') or {}).get('utc'),
                'metadata_percent_coverage': cov.get('percentCoverage'),
                'metadata_observed_count': cov.get('observedCount'),
            })
    sens_df = pd.DataFrame(sensor_rows).sort_values(['location_id', 'parameter', 'sensor_id'])
    first = pd.to_datetime(sens_df['metadata_first_utc'], utc=True)
    last = pd.to_datetime(sens_df['metadata_last_utc'], utc=True)
    sens_df['overlaps_audit_window'] = (last >= PERIOD_START_IST.tz_convert('UTC')) & \
                                       (first <= (PERIOD_END_IST + pd.Timedelta(hours=1)).tz_convert('UTC'))
    print(f"  sensors: {len(sens_df)} total, {int(sens_df['overlaps_audit_window'].sum())} overlap window")

    # 3+4. Download & compute
    cov_rows, month_rows = [], []
    loc_month_hours: Dict[int, Dict[str, set]] = {}
    todo = sens_df[sens_df['overlaps_audit_window']]
    # Download concurrently (throttled globally inside the client), then compute serially.
    with ThreadPoolExecutor(max_workers=5) as pool:
        downloaded = dict(zip(todo['sensor_id'].astype(int),
                              pool.map(lambda sid: fetch_sensor(client, int(sid), refresh_current_month, failures),
                                       todo['sensor_id'])))
    for _, s in todo.iterrows():
        sid = int(s['sensor_id'])
        print(f"  sensor {sid:>9} {s['location_name'][:34]:34s} {s['parameter']:>16s}", flush=True)
        df = downloaded[sid]
        stats, valid = sensor_stats(df, s['parameter'])
        cov_rows.append({'sensor_id': sid, 'location_id': s['location_id'], 'location_name': s['location_name'],
                         'parameter': s['parameter'], 'metadata_units': s['units'], **stats})
        if len(valid):
            mc = monthly_counts(valid['hour_ist'])
            for month, n in mc.items():
                month_rows.append({'sensor_id': sid, 'location_id': s['location_id'],
                                   'location_name': s['location_name'], 'parameter': s['parameter'],
                                   'month': month, 'valid_hours': int(n)})
            if s['parameter'] in POLLUTANTS:
                d = loc_month_hours.setdefault(int(s['location_id']), {})
                for h in valid['hour_ist']:
                    d.setdefault(h.strftime('%Y-%m'), set()).add(h)

    exp_month = expected_hours_by_month()
    cov_df = pd.DataFrame(cov_rows)
    mon_df = pd.DataFrame(month_rows)
    if len(mon_df):
        mon_df['expected_hours'] = mon_df['month'].map(exp_month)
        mon_df['coverage_pct'] = (100.0 * mon_df['valid_hours'] / mon_df['expected_hours']).round(2)
    loc_mon = []
    for lid, months in loc_month_hours.items():
        name = loc_df.loc[loc_df['location_id'] == lid, 'name'].iloc[0]
        for month in exp_month.index:
            n = len(months.get(month, ()))
            loc_mon.append({'location_id': lid, 'location_name': name, 'month': month,
                            'hours_with_any_pollutant': n, 'expected_hours': int(exp_month[month]),
                            'coverage_pct': round(100.0 * n / exp_month[month], 2)})
    loc_mon_df = pd.DataFrame(loc_mon)
    fail_df = pd.DataFrame(failures, columns=['sensor_id', 'month', 'error'])
    req_log = pd.DataFrame(client.request_log)

    loc_df.to_csv(os.path.join(OUT_DIR, 'locations.csv'), index=False)
    sens_df.to_csv(os.path.join(OUT_DIR, 'sensors.csv'), index=False)
    cov_df.to_csv(os.path.join(OUT_DIR, 'sensor_coverage.csv'), index=False)
    mon_df.to_csv(os.path.join(OUT_DIR, 'monthly_coverage_sensor.csv'), index=False)
    loc_mon_df.to_csv(os.path.join(OUT_DIR, 'monthly_coverage_location.csv'), index=False)
    fail_df.to_csv(os.path.join(OUT_DIR, 'failed_requests.csv'), index=False)

    write_summary(run_time, loc_df, sens_df, cov_df, loc_mon_df, fail_df, req_log)
    print(f"[OpenAQ audit] done. failed sensor-months: {len(fail_df)}. Outputs in {OUT_DIR}")


def write_summary(run_time, loc_df, sens_df, cov_df, loc_mon_df, fail_df, req_log):
    n_req = len(req_log)
    n_req_fail = int(req_log['error'].notna().sum()) if n_req else 0
    lines = [
        '# OpenAQ Audit — Chennai',
        '',
        f'- Run at (UTC): {run_time}',
        f'- Audit window: {PERIOD_START_IST:%Y-%m-%d %H:%M} to {PERIOD_END_IST:%Y-%m-%d %H:%M} IST '
        f'(hour starts; {EXPECTED_HOURS} expected hours)',
        '- Source: OpenAQ v3 `/locations` (bbox), `/locations/{id}/sensors`, `/sensors/{id}/hours`.',
        '- Hourly values are OpenAQ hourly aggregates of the provider\'s sub-hourly measurements; '
        'coverage below is computed from downloaded records, not from sensor metadata.',
        f'- HTTP requests made this run: {n_req} (failed after retries: {n_req_fail}); '
        f'sensor-months that could not be downloaded: {len(fail_df)}.',
        '',
        '## Locations',
        '',
        '| id | name | provider | lat | lon | point-in-polygon zone | metadata last (UTC) |',
        '|---|---|---|---|---|---|---|',
    ]
    for _, r in loc_df.iterrows():
        z = (f"{int(r['pip_zone_id'])} {r['pip_zone_label']}" if pd.notna(r.get('pip_zone_id'))
             else f"outside GCC ({r.get('distance_to_nearest_zone_km')} km from {r.get('nearest_zone_label')})")
        lines.append(f"| {r['location_id']} | {r['name']} | {r['provider']} | {r['latitude']:.4f} | "
                     f"{r['longitude']:.4f} | {z} | {r['metadata_last_utc']} |")
    lines += ['', '## Sensor coverage in the audit window (sensors with ≥1 valid hour)', '',
              '| location | parameter | sensor | units | valid hours | first valid (IST) | last valid (IST) | '
              'full-period % | active-period % | dup hours | neg | sentinel |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    if len(cov_df):
        c = cov_df[cov_df['valid_records'] > 0].sort_values(['location_name', 'parameter'])
        for _, r in c.iterrows():
            lines.append(
                f"| {r['location_name']} | {r['parameter']} | {r['sensor_id']} | {r['units_seen']} | "
                f"{int(r['valid_records'])} | {str(r.get('first_valid_ist'))[:16]} | {str(r.get('last_valid_ist'))[:16]} | "
                f"{r['coverage_full_period_pct']} | {r.get('coverage_active_period_pct')} | {int(r['duplicate_hours'])} | "
                f"{int(r['negative_values'])} | {int(r['sentinel_values'])} |")
        empty = cov_df[cov_df['valid_records'] == 0]
        lines += ['', f'Sensors whose metadata overlapped the window but returned **no valid hours**: '
                      f'{len(empty)} ({", ".join(str(x) for x in empty["sensor_id"].tolist()) or "none"}).']
    if len(loc_mon_df):
        lines += ['', '## Location × month coverage (hours with ≥1 valid pollutant value, %)', '']
        piv = loc_mon_df.pivot(index='location_name', columns='month', values='coverage_pct')
        lines.append('| location | ' + ' | '.join(piv.columns) + ' |')
        lines.append('|---' * (len(piv.columns) + 1) + '|')
        for name, row in piv.iterrows():
            lines.append(f'| {name} | ' + ' | '.join(f'{v:.0f}' for v in row.values) + ' |')
    lines += ['', '## Failed requests', '']
    if len(fail_df):
        for _, r in fail_df.iterrows():
            lines.append(f"- sensor {r['sensor_id']} month {r['month']}: {r['error']}")
    else:
        lines.append('None — every requested sensor-month was downloaded.')
    lines += ['', '## Caveats', '',
              '- Sensor metadata `coverage` fields are inconsistent (e.g. percentCoverage > 100%) and were not used.',
              '- Several CPCB stations expose two sensors per gas: a legacy series in µg/m³ and a newer series in ppb. '
              'They are kept separate here; no unit conversion is applied in this audit.',
              '- Location coordinates are as reported by OpenAQ. Some are inconsistent with the station name '
              '(see final_data_audit.md for the station-to-zone reconciliation).',
              '- OpenAQ hourly bins start at :30 UTC (IST hour boundaries); hours are reported in IST.']
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh-current-month', action='store_true',
                    help='Re-download the last (possibly incomplete) month even if cached')
    args = ap.parse_args()
    run(refresh_current_month=args.refresh_current_month)
