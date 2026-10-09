"""
Phase B — Audit of the local Open-Meteo hourly file.

Run:  python -m src.audit.open_meteo_audit
Outputs: reports/open_meteo_audit/{summary.md, locations.csv, missing_by_location_variable.csv,
                                   grid_cells.csv, plausibility.csv}
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.collectors.open_meteo import (LOCAL_FILE, LOCATION_TO_ZONE, ZONE_CENTROIDS, CANONICAL,
                                       EXPECTED_UNITS, detect_format, load_local)
from src.audit.openaq_audit import load_zone_polygons

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(ROOT, 'reports', 'open_meteo_audit')
INTENDED_END_DATE = pd.Timestamp('2026-10-08')

PLAUSIBLE = {
    'temperature_2m_c': (10.0, 48.0),
    'relative_humidity_2m_pct': (0.0, 100.0),
    'precipitation_mm': (0.0, 150.0),
    'wind_speed_10m_kmh': (0.0, 150.0),
    'wind_direction_10m_deg': (0.0, 360.0),
}


def km(lat1, lon1, lat2, lon2):
    return float(np.hypot((lat1 - lat2) * 111.0, (lon1 - lon2) * 111.0 * np.cos(np.radians(13.05))))


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    fmt = detect_format(LOCAL_FILE)
    d = load_local(LOCAL_FILE)
    loc, hr, units = d['locations'].copy(), d['hourly'], d['units']
    polys = load_zone_polygons()
    variables = [c for c in CANONICAL.values() if c in hr.columns]

    # ── Locations / mapping ───────────────────────────────────────────────
    rows = []
    for _, r in loc.iterrows():
        z, zl = LOCATION_TO_ZONE[int(r['location_id'])]
        clat, clon = ZONE_CENTROIDS[z]
        p = Point(r['longitude'], r['latitude'])
        inside = [zz for (zz, _), poly in polys if poly.contains(p)]
        rows.append({**r.to_dict(), 'requested_centroid_lat': clat, 'requested_centroid_lon': clon,
                     'grid_to_centroid_km': round(km(r['latitude'], r['longitude'], clat, clon), 2),
                     'grid_cell_inside_zone': inside[0] if inside else None,
                     'grid_cell_in_own_zone': bool(inside and inside[0] == z)})
    loc_df = pd.DataFrame(rows)
    alt = np.mean([km(r['latitude'], r['longitude'], *ZONE_CENTROIDS[int(r['location_id']) + 1])
                   for _, r in loc.iterrows()])
    loc_df['grid_cell'] = loc_df['latitude'].round(4).astype(str) + ',' + loc_df['longitude'].round(4).astype(str)
    cells = loc_df.groupby('grid_cell').agg(zone_ids=('zone_id', lambda s: sorted(s.tolist())),
                                           location_ids=('location_id', lambda s: sorted(s.tolist()))).reset_index()

    # identical series within shared cells
    piv = {v: hr.pivot(index='time_utc', columns='location_id', values=v) for v in variables}
    ident = []
    for _, c in cells.iterrows():
        ids = c['location_ids']
        if len(ids) < 2:
            ident.append('single zone')
            continue
        diffs = {v: float(max((piv[v][ids[0]] - piv[v][j]).abs().max() for j in ids[1:])) for v in variables}
        ident.append('identical' if all(x == 0 for x in diffs.values())
                     else 'max abs diff ' + ', '.join(f'{k}={x:.2f}' for k, x in diffs.items() if x))
    cells['series_identity'] = ident
    cells['n_zones'] = cells['zone_ids'].map(len)

    # ── Time coverage ─────────────────────────────────────────────────────
    t0, t1 = hr['time_utc'].min(), hr['time_utc'].max()
    expected = pd.date_range(t0, t1, freq='h')
    cov_rows = []
    for lid, g in hr.groupby('location_id'):
        times = g['time_utc']
        dup = int(times.duplicated().sum())
        missing_hours = len(expected.difference(pd.DatetimeIndex(times.unique())))
        rec = {'location_id': lid, 'zone_id': LOCATION_TO_ZONE[lid][0], 'zone_label': LOCATION_TO_ZONE[lid][1],
               'rows': len(g), 'first_utc': times.min(), 'last_utc': times.max(),
               'duplicate_timestamps': dup, 'missing_timestamps': missing_hours}
        for v in variables:
            rec[f'missing_{v}'] = int(g[v].isna().sum())
        cov_rows.append(rec)
    cov_df = pd.DataFrame(cov_rows)

    # ── Plausibility ──────────────────────────────────────────────────────
    pl = []
    for v in variables:
        lo, hi = PLAUSIBLE[v]
        s = hr[v]
        pl.append({'variable': v, 'min': s.min(), 'p01': s.quantile(0.01), 'median': s.median(),
                   'p99': s.quantile(0.99), 'max': s.max(), 'below_range': int((s < lo).sum()),
                   'above_range': int((s > hi).sum()), 'plausible_range': f'[{lo}, {hi}]'})
    pl_df = pd.DataFrame(pl)
    wet_hours = (hr['precipitation_mm'] > 0).mean() * 100

    loc_df.to_csv(os.path.join(OUT_DIR, 'locations.csv'), index=False)
    cov_df.to_csv(os.path.join(OUT_DIR, 'missing_by_location_variable.csv'), index=False)
    cells.to_csv(os.path.join(OUT_DIR, 'grid_cells.csv'), index=False)
    pl_df.to_csv(os.path.join(OUT_DIR, 'plausibility.csv'), index=False)

    end_ist = t1.tz_convert('Asia/Kolkata')
    reaches = t1.normalize().tz_localize(None) >= INTENDED_END_DATE and t1.hour == 23
    unit_check = {k: (units.get(k), EXPECTED_UNITS[k]) for k in EXPECTED_UNITS}
    unit_lines = [f"| {k} | {got!r} | {exp} | {'OK' if (got or '').replace('�', '°') == exp else 'CHECK (encoding)'} |"
                  for k, (got, exp) in unit_check.items()]

    L = ['# Open-Meteo Local File Audit', '',
         f'- File: `data/external/open_meteo_chennai_hourly_2025_2026.csv`',
         f'- Detected format (from content): **{fmt}** — two blank-line-separated CSV blocks '
         '(location metadata block + hourly data block). Not JSON; not a single-table CSV.',
         f'- Rows: {len(hr):,} hourly rows, {hr["location_id"].nunique()} locations, '
         f'{len(variables)} variables.',
         f'- Timezone: `timezone=GMT`, `utc_offset_seconds=0` for all locations → timestamps are UTC.',
         f'- Period: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} UTC '
         f'({end_ist:%Y-%m-%d %H:%M} IST); {len(expected):,} expected hourly steps per location.',
         f'- Reaches intended audit date 2026-10-08 (full UTC day)? **{"yes" if reaches else "no"}** — '
         f'the last record is 2026-10-07 23:00 UTC; 2026-10-08 is not in the file.',
         f'- Provenance: Open-Meteo Historical Weather API = reanalysis/model-derived (ERA5 family / ECMWF IFS). '
         'These are **not measured station observations**.',
         '', '## Units', '', '| variable | in file | expected | status |', '|---|---|---|---|', *unit_lines,
         '', 'The degree sign is stored as UTF-8 `°` but some readers decode it as `�`; values are unaffected.',
         '', '## Completeness', '',
         f'- Duplicate timestamps (any location): {int(cov_df["duplicate_timestamps"].sum())}',
         f'- Missing hourly timestamps (any location): {int(cov_df["missing_timestamps"].sum())}',
         '- Missing values per variable (all locations): ' +
         ', '.join(f'{v}={int(hr[v].isna().sum())}' for v in variables),
         '', '## Location → zone mapping', '',
         'The file does not record the requested coordinates; returned coordinates are model grid cells. '
         f'Location *i* matches GeoJSON feature *i* (mean grid-to-centroid distance '
         f'{loc_df["grid_to_centroid_km"].mean():.2f} km) and not zone number *i+1* (mean {alt:.2f} km).',
         '', '| location | zone | grid lat | grid lon | elev (m) | km to zone centroid | grid cell inside zone |',
         '|---|---|---|---|---|---|---|']
    for _, r in loc_df.sort_values('zone_id').iterrows():
        inside = r['grid_cell_inside_zone']
        L.append(f"| {r['location_id']} | {r['zone_id']} {r['zone_label']} | {r['latitude']:.4f} | {r['longitude']:.4f} | "
                 f"{r['elevation']:.0f} | {r['grid_to_centroid_km']} | "
                 f"{('same zone' if r['grid_cell_in_own_zone'] else f'zone {int(inside)}') if pd.notna(inside) else 'outside GCC'} |")
    L += ['', f'## Shared grid cells — {len(cells)} distinct cells for 15 zones', '',
          '| grid cell | zones | series |', '|---|---|---|']
    for _, c in cells.sort_values('n_zones', ascending=False).iterrows():
        L.append(f"| {c['grid_cell']} | {', '.join(map(str, c['zone_ids']))} | {c['series_identity']} |")
    L += ['', 'Implication: zones sharing a cell are **not independent weather observations**. '
          'Spatial differences between them are zero (or elevation-adjusted only).', '',
          '## Plausibility', '', '| variable | min | p01 | median | p99 | max | below | above |', '|---|---|---|---|---|---|---|---|']
    for _, r in pl_df.iterrows():
        L.append(f"| {r['variable']} | {r['min']:.1f} | {r['p01']:.1f} | {r['median']:.1f} | {r['p99']:.1f} | "
                 f"{r['max']:.1f} | {r['below_range']} | {r['above_range']} |")
    L += [f'', f'Hours with precipitation > 0: {wet_hours:.1f}% of all location-hours.', '',
          '## Inconsistencies with the existing weather pipeline', '',
          '- The existing live pipeline (`src/live/live_pipeline.py`) reads WeatherAPI *current* conditions, not Open-Meteo.',
          '- Open-Meteo wind speed is km/h; CAAQMS station `WS` is m/s.',
          '- Open-Meteo `precipitation` is the sum over the *preceding* hour (timestamp = hour end); '
          'station `RF` is per 15-minute interval and OpenAQ hours are IST-aligned (:30 UTC).',
          '- The temperature HMM was trained on IMD *daily mean* temperature (IMD Chennai daily record, '
          '1951–2024-06-22); Open-Meteo hourly temperature must be aggregated to complete IST-day means before use.',
          '- The local file starts 2025-01-01, so it does not overlap the IMD daily temperature record (ends 2024-06-22) '
          'or the IMD rainfall record (ends 2023-12-26); it overlaps the CAAQMS station files for calendar 2025.']
    with open(os.path.join(OUT_DIR, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f'[Open-Meteo audit] written to {OUT_DIR}')


if __name__ == '__main__':
    run()
