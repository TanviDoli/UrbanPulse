"""
Per-year stability checks for the 2017-2023 Open-Meteo history (before any training).

Computes, per location and calendar year: missingness; temperature and humidity distributions;
rainfall frequency, wet-hour amount, annual total and extremes; wind speed and direction; and diurnal
patterns (IST). Flags possible source/model-regime changes with leave-one-year-out z-scores and
year-over-year jumps. FLAGS ONLY - no year is ever dropped automatically.

The sealed 2024 file is never read here. The existing (already examined) 2025 data can be added as a
reference year for comparison.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TEMP, RH, RAIN, WSPD, WDIR = ('temperature_2m_c', 'relative_humidity_2m_pct', 'precipitation_mm',
                              'wind_speed_10m_kmh', 'wind_direction_10m_deg')
WET_MM = 0.1
Z_FLAG = 3.0
STATS_FOR_FLAGS = ['temp_mean', 'temp_sd', 'temp_p05', 'temp_p95', 'rh_mean', 'rh_sd', 'wet_hour_freq',
                   'mean_wet_amount', 'annual_total_mm', 'wind_mean', 'wind_sd', 'wind_dir_resultant',
                   'temp_diurnal_amplitude', 'rain_diurnal_peak_share']


def yearly_stats(df: pd.DataFrame, year_label: Optional[str] = None) -> pd.DataFrame:
    """df: columns location_key, time_utc (UTC), and the five canonical variables."""
    d = df.copy()
    d['time_utc'] = pd.to_datetime(d['time_utc'], utc=True)
    d['year'] = year_label if year_label is not None else d['time_utc'].dt.year.astype(str)
    d['ist_hour'] = d['time_utc'].dt.tz_convert('Asia/Kolkata').dt.hour
    rows = []
    for (loc, yr), g in d.groupby(['location_key', 'year']):
        t, rh, r, ws, wd = (g[c].astype(float) for c in (TEMP, RH, RAIN, WSPD, WDIR))
        wet = r >= WET_MM
        rad = np.deg2rad(wd.dropna())
        diurnal_t = g.groupby('ist_hour')[TEMP].mean()
        diurnal_w = (g.assign(w=wet).groupby('ist_hour')['w'].mean())
        rows.append({
            'location_key': loc, 'year': yr, 'hours': int(len(g)),
            **{f'missing_{c.split("_")[0]}{"_dir" if c == WDIR else ""}': float(g[c].isna().mean()) for c in (TEMP, RH, RAIN, WSPD, WDIR)},
            'temp_mean': t.mean(), 'temp_sd': t.std(), 'temp_p05': t.quantile(0.05), 'temp_p95': t.quantile(0.95),
            'rh_mean': rh.mean(), 'rh_sd': rh.std(), 'rh_p05': rh.quantile(0.05), 'rh_p95': rh.quantile(0.95),
            'wet_hour_freq': float(wet.mean()), 'mean_wet_amount': float(r[wet].mean()) if wet.any() else np.nan,
            'annual_total_mm': float(r.sum()), 'max_hourly_mm': float(r.max()),
            'p99_hourly_mm': float(r.quantile(0.99)),
            'wind_mean': ws.mean(), 'wind_sd': ws.std(), 'wind_p95': ws.quantile(0.95),
            'wind_dir_mean_deg': float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360) if len(rad) else np.nan,
            'wind_dir_resultant': float(np.hypot(np.sin(rad).mean(), np.cos(rad).mean())) if len(rad) else np.nan,
            'temp_diurnal_amplitude': float(diurnal_t.max() - diurnal_t.min()),
            'temp_peak_ist_hour': int(diurnal_t.idxmax()),
            'rain_peak_ist_hour': int(diurnal_w.idxmax()) if wet.any() else -1,
            'rain_diurnal_peak_share': float(diurnal_w.max() / diurnal_w.sum()) if diurnal_w.sum() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def flag_regime_changes(stats: pd.DataFrame, years: Optional[List[str]] = None, z: float = Z_FLAG) -> pd.DataFrame:
    """
    For each location and statistic: leave-one-year-out z-score of every year against the other years,
    and the year-over-year change relative to the spread of all year-over-year changes. Returns one row
    per raised flag (empty if none). Years outside `years` (e.g. a reference year) are compared against
    the `years` set but do not enter its reference distribution.
    """
    flags = []
    years = sorted(stats['year'].unique()) if years is None else list(years)
    for loc, g in stats.groupby('location_key'):
        g = g.set_index('year')
        for s in STATS_FOR_FLAGS:
            if s not in g:
                continue
            core = g.loc[[y for y in years if y in g.index], s].astype(float)
            for yr in g.index:
                others = core.drop(yr, errors='ignore').dropna()
                if len(others) < 3 or pd.isna(g.loc[yr, s]):
                    continue
                # floor the spread so a year that differs from otherwise identical years is still flagged
                sd = max(others.std(), 1e-6 * max(1.0, abs(others.mean())))
                zval = (g.loc[yr, s] - others.mean()) / sd
                if abs(zval) >= z:
                    flags.append({'location_key': loc, 'year': yr, 'statistic': s, 'value': float(g.loc[yr, s]),
                                  'others_mean': float(others.mean()), 'z': float(zval), 'test': 'leave_one_year_out'})
            diffs = core.sort_index().diff().dropna()
            if len(diffs) >= 3 and diffs.std() > 0:
                for yr, dv in diffs.items():
                    rest = diffs.drop(yr)
                    zj = (dv - rest.mean()) / rest.std() if len(rest) >= 2 and rest.std() > 0 else 0.0
                    if abs(zj) >= z:
                        flags.append({'location_key': loc, 'year': yr, 'statistic': s, 'value': float(dv),
                                      'others_mean': float(rest.mean()), 'z': float(zj), 'test': 'year_over_year_jump'})
    return pd.DataFrame(flags, columns=['location_key', 'year', 'statistic', 'value', 'others_mean', 'z', 'test'])


def summarize_flags(flags: pd.DataFrame, n_locations: int) -> pd.DataFrame:
    """Years flagged for the same statistic in many locations point to a source/model change rather than weather."""
    if flags.empty:
        return pd.DataFrame(columns=['year', 'statistic', 'test', 'locations_flagged', 'share_of_locations'])
    s = flags.groupby(['year', 'statistic', 'test'])['location_key'].nunique().rename('locations_flagged').reset_index()
    s['share_of_locations'] = s['locations_flagged'] / n_locations
    return s.sort_values('share_of_locations', ascending=False)


def run_stability(include_2025_reference: bool = True, out_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Development years only (the sealed 2024 file is not read)."""
    import os
    from src.collectors import open_meteo
    from src.collectors.open_meteo_history import load_history
    dev = load_history(include_sealed=False)
    stats = yearly_stats(dev)
    dev_years = sorted(stats['year'].unique())
    if include_2025_reference:
        loc = open_meteo.load_local()['hourly']
        loc = loc[loc['time_utc'].dt.year == 2025].copy()
        # the existing file's zone 1 is the ORIGINAL coastal cell -> compare with the coastal reference rows
        loc['location_key'] = loc['zone_id'].map(lambda z: 'zone_01_coastal_original' if int(z) == 1 else f'zone_{int(z):02d}')
        stats = pd.concat([stats, yearly_stats(loc, year_label='2025_ref')], ignore_index=True)
    allowed = {str(y) for y in range(2017, 2024)} | {'2025_ref'}
    if not set(stats['year']) <= allowed:                     # defensive: 2024 (sealed) must never appear here
        raise RuntimeError(f'unexpected years in stability input: {sorted(set(stats["year"]) - allowed)}')
    flags = flag_regime_changes(stats, years=dev_years)
    summary = summarize_flags(flags, stats['location_key'].nunique())
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stats.to_csv(os.path.join(out_dir, 'stability_yearly.csv'), index=False)
        flags.to_csv(os.path.join(out_dir, 'stability_flags.csv'), index=False)
        summary.to_csv(os.path.join(out_dir, 'stability_flag_summary.csv'), index=False)
    return {'stats': stats, 'flags': flags, 'summary': summary}


if __name__ == '__main__':
    import os
    res = run_stability(out_dir=os.path.join('results', 'experiments', 'history_2017_2024'))
    print(res['summary'].head(30).to_string(index=False) if len(res['summary']) else 'No stability flags raised.')
