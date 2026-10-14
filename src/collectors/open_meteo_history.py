"""
One-time Open-Meteo archive history (2017-2024) for the UrbanPulse 6 h / 24 h experiment.
ISOLATED: not imported by the live pipeline; writes only the files below.

Source: Open-Meteo Historical Weather API, model pinned to `ecmwf_ifs` (preliminary checks showed the
default `archive_best_match` equals ECMWF IFS exactly for both 2017-01 and 2025-01, same grid).

Locations (16): the 15 current zone centroids used by the live collector (default cell selection, as
live), plus a separately labelled reference location at the ORIGINAL Thiruvottiyur coastal grid cell of the
existing 2025-2026 file, requested with cell_selection=nearest (Open-Meteo's default 'land' selection is the
likely reason zone 1 now snaps inland; cell_selection applies per request, so the reference location is
fetched in its own request covering 2017-2024):
  zone_01                   request (13.1785, 80.3015) -> currently snapped to cell 13.1810 N, 80.2370 E
                            (the cell of zones 2 and 3; also what the live scheduler receives)
  zone_01_coastal_original  request (13.251318, 80.31022) -> expected cell 13.2513 N, 80.3102 E
                            (the zone-1 cell in data/external/open_meteo_chennai_hourly_2025_2026.csv)
The two zone-1 representations are kept as separate rows (location_key); nothing is merged.

Sealing: 2024 rows are written to a SEPARATE file and are never loaded unless load_history(
include_sealed=True, reason=...) is called with URBANPULSE_UNSEAL_2024 set (every access is logged).
Only structural checks (row counts, missingness, units, coordinates) are computed for 2024 at fetch time.
"""

import email.utils
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from src.collectors.open_meteo import ARCHIVE_URL, CANONICAL, EXPECTED_UNITS, HOURLY_VARS, LOCATION_TO_ZONE, ZONE_CENTROIDS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY_DEV_FILE = os.path.join(ROOT, 'data', 'external', 'open_meteo_chennai_hourly_2017_2023.csv.gz')
HISTORY_SEALED_FILE = os.path.join(ROOT, 'data', 'external', 'open_meteo_chennai_hourly_2024_SEALED.csv.gz')
HISTORY_MANIFEST = os.path.join(ROOT, 'data', 'external', 'open_meteo_chennai_hourly_2017_2024.manifest.json')
SEALED_ACCESS_LOG = os.path.join(ROOT, 'results', 'experiments', 'history_2017_2024', 'sealed_access_log.jsonl')
DEV_YEARS = list(range(2017, 2024))
SEALED_YEAR = 2024
SOURCE_MODEL = 'ecmwf_ifs'
UNSEAL_ENV = 'URBANPULSE_UNSEAL_2024'
UNSEAL_TOKEN = 'FINAL_EVALUATION_CONFIG_FROZEN'
# rate-limit handling (per request): 1 try + up to 5 retries; backoff 15, 30, 60, 120, 240 s (equal jitter,
# so each wait is 50-100 % of that), Retry-After honoured up to 15 min. Worst case ~7.5 min per request.
REQUEST_PAUSE_S = 60.0                  # pause between successful archive requests (minute rate limit)
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 15.0
BACKOFF_MAX_S = 240.0
RETRY_AFTER_MAX_S = 900.0

ZONE_LABEL = {z: lbl for z, lbl in LOCATION_TO_ZONE.values()}
ZONE1_CURRENT_CELL = (13.181019, 80.237015)
ZONE1_COASTAL = {'location_key': 'zone_01_coastal_original', 'zone_id': 1,
                 'label': 'Thiruvottiyur (original coastal cell, reference only)',
                 'lat': 13.251318, 'lon': 80.31022, 'expected_cell': (13.251318, 80.31022), 'is_reference': True,
                 'cell_selection': 'nearest'}


def current_locations() -> List[Dict[str, Any]]:
    """The 15 live-collector zone centroids (default cell selection, exactly as the live pipeline)."""
    return [{'location_key': f'zone_{z:02d}', 'zone_id': z, 'label': ZONE_LABEL[z], 'lat': ZONE_CENTROIDS[z][0],
             'lon': ZONE_CENTROIDS[z][1], 'is_reference': False, 'cell_selection': None} for z in sorted(ZONE_CENTROIDS)]


def history_locations() -> List[Dict[str, Any]]:
    return current_locations() + [dict(ZONE1_COASTAL)]


def _series(hourly: Dict[str, Any], var: str) -> Optional[list]:
    """Value list for `var`, accepting plain or model-suffixed keys."""
    if var in hourly:
        return hourly[var]
    for k in (f'{var}_{SOURCE_MODEL}', f'{var}_archive_best_match'):
        if k in hourly:
            return hourly[k]
    return None


def _parse(items: List[Dict[str, Any]], locs: List[Dict[str, Any]], start: str, end: str) -> pd.DataFrame:
    if len(items) != len(locs):
        raise ValueError(f'Open-Meteo returned {len(items)} locations, expected {len(locs)}')
    expected = pd.date_range(pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC') + pd.Timedelta(hours=23), freq='h')
    frames = []
    for loc, it in zip(locs, items):
        if it.get('utc_offset_seconds') != 0:
            raise ValueError(f"non-UTC response for {loc['location_key']}")
        units = it.get('hourly_units') or {}
        h = it['hourly']
        t = pd.to_datetime(h['time'], utc=True)
        if len(t) != len(expected) or not (t == expected).all():
            raise ValueError(f"time axis for {loc['location_key']} is not the complete hourly UTC range {start}..{end}")
        rec = {'time_utc': t}
        for var in HOURLY_VARS:
            vals = _series(h, var)
            if vals is None:
                raise ValueError(f"missing variable {var} for {loc['location_key']}")
            unit = units.get(var, units.get(f'{var}_{SOURCE_MODEL}', units.get(f'{var}_archive_best_match')))
            if unit != EXPECTED_UNITS[var]:
                raise ValueError(f'unexpected unit for {var}: {unit} (expected {EXPECTED_UNITS[var]})')
            rec[CANONICAL[var]] = np.asarray(vals, dtype=float)
        f = pd.DataFrame(rec)
        f['location_key'], f['zone_id'], f['is_reference_location'] = loc['location_key'], loc['zone_id'], loc['is_reference']
        f['request_latitude'], f['request_longitude'] = loc['lat'], loc['lon']
        f['grid_latitude'], f['grid_longitude'], f['grid_elevation'] = it.get('latitude'), it.get('longitude'), it.get('elevation')
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df['source'], df['model'], df['provenance'] = 'open_meteo_archive_api', SOURCE_MODEL, 'reanalysis/model analysis'
    return df


def _request(locs, start, end, session=None, timeout_s=180.0) -> List[Dict[str, Any]]:
    selections = {l.get('cell_selection') for l in locs}
    if len(selections) != 1:
        raise ValueError('cell_selection must be identical for all locations of one request')
    params = {'latitude': ','.join(str(l['lat']) for l in locs), 'longitude': ','.join(str(l['lon']) for l in locs),
              'start_date': start, 'end_date': end, 'hourly': ','.join(HOURLY_VARS), 'timezone': 'GMT',
              'models': SOURCE_MODEL}
    sel = selections.pop()
    if sel:
        params['cell_selection'] = sel
    resp = _get_with_retry(session or requests.Session(), params, timeout_s)
    payload = resp.json()
    return payload if isinstance(payload, list) else [payload]


class RetriesExhausted(RuntimeError):
    """Rate limit / transient server errors persisted; the build aborts before anything is written."""


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """Retry-After as delta-seconds or an HTTP date; None if absent or unparseable."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, (when - pd.Timestamp.now(tz='UTC').to_pydatetime()).total_seconds())
    except (TypeError, ValueError):
        return None


def _get_with_retry(session, params, timeout_s):
    """
    GET with bounded retries for HTTP 429, transient 5xx (500/502/503/504) and connection errors/timeouts.
    Waits Retry-After when the server sends it (refusing waits longer than RETRY_AFTER_MAX_S), otherwise
    exponential backoff with jitter. Other 4xx (400, 404, ...) fail immediately without retry.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = session.get(ARCHIVE_URL, params=params, timeout=timeout_s)
        except (requests.ConnectionError, requests.Timeout) as e:
            resp, err = None, f'{type(e).__name__}'
        else:
            if resp.status_code not in RETRY_STATUS:
                resp.raise_for_status()                   # permanent client errors raise here, no retry
                return resp
            err = f'HTTP {resp.status_code}'
        if attempt == MAX_ATTEMPTS:
            break
        backoff = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
        wait = backoff / 2 + random.uniform(0, backoff / 2)          # exponential backoff, equal jitter
        ra = _retry_after_seconds(resp.headers.get('Retry-After')) if resp is not None else None
        if ra is not None:
            if ra > RETRY_AFTER_MAX_S:
                raise RetriesExhausted(f'{err}: server asks to wait {ra:.0f} s (> {RETRY_AFTER_MAX_S} s) for '
                                       f"{params['start_date']}..{params['end_date']}; aborting, nothing written")
            wait = ra + random.uniform(0, 1.0)                        # never earlier than the server asked
        print(f"  {err} for {params['start_date']}..{params['end_date']} - retry {attempt}/{MAX_ATTEMPTS - 1} "
              f'in {wait:.0f} s')
        time.sleep(wait)
    raise RetriesExhausted(f"{err} persisted after {MAX_ATTEMPTS} attempts for {params['start_date']}.."
                           f"{params['end_date']}; aborting, nothing written")


def fetch_year(year: int, session=None) -> pd.DataFrame:
    """The 15 current zone locations for one calendar year (one HTTP request)."""
    locs = current_locations()
    start, end = f'{year}-01-01', f'{year}-12-31'
    return _parse(_request(locs, start, end, session), locs, start, end)


def fetch_reference(start: str, end: str, session=None) -> pd.DataFrame:
    """The coastal zone-1 reference location (cell_selection=nearest) for start..end (one HTTP request)."""
    loc = [dict(ZONE1_COASTAL)]
    return _parse(_request(loc, start, end, session), loc, start, end)


def verify_coastal_reference(session=None, local_file_hourly: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """ONE small request (reference location only, January 2025): does the coastal request reproduce the
    zone-1 series and cell of the existing 2025-2026 file? Read-only; writes nothing."""
    from src.collectors import open_meteo
    df = fetch_reference('2025-01-01', '2025-01-31', session)
    local = local_file_hourly if local_file_hourly is not None else open_meteo.load_local()['hourly']
    z1 = local[local['zone_id'] == 1].set_index('time_utc').reindex(df['time_utc'])
    diffs = {CANONICAL[v]: float(np.nanmax(np.abs(df[CANONICAL[v]].to_numpy() - z1[CANONICAL[v]].to_numpy())))
             for v in HOURLY_VARS}
    cell = (float(df['grid_latitude'].iloc[0]), float(df['grid_longitude'].iloc[0]))
    ok_cell = bool(np.allclose(cell, ZONE1_COASTAL['expected_cell'], atol=1e-4))
    return {'returned_cell': cell, 'expected_cell': ZONE1_COASTAL['expected_cell'], 'cell_matches': ok_cell,
            'max_abs_diff_vs_local_zone1': diffs, 'passed': ok_cell and max(diffs.values()) == 0.0}


def structural_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Structure only (safe for the sealed year): rows, hours, missingness, coordinates. No value statistics."""
    out = {'rows': int(len(df)), 'hours_per_location': sorted(df.groupby('location_key').size().unique().tolist()),
           'missing_share': {c: float(df[c].isna().mean()) for c in CANONICAL.values()},
           'grid_cells': {k: [float(g['grid_latitude'].iloc[0]), float(g['grid_longitude'].iloc[0])]
                          for k, g in df.groupby('location_key')}}
    return out


def _sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check_outputs_absent() -> None:
    """Refuse up front (before any request) if any output or a leftover .tmp exists; nothing is touched."""
    for p in (HISTORY_DEV_FILE, HISTORY_SEALED_FILE, HISTORY_MANIFEST):
        for q in (p, p + '.tmp'):
            if os.path.exists(q):
                raise FileExistsError(f'{q} already exists - refusing to overwrite')


def _publish_all(writers: List[Any]) -> None:
    """
    Two-phase commit for several files. writers: [(path, write_fn(tmp_path)), ...] in publish order (the
    manifest last, as the commit marker). Phase 1 writes every <path>.tmp; phase 2 publishes them with
    os.replace. On ANY failure: files published by this call are removed (they did not exist before - the
    caller refuses pre-existing outputs) and every temp file created by this call is deleted. Pre-existing
    files, including stray .tmp files, are never touched: their presence aborts before anything is written.
    """
    for path, _ in writers:
        if os.path.exists(path) or os.path.exists(path + '.tmp'):
            raise FileExistsError(f'{path} (or its .tmp) already exists - refusing to overwrite')
    created, published = [], []
    try:
        for path, write in writers:
            tmp = path + '.tmp'
            created.append(tmp)
            write(tmp)
        for path, _ in writers:
            if os.path.exists(path):
                raise FileExistsError(f'{path} appeared during the build - aborting')
        for path, _ in writers:
            os.replace(path + '.tmp', path)
            published.append(path)
    except BaseException:
        for p in published:
            if os.path.exists(p):
                os.remove(p)
        raise
    finally:
        for tmp in created:
            if os.path.exists(tmp):
                os.remove(tmp)


def build_history(session=None, sleep_s: Optional[float] = None, coastal_check: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetch 2017..2024: 8 yearly requests for the 15 current locations + 1 request for the coastal reference
    location over 2017-01-01..2024-12-31. Dev years -> HISTORY_DEV_FILE, 2024 -> HISTORY_SEALED_FILE.
    Refuses to overwrite existing outputs; aborts (writing nothing) on any validation error.
    """
    _check_outputs_absent()
    sleep_s = REQUEST_PAUSE_S if sleep_s is None else sleep_s
    years = DEV_YEARS + [SEALED_YEAR]
    parts = []
    for i, year in enumerate(years):
        if i:
            time.sleep(sleep_s)
        parts.append(fetch_year(year, session))
    time.sleep(sleep_s)
    parts.append(fetch_reference(f'{years[0]}-01-01', f'{years[-1]}-12-31', session))
    allrows = pd.concat(parts, ignore_index=True)
    yr = allrows['time_utc'].dt.year
    dev = [allrows[yr != SEALED_YEAR]]
    sealed = allrows[yr == SEALED_YEAR]
    structure = {str(y): structural_summary(allrows[yr == y]) for y in years}
    os.makedirs(os.path.dirname(HISTORY_DEV_FILE), exist_ok=True)
    dev_df = pd.concat(dev, ignore_index=True)
    if dev_df['time_utc'].dt.year.max() >= SEALED_YEAR or sealed['time_utc'].dt.year.nunique() != 1:
        raise RuntimeError('sealed-year rows are not cleanly separated')
    hashes: Dict[str, str] = {}

    def write_gz(df, key):
        def w(tmp):
            df.to_csv(tmp, index=False, compression='gzip')
            hashes[key] = _sha256(tmp)                     # same bytes that will be published
        return w

    def write_manifest(tmp):
        man = _manifest(structure, coastal_check, hashes)
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(man, f, indent=2, default=str)
    _publish_all([(HISTORY_DEV_FILE, write_gz(dev_df, 'dev')), (HISTORY_SEALED_FILE, write_gz(sealed, 'sealed')),
                  (HISTORY_MANIFEST, write_manifest)])
    with open(HISTORY_MANIFEST, encoding='utf-8') as f:
        return json.load(f)


def _manifest(structure: Dict[str, Any], coastal_check: Optional[Dict[str, Any]], hashes: Dict[str, str]) -> Dict[str, Any]:
    return {'created_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'archive_url': ARCHIVE_URL,
            'model': SOURCE_MODEL, 'hourly_variables': HOURLY_VARS, 'timezone': 'GMT',
            'locations': history_locations(), 'zone1_current_cell': ZONE1_CURRENT_CELL,
            'dev_years': DEV_YEARS, 'sealed_year': SEALED_YEAR,
            'files': {'dev': {'path': os.path.relpath(HISTORY_DEV_FILE, ROOT), 'sha256': hashes['dev']},
                      'sealed': {'path': os.path.relpath(HISTORY_SEALED_FILE, ROOT), 'sha256': hashes['sealed'],
                                 'note': 'not loaded until the final evaluation; structural checks only'}},
            'structure_by_year': structure, 'coastal_reference_check': coastal_check,
            'preliminary_checks': 'archive_best_match == ecmwf_ifs exactly (2017-01-01..07 and 2025-01); zones 2-15 '
                                  'reproduce the local 2025 file exactly; zone 1 snaps to the zone 2/3 cell'}


def load_history(include_sealed: bool = False, reason: Optional[str] = None) -> pd.DataFrame:
    """Development history (2017-2023). The sealed 2024 file is added only for the final evaluation:
    include_sealed=True, a non-empty reason, and env URBANPULSE_UNSEAL_2024=FINAL_EVALUATION_CONFIG_FROZEN.
    Every sealed access is appended to SEALED_ACCESS_LOG. Data count as published only when the manifest
    (written last by build_history) exists and the file hashes match it."""
    if not os.path.exists(HISTORY_MANIFEST):
        raise FileNotFoundError('history manifest missing - dataset not (completely) published')
    with open(HISTORY_MANIFEST, encoding='utf-8') as f:
        man = json.load(f)
    if _sha256(HISTORY_DEV_FILE) != man['files']['dev']['sha256']:
        raise RuntimeError('development file does not match its manifest hash')
    df = pd.read_csv(HISTORY_DEV_FILE, parse_dates=['time_utc'])
    if df['time_utc'].dt.year.max() >= SEALED_YEAR:
        raise RuntimeError('development file contains sealed-year rows')
    if not include_sealed:
        return df
    if not reason or os.environ.get(UNSEAL_ENV) != UNSEAL_TOKEN:
        raise PermissionError(f'2024 is sealed: set {UNSEAL_ENV}={UNSEAL_TOKEN} and give a reason (final evaluation only)')
    sealed_sha = _sha256(HISTORY_SEALED_FILE)
    os.makedirs(os.path.dirname(SEALED_ACCESS_LOG), exist_ok=True)
    with open(SEALED_ACCESS_LOG, 'a', encoding='utf-8') as f:            # every attempt is logged
        f.write(json.dumps({'accessed_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'reason': reason,
                            'sealed_sha256': sealed_sha,
                            'matches_manifest': sealed_sha == man['files']['sealed']['sha256']}) + '\n')
    if sealed_sha != man['files']['sealed']['sha256']:
        raise RuntimeError('sealed file does not match its manifest hash')
    return pd.concat([df, pd.read_csv(HISTORY_SEALED_FILE, parse_dates=['time_utc'])], ignore_index=True)


def main(argv=None, session=None, local_file_hourly: Optional[pd.DataFrame] = None) -> int:
    """
    --verify-coastal : request 1 only (coastal reference, January 2025).
    --build          : request 1 (coastal check; abort with nothing written if it fails), then requests 2-9
                       (yearly 2017..2024, 15 current locations), then request 10 (coastal reference 2017-2024).
    """
    import argparse
    ap = argparse.ArgumentParser(description='One-time Open-Meteo history (network; requires approval)')
    ap.add_argument('--verify-coastal', action='store_true', help='ONE request: reference location, Jan 2025')
    ap.add_argument('--build', action='store_true',
                    help='10 requests: coastal check, 8 yearly (15 current locations), coastal reference 2017-2024')
    a = ap.parse_args(argv)
    if a.verify_coastal:
        print(json.dumps(verify_coastal_reference(session, local_file_hourly), indent=2, default=str))
        return 0
    if a.build:
        _check_outputs_absent()                                  # before request 1
        chk = verify_coastal_reference(session, local_file_hourly)
        print('coastal reference check:', json.dumps(chk, default=str))
        if not chk['passed']:
            print('Coastal reference check failed - nothing downloaded or written (report for review).')
            return 2
        man = build_history(session=session, coastal_check=chk)
        print(json.dumps(man['structure_by_year'], indent=1, default=str)[:2000])
        return 0
    ap.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
