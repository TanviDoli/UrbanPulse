"""
Promotion package for the frozen long-horizon temperature candidates (+6 h, +24 h; T4_lookup_clim_full).

- Models: taken unchanged from models/experiments/long_horizon_history/refit_2017_2023_candidates.joblib
  (selected on 2019-2022, validated on 2023, refit on 2017-2023, passed the frozen 2024 evaluation). No refit.
- Lookup tables (clim_target, clim_origin): built only from 2017-2023 TRAINING rows of the development history,
  with the experiment's levels and min_count (= what every evaluation row received; no leave-one-out at lookup).
  2024 and 2025-26 data are never read.
- Zone mapping: history grid cells gc1..gc8; every live zone maps to the grid cell of its own live request.
- Writes ONLY the new artifact models/temperature/temp_hourly_history_6h_24h.joblib and a reference-row CSV.
  Production bundles and prediction files are never touched; the live forecaster uses this package only when
  explicitly enabled (see live_forecaster.TEMP_HISTORY_ENV).

  python -m src.models.package_history_models            # build (refuses to overwrite an existing package)
  python -m src.models.package_history_models --replace  # rebuild the package file only
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.models import long_horizon_history as lh
from src.models import train_and_validate_hmm as tv
from src.models.encoding import fit_table

ROOT = lh.ROOT
ARTIFACT = os.path.join('models', 'temperature', 'temp_hourly_history_6h_24h.joblib')
REFERENCE_CSV = os.path.join('results', 'experiments', 'long_horizon_history', 'promotion_temp_6h_24h',
                             'reference_rows_2023.csv')
FROZEN_ARTIFACT = lh.ARTIFACT
FINAL_RESULTS = os.path.join('results', 'experiments', 'final_eval_2024', 'final_results.json')
PROTOCOL_LOCK = os.path.join('results', 'experiments', 'final_eval_2024', 'protocol_lock.json')
VARIANT = 'T4_lookup_clim_full'
HORIZONS = (6, 24)
TRAIN = ('2017-01-01 00:00', '2023-12-31 23:00')
REFERENCE_PERIOD = ('2023-01-01', '2023-12-31')
REFERENCE_ROWS_PER_CELL = 40
ZONE1_NOTE = ('Live zone 1 maps to gc2: the live collector requests the zone-1 centroid with default cell '
              'selection, which returns the inland cell 13.1810 N, 80.2370 E shared with zones 2 and 3 - i.e. this '
              'follows the inland live-input cell. The coastal reference cell gc1 (13.2513 N, 80.3102 E) receives no '
              'live input; its tables are kept but unused. Existing production bundles keep their own mapping.')


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _sha256(p: str) -> str:
    with open(_abs(p), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _raw_required() -> List[str]:
    from src.models.temp_model import TEMP_CANDIDATE_EXTRA, TEMP_FEATURES
    return [f for f in TEMP_FEATURES + TEMP_CANDIDATE_EXTRA if f not in ('clim_target', 'anom_now')]


def history_frames(dev: Optional[pd.DataFrame] = None):
    """Development history only (manifest-verified, 2017-2023) -> grid cells and per-cell frames."""
    from src.collectors.open_meteo_history import load_history
    dev = load_history(include_sealed=False) if dev is None else dev
    if dev['time_utc'].dt.year.max() >= 2024 or dev['time_utc'].dt.year.min() < 2017:
        raise RuntimeError('package input must be 2017-2023 development data only')
    cells = lh.grid_cells(dev)
    return dev, cells, lh.cell_frames(cells, dev)                         # cell_frames asserts no 2024 values


def table_frame(frames: Dict[str, pd.DataFrame], h: int, splits=None) -> pd.DataFrame:
    """The frame on which the experiment computed its encodings (temp_rows output before the final dropna)."""
    series = {c: f['temperature_2m_c'] for c, f in frames.items()}
    return tv.temp_rows(series, h, weather=frames, required=_raw_required(), splits=splits or {'train': TRAIN})


def build_tables(df: pd.DataFrame) -> Dict[str, Any]:
    """Training-only tables, exact levels and min_count of the experiment (train_encoding / fit_table)."""
    return {'clim_target': fit_table(df, df['y'].to_numpy(), tv.TEMP_CLIM_TARGET_LEVELS, min_count=tv.ENCODING_MIN_COUNT),
            'clim_origin': fit_table(df, df['t_now'].to_numpy(), tv.TEMP_CLIM_ORIGIN_LEVELS, min_count=tv.ENCODING_MIN_COUNT)}


def zone_mapping(cells: pd.DataFrame) -> Dict[str, Any]:
    """Live zone -> history grid cell (cell of the zone's own live request), and cell -> representative live zone
    (the zone whose series represents that cell in the experiment)."""
    z2c: Dict[int, str] = {}
    for _, c in cells.iterrows():
        for loc in str(c['locations']).split(','):
            if loc.startswith('zone_') and loc[5:].isdigit():
                z2c[int(loc[5:])] = c['cell']
    rep = {}
    for _, c in cells.iterrows():
        key = c['rep_location_key']
        if key.startswith('zone_') and key[5:].isdigit() and c['cell'] in z2c.values():
            rep[c['cell']] = int(key[5:])
    if sorted(z2c) != list(range(1, 16)):
        raise RuntimeError(f'zone mapping incomplete: {sorted(z2c)}')
    if any(z2c[z] != c for c, z in rep.items()):
        raise RuntimeError('a representative zone does not lie in its own cell')
    return {'zone_to_cell': z2c, 'rep_zone': rep}


def reference_rows(df: pd.DataFrame, model, h: int, rep_zone: Dict[str, int], n_per_cell: int = REFERENCE_ROWS_PER_CELL) -> pd.DataFrame:
    """Evenly spaced 2023 origins per live-mapped cell, with features computed the way every evaluation row was:
    raw features from temp_rows and NON-leave-one-out training means (train_encoding(loo=False))."""
    df = df.copy()
    df['clim_target'] = tv.train_encoding(df, df['y'].to_numpy(), tv.TEMP_CLIM_TARGET_LEVELS, loo=False)
    df['anom_now'] = df['t_now'] - tv.train_encoding(df, df['t_now'].to_numpy(), tv.TEMP_CLIM_ORIGIN_LEVELS, loo=False)
    lo, hi = (pd.Timestamp(t, tz='UTC') for t in REFERENCE_PERIOD)
    feats = list(model.features)
    parts = []
    for cell in sorted(rep_zone):
        s = df[(df['cell'] == cell) & (df.index >= lo) & (df.index <= hi + pd.Timedelta(hours=23))].dropna(subset=feats)
        idx = np.unique(np.linspace(0, len(s) - 1, min(n_per_cell, len(s))).astype(int))
        parts.append(s.iloc[idx])
    ref = pd.concat(parts)
    out = ref[['cell'] + feats].copy()
    out.insert(0, 'horizon', f'{h}h')
    out.insert(1, 'origin_utc', [t.isoformat() for t in ref.index])
    out['rep_zone'] = out['cell'].map(rep_zone)
    out['pred_temp_c'] = model.predict(ref[feats])
    return out.reset_index(drop=True)


def _gate(results: Dict[str, Any], key: str, lh_summary: pd.DataFrame) -> Dict[str, Any]:
    r = results['results'][f'temperature_{key}']
    s = lh_summary[(lh_summary['variable'] == 'temperature') & (lh_summary['horizon'] == key) & (lh_summary['role'] == 'selected')].iloc[0]
    return {'passed': bool(r['gate']['passed']), 'source': 'final_eval_2024 (frozen protocol v1)',
            'test_skill': r['summary']['skill'], 'test_ci_week': [r['summary']['ci_week_low'], r['summary']['ci_week_high']],
            'test_best_baseline': r['summary']['best_baseline'], 'test_period': '2024',
            'validate_skill': float(s['val_skill']), 'validate_period': '2023',
            'secondary_2025_26_skill': float(s['rob_skill']), 'checks_2024': r['gate']}


def build_package(dev: Optional[pd.DataFrame] = None, save: bool = True, replace: bool = False) -> Dict[str, Any]:
    from src.models.temp_model import TEMP_CANDIDATE_EXTRA, TEMP_FEATURES, HourlyTempModel
    if save and os.path.exists(_abs(ARTIFACT)) and not replace:
        raise FileExistsError(f'{ARTIFACT} exists - pass replace=True (--replace) to rebuild the package file')
    lock = json.load(open(_abs(PROTOCOL_LOCK), encoding='utf-8'))
    frozen_sha = _sha256(FROZEN_ARTIFACT)
    if lock['sha256']['artifact_long_horizon_history'] != frozen_sha:
        raise RuntimeError('frozen artifact differs from the one locked by the final-evaluation protocol')
    results = json.load(open(_abs(FINAL_RESULTS), encoding='utf-8'))
    if results['lock']['sha256']['protocol'] != lock['sha256']['protocol']:
        raise RuntimeError('final results were not produced under the locked protocol')
    frozen = joblib.load(_abs(FROZEN_ARTIFACT))
    lh_summary = pd.read_csv(_abs(lh.SUMMARY_CSV), float_precision='round_trip')
    dev, cells, frames = history_frames(dev)
    mapping = zone_mapping(cells)
    models, encodings, gates, refs, counts = {}, {}, {}, [], {}
    for h in HORIZONS:
        key = f'{h}h'
        if frozen['selections'][('temperature', key)]['selected'] != VARIANT:
            raise RuntimeError(f'BLOCKER: frozen temperature {key} is not {VARIANT}')
        m = frozen['models'][('temperature', key)]
        if not isinstance(m, HourlyTempModel) or list(m.features) != TEMP_FEATURES + TEMP_CANDIDATE_EXTRA or m.horizon_h != h:
            raise RuntimeError(f'BLOCKER: frozen temperature {key} model has unexpected type/features/horizon')
        df = table_frame(frames, h)
        encodings[key] = build_tables(df)
        counts[key] = int((df['split'] == 'train').sum())
        gates[key] = _gate(results, key, lh_summary)
        models[key] = m
        refs.append(reference_rows(df, m, h, mapping['rep_zone']))
    bundle = {
        'models': models, 'encodings': encodings, 'gate': gates,
        'zone_to_cell': mapping['zone_to_cell'], 'rep_zone': mapping['rep_zone'], 'cells': sorted(cells['cell']),
        'grid_cells': cells.to_dict('records'),
        'meta': {'status': 'PACKAGED FOR PROMOTION REVIEW - served only when explicitly enabled; not promoted',
                 'variant': VARIANT, 'features': TEMP_FEATURES + TEMP_CANDIDATE_EXTRA,
                 'training_period': TRAIN, 'table_rows_train': counts,
                 'table_levels': {'clim_target': tv.TEMP_CLIM_TARGET_LEVELS, 'clim_origin': tv.TEMP_CLIM_ORIGIN_LEVELS},
                 'min_count': tv.ENCODING_MIN_COUNT,
                 'source': 'data/external/open_meteo_chennai_hourly_2017_2023.csv.gz (development years only)',
                 'zone1_mapping': {'zone': 1, 'cell': mapping['zone_to_cell'][1], 'note': ZONE1_NOTE},
                 'unused_cells': sorted(set(cells['cell']) - set(mapping['zone_to_cell'].values())),
                 'frozen_artifact': FROZEN_ARTIFACT, 'frozen_artifact_sha256': frozen_sha,
                 'protocol_sha256': lock['sha256']['protocol'], 'final_results_sha256': _sha256(FINAL_RESULTS),
                 'limitations': ['weakest in winter (2024 skill 0.249 at +6 h, 0.112 at +24 h)',
                                 'slight cold bias (2024 mean error -0.07 / -0.12 C; 2025-26 -0.11 / -0.19 C)',
                                 'reanalysis targets, not station observations'],
                 'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}}
    ref = pd.concat(refs, ignore_index=True)
    if save:
        assert os.path.normpath(ARTIFACT) not in {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS}
        os.makedirs(os.path.dirname(_abs(REFERENCE_CSV)), exist_ok=True)
        tmp = _abs(ARTIFACT) + '.tmp'
        joblib.dump(bundle, tmp)
        os.replace(tmp, _abs(ARTIFACT))
        ref.to_csv(_abs(REFERENCE_CSV), index=False)
    return {'bundle': bundle, 'reference': ref}


def verify_live_parity(bundle: Dict[str, Any], reference: pd.DataFrame, dev: pd.DataFrame,
                       tolerance: float = tv.PROMOTION_TOLERANCE) -> Dict[str, Any]:
    """Rebuild every reference row with the LIVE builder (live_forecaster.temp_candidate_row) from a live-format
    72 h window of the representative zone (development data), and compare features and predictions."""
    from src.forecasting.live_forecaster import WEATHER_COLUMNS, WEATHER_CONTEXT_HOURS, temp_candidate_row
    from src.preprocessing.live_inputs import hourly_frame
    live = dev[dev['location_key'].str.fullmatch(r'zone_\d\d')].copy()      # current live requests (no coastal ref)
    live['zone_id'] = live['location_key'].str[5:].astype(int)
    out = {'tolerance': tolerance}
    for key, ref in reference.groupby('horizon'):
        m, enc = bundle['models'][key], bundle['encodings'][key]
        fdiff = pdiff = 0.0
        for _, r in ref.iterrows():
            origin = pd.Timestamp(r['origin_utc'])
            frame = hourly_frame(live, int(r['rep_zone']), WEATHER_COLUMNS, origin, WEATHER_CONTEXT_HOURS)
            x = temp_candidate_row(frame, m.horizon_h, r['cell'], enc)
            fdiff = max(fdiff, float(np.nanmax(np.abs(x[m.features].to_numpy(dtype=float)[0] - r[m.features].to_numpy(dtype=float)))))
            pdiff = max(pdiff, abs(float(m.predict(x[m.features])[0]) - float(r['pred_temp_c'])))
        out[key] = {'rows': int(len(ref)), 'max_feature_diff': fdiff, 'max_prediction_diff': pdiff}
    out['passed'] = all(out[k][d] <= tolerance for k in reference['horizon'].unique()
                        for d in ('max_feature_diff', 'max_prediction_diff'))
    return out


# ── rainfall: frozen R3 (+6 h), R4 (+24 h) occurrence/amount and C2 (24 h total) ────────────────────────────
RAIN_ARTIFACT = os.path.join('models', 'rainfall', 'rain_history_6h_24h.joblib')
RAIN_CONFIGS = {'6h': ('rainfall', '6h', 'R3_harmonic_prob_weather_cond', 'rain_occurrence_6h'),
                '24h': ('rainfall', '24h', 'R4_lookup_full', 'rain_occurrence_24h')}


def _harmonic_logistic(df: pd.DataFrame, h: int):
    """Same fit as train_and_validate_hmm._harmonic_probability (training rows only), returning the model."""
    from sklearn.linear_model import LogisticRegression
    from src.models.rain_model import WET_MM
    X = tv.harmonic_design(df.index + pd.Timedelta(hours=h))[:, 1:]
    tr = (df['split'] == 'train').to_numpy()
    o = (df['y'].to_numpy() >= WET_MM).astype(int)
    return LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], o[tr])


def build_rain_package(save: bool = True, replace: bool = False) -> Dict[str, Any]:
    """Package the frozen rainfall candidates with training-only (2017-2023) tables. No refit of any model."""
    from src.models import final_eval_2024 as fe
    from src.models import rain24_total_experiment as rx
    from src.models.encoding import apply_table, logit_clip
    from src.models.rain_model import WET_MM
    if save and os.path.exists(_abs(RAIN_ARTIFACT)) and not replace:
        raise FileExistsError(f'{RAIN_ARTIFACT} exists - pass replace=True to rebuild it')
    lock = json.load(open(_abs(PROTOCOL_LOCK), encoding='utf-8'))
    for k, p in (('artifact_long_horizon_history', FROZEN_ARTIFACT), ('artifact_rain24_total', rx.ARTIFACT)):
        if lock['sha256'][k] != _sha256(p):
            raise RuntimeError(f'{p} differs from the artifact locked by the final-evaluation protocol')
    results = json.load(open(_abs(FINAL_RESULTS), encoding='utf-8'))['results']
    frozen = joblib.load(_abs(FROZEN_ARTIFACT))
    c2art = fe.load_with_main_remap(rx.ARTIFACT)
    dev, cells, frames = history_frames()
    mapping = zone_mapping(cells)
    series = {c: f['precipitation_mm'] for c, f in frames.items()}
    models, enc, gates, parity = {}, {}, {}, {}
    for key, (var, k, variant, rid) in RAIN_CONFIGS.items():
        if frozen['selections'][(var, k)]['selected'] != variant:
            raise RuntimeError(f'BLOCKER: frozen rainfall {key} is not {variant}')
        m = frozen['models'][(var, k)]
        h = int(key[:-1])
        full = tv.twopart_rows(series, key, {'train': TRAIN})            # frame the experiment's tables came from
        full['state'] = (full['persistence'].to_numpy() >= WET_MM).astype(int)
        wet = (full['y'].to_numpy() >= WET_MM).astype(float)
        e = {'pclim': fit_table(full, wet, tv.RAIN_PCLIM_LEVELS, per_cell=False, min_count=tv.ENCODING_MIN_COUNT),
             'pcond': fit_table(full, wet, tv.RAIN_PCOND_LEVELS, per_cell=False, min_count=tv.ENCODING_MIN_COUNT)}
        df, ph = lh._frames_for('rainfall', frames, key, {'train': TRAIN})
        if 'logit_pharm' in m.feature_names:
            hl = _harmonic_logistic(df, h)
            got = hl.predict_proba(tv.harmonic_design(df.index + pd.Timedelta(hours=h))[:, 1:])[:, 1]
            if not np.allclose(got, ph[WET_MM], rtol=1e-12, atol=1e-12):
                raise RuntimeError('BLOCKER: harmonic probability not reproduced')
            e['pharm'] = hl
        # reference rows (2023): experiment features with non-leave-one-out table values
        ref = df[(df.index.year == 2023) & df['cell'].isin(list(mapping['rep_zone']))].iloc[::997].copy()
        ref['logit_pclim'] = logit_clip(apply_table(e['pclim'], ref))
        ref['logit_pcond'] = logit_clip(apply_table(e['pcond'], ref))
        models[key], enc[key] = m, e
        g = results[rid]
        gates[key] = {'probability_passed': bool(g['gate']['passed']), 'amount_passed': bool(g['gate'].get('amount_passed')),
                      'test_period': '2024', 'test_brier_skill': g['summary']['skill'],
                      'test_ci_week': [g['summary']['ci_week_low'], g['summary']['ci_week_high']],
                      'test_auc': g['summary']['auc'], 'test_mae_median_mm': g['summary']['mae_model_median'],
                      'test_rmse_mean_mm': g['summary']['rmse_model_mean'], 'test_pod': g['summary']['pod'],
                      'test_far': g['summary']['far'], 'test_csi': g['summary']['csi'],
                      'amount_note': 'amount check is lenient (CRPS vs climatological occurrence with the same amounts)'}
        parity[key] = ref
    # C2: 24 h total
    base = rx.build_base(frames)
    cdf = rx.make_split(base, {'train': TRAIN})
    ev = (cdf['y'].to_numpy() >= rx.EVENT_MM).astype(float)
    c2enc = {'pclim25': fit_table(cdf, ev, [['origin_month'], []], per_cell=True, min_count=tv.ENCODING_MIN_COUNT),
             'pcond25': fit_table(cdf, ev, [['state', 'origin_month'], ['state'], []], per_cell=False, min_count=tv.ENCODING_MIN_COUNT),
             'clim_mean': fit_table(cdf, cdf['y'].to_numpy(), [['origin_month'], []], per_cell=True, min_count=tv.ENCODING_MIN_COUNT)}
    g = results['rain_total_24h']
    c2s = g['summary']['C2']
    gates['24h_total'] = {'passed': bool(g['gate']['passed']), 'test_period': '2024', 'test_crps_skill': c2s['crpss'],
                          'test_brier_skill_ge2.5': c2s['bss'], 'test_mae_median_mm': c2s['mae_median'],
                          'test_rmse_mean_mm': c2s['rmse_mean'], 'test_ratio_pred_obs': c2s['ratio_pred_obs'],
                          'test_pod': c2s['pod'], 'test_far': c2s['far'], 'test_csi': c2s['csi']}
    bundle = {'models': models, 'encodings': enc, 'c2': {'model': c2art['models']['C2'], 'encodings': c2enc},
              'gate': gates, 'zone_to_cell': mapping['zone_to_cell'], 'rep_zone': mapping['rep_zone'],
              'cells': sorted(cells['cell']),
              'meta': {'status': 'EXPERIMENTAL - frozen history-trained rainfall models; passed the 2024 test; not promoted',
                       'training_period': TRAIN, 'min_count': tv.ENCODING_MIN_COUNT,
                       'zone1_mapping': {'zone': 1, 'cell': mapping['zone_to_cell'][1], 'note': ZONE1_NOTE},
                       'c2_context_hours': 721, 'heavy_rain_alerts': 'not evaluated (too few heavy-rain days)',
                       'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}}
    if save:
        tmp = _abs(RAIN_ARTIFACT) + '.tmp'
        joblib.dump(bundle, tmp)
        os.replace(tmp, _abs(RAIN_ARTIFACT))
    return {'bundle': bundle, 'reference': parity, 'c2_frame': cdf}


def verify_rain_parity(bundle, reference, dev, c2_frame=None, n_c2: int = 40) -> Dict[str, Any]:
    """Rebuild reference rows with the LIVE builders (live forecaster) from live-format windows of 2023 data."""
    from src.forecasting import live_forecaster as lf
    from src.preprocessing.live_inputs import hourly_frame
    live = dev[dev['location_key'].str.fullmatch(r'zone_\d\d')].copy()
    live['zone_id'] = live['location_key'].str[5:].astype(int)
    out = {}
    for key, ref in reference.items():
        m, e = bundle['models'][key], bundle['encodings'][key]
        fd = pd_ = 0.0
        for t, r in ref.iterrows():
            z = bundle['rep_zone'][r['cell']]
            frame = hourly_frame(live, z, lf.WEATHER_COLUMNS, t, lf.WEATHER_CONTEXT_HOURS)
            x = lf.rain_history_row(frame['precipitation_mm'], frame, int(key[:-1]), r['cell'], bundle['cells'], e)
            want = r[m.feature_names].to_numpy(dtype=float)
            fd = max(fd, float(np.nanmax(np.abs(x[m.feature_names].to_numpy(dtype=float)[0] - want))))
            a = m.predict(x[m.feature_names], np.array([r['season']]))['p_wet'][0]
            b = m.predict(r[m.feature_names].to_frame().T.astype(float), np.array([r['season']]))['p_wet'][0]
            pd_ = max(pd_, abs(a - b))
        out[key] = {'rows': int(len(ref)), 'max_feature_diff': fd, 'max_prob_diff': pd_}
    if c2_frame is not None:
        from src.models.encoding import apply_table, logit_clip
        c2, e = bundle['c2']['model'], bundle['c2']['encodings']
        ref = c2_frame[(c2_frame.index.year == 2023) & c2_frame['cell'].isin(list(bundle['rep_zone']))]
        ref = ref.iloc[np.linspace(0, len(ref) - 1, n_c2).astype(int)].copy()
        ref['logit_pclim25'] = logit_clip(apply_table(e['pclim25'], ref))
        ref['logit_pcond25'] = logit_clip(apply_table(e['pcond25'], ref))
        ref['log_clim_mean'] = np.log1p(apply_table(e['clim_mean'], ref))
        fd = pd_ = 0.0
        for t, r in ref.iterrows():
            frame = hourly_frame(live, bundle['rep_zone'][r['cell']], lf.WEATHER_COLUMNS, t, lf.RAIN_HISTORY_CONTEXT_HOURS)
            x = lf.c2_feature_row(frame, r['cell'], bundle['cells'], e)
            fd = max(fd, float(np.nanmax(np.abs(x[c2.feats].to_numpy(dtype=float)[0] - r[c2.feats].to_numpy(dtype=float)))))
            a, b = c2.predict(x), c2.predict(r.to_frame().T.infer_objects())
            pd_ = max(pd_, abs(float(a['mean'][0]) - float(b['mean'][0])), abs(float(a['p25'][0]) - float(b['p25'][0])))
        out['24h_total'] = {'rows': int(len(ref)), 'max_feature_diff': fd, 'max_prediction_diff': pd_}
    out['passed'] = all(v.get('max_feature_diff', 0) <= 1e-9 for v in out.values() if isinstance(v, dict))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Package frozen history candidates (no refit)')
    ap.add_argument('--replace', action='store_true', help='rebuild the package file if it exists')
    ap.add_argument('--rain', action='store_true', help='build the rainfall package instead of temperature')
    a = ap.parse_args()
    if a.rain:
        r = build_rain_package(replace=a.replace)
        dev, _, _ = history_frames()
        print(json.dumps({'artifact': RAIN_ARTIFACT, 'parity': verify_rain_parity(r['bundle'], r['reference'], dev, r['c2_frame']),
                          'gates': r['bundle']['gate']}, indent=2, default=str))
        raise SystemExit(0)
    res = build_package(replace=a.replace)
    dev, _, _ = history_frames()
    parity = verify_live_parity(res['bundle'], res['reference'], dev)
    print(json.dumps({'artifact': ARTIFACT, 'reference': REFERENCE_CSV, 'parity': parity,
                      'zone_to_cell': res['bundle']['zone_to_cell'], 'rep_zone': res['bundle']['rep_zone'],
                      'gates': {k: v['passed'] for k, v in res['bundle']['gate'].items()}}, indent=2, default=str))
