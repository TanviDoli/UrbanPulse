"""
Long-horizon (6 h / 24 h) experiment on the 2017-2023 Open-Meteo history. EXPERIMENT ONLY.

Reuses the existing long-horizon framework in train_and_validate_hmm (feature frames, candidate variants
T0-T4 / R0-R4, harmonic and lookup climatologies, evaluate_temp / evaluate_twopart, day-block bootstrap).
Adds: unique-grid-cell series from the history file, year-based expanding folds, the pre-declared rules in
results/experiments/long_horizon_history/selection_criteria.md, 7-day-block intervals, shift diagnostics,
and a secondary (already examined) 2025-26 robustness check.

Never touches 2024: development data come only from load_history(include_sealed=False); an audit hook
refuses any open() of the sealed file during the run; frames are checked to hold no 2024 values.
Writes only under models/experiments/long_horizon_history and results/experiments/long_horizon_history.
"""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_MODELS = os.path.join('models', 'experiments', 'long_horizon_history')
OUT_RESULTS = os.path.join('results', 'experiments', 'long_horizon_history')
CRITERIA = os.path.join(OUT_RESULTS, 'selection_criteria.md')
ARTIFACT = os.path.join(OUT_MODELS, 'refit_2017_2023_candidates.joblib')
CV_CSV = os.path.join(OUT_RESULTS, 'cv_selection.csv')
CV_DETAIL_CSV = os.path.join(OUT_RESULTS, 'cv_metrics_by_cell_season.csv')
VAL_CSV = os.path.join(OUT_RESULTS, 'validation_2023_metrics.csv')
ROB_CSV = os.path.join(OUT_RESULTS, 'robustness_2025_2026_metrics.csv')
SUMMARY_CSV = os.path.join(OUT_RESULTS, 'summary.csv')
DIAG_CSV = os.path.join(OUT_RESULTS, 'shift_diagnostics.csv')
CELLS_CSV = os.path.join(OUT_RESULTS, 'grid_cells.csv')
REPORT = os.path.join(OUT_RESULTS, 'report.md')
RUN_MANIFEST = os.path.join(OUT_RESULTS, 'run_manifest.json')
OUTPUTS = (ARTIFACT, CV_CSV, CV_DETAIL_CSV, VAL_CSV, ROB_CSV, SUMMARY_CSV, DIAG_CSV, CELLS_CSV, REPORT, RUN_MANIFEST)

FOLDS = {f'F{y}': {'train': ('2017-01-01 00:00', f'{y - 1}-12-31 23:00'),
                   'validate': (f'{y}-01-01 00:00', f'{y}-12-31 23:00')} for y in (2019, 2020, 2021, 2022)}
VALIDATION = {'train': ('2017-01-01 00:00', '2022-12-31 23:00'), 'validate': ('2023-01-01 00:00', '2023-12-31 23:00')}
ROBUST = {'train': ('2017-01-01 00:00', '2023-12-31 23:00'), 'robust': ('2025-01-01 00:00', '2026-10-07 23:00')}
TEMP_HORIZONS = [6, 24]
RAIN_KEYS = ['6h', '24h', tv.RAIN_OM_ACCUM_KEY]
REFERENCE = {'temperature': 'T0_production', 'rainfall': 'R0_production'}

# pre-declared rules (selection_criteria.md)
MIN_POSITIVE_FOLDS = 3
MIN_POSITIVE_CELLS = 6
MIN_EVENTS, MIN_EVENT_DAYS = 100, 30
HEAVY_MIN_DAYS = 10
BLOCK_DAYS = 7
VARS = ['temperature_2m_c', 'relative_humidity_2m_pct', 'precipitation_mm', 'wind_speed_10m_kmh', 'wind_direction_10m_deg']


# ── safety ──────────────────────────────────────────────────────────────────
def sealed_open_guard(event: str, args: tuple) -> None:
    """Audit hook: refuse any open() of the sealed 2024 file."""
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    from src.collectors.open_meteo_history import HISTORY_SEALED_FILE
    p = os.fsdecode(args[0])
    if '2024_SEALED' in p or os.path.normcase(os.path.abspath(p)) == os.path.normcase(os.path.abspath(HISTORY_SEALED_FILE)):
        raise PermissionError('long-horizon history experiment: the sealed 2024 file must not be opened')


def assert_no_2024(frames: Dict[str, pd.DataFrame]) -> None:
    for cell, f in frames.items():
        in24 = f.index.year == 2024
        if in24.any() and f.loc[in24, [c for c in VARS if c in f]].notna().to_numpy().any():
            raise RuntimeError(f'2024 values present in frame {cell}')


def _sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── grid cells ──────────────────────────────────────────────────────────────
def grid_cells(dev: pd.DataFrame) -> pd.DataFrame:
    """
    One row per unique Open-Meteo grid cell: representative history location and the zone of the existing
    2025-26 file holding the same series. Zones sharing a cell are listed but not used as separate series.
    The current zone-1 request (snapped to the zone 2/3 cell) is never a representative; the coastal
    reference location is its own cell (= zone 1 of the 2025-26 file).
    """
    locs = dev.groupby('location_key')[['zone_id', 'grid_latitude', 'grid_longitude']].first().reset_index()
    rows = []
    for (la, lo), g in locs.groupby(['grid_latitude', 'grid_longitude']):
        if 'zone_01_coastal_original' in set(g['location_key']):
            rep, local_zone = 'zone_01_coastal_original', 1
        else:
            cand = g[g['location_key'] != 'zone_01'].sort_values('zone_id')
            rep, local_zone = cand['location_key'].iloc[0], int(cand['zone_id'].iloc[0])
        rows.append({'grid_latitude': la, 'grid_longitude': lo, 'rep_location_key': rep, 'local_file_zone': local_zone,
                     'locations': ','.join(sorted(g['location_key']))})
    out = pd.DataFrame(rows).sort_values(['grid_latitude', 'grid_longitude'], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, 'cell', [f'gc{i + 1}' for i in range(len(out))])
    return out


def cell_frames(cells: pd.DataFrame, dev: pd.DataFrame, local: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Per cell: hourly frame (all variables) of the representative history location, optionally followed by
    the matching zone of the 2025-26 file. The gap between them (2024) is all-NaN by construction."""
    frames = {}
    for _, c in cells.iterrows():
        parts = [dev[dev['location_key'] == c['rep_location_key']].set_index('time_utc')[VARS]]
        if local is not None:
            parts.append(local[local['zone_id'] == c['local_file_zone']].set_index('time_utc')[VARS])
        g = pd.concat(parts).sort_index()
        g = g[~g.index.duplicated()]
        frames[c['cell']] = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq='h'))
    assert_no_2024(frames)
    return frames


def _series(frames, col):
    return {c: f[col] for c, f in frames.items()}


def _weeks(days: np.ndarray) -> np.ndarray:
    return ((pd.to_datetime(pd.Series(days)) - pd.Timestamp('2017-01-02')).dt.days // BLOCK_DAYS).to_numpy()


# ── evaluation of one fitted variant ────────────────────────────────────────
def eval_temp(df: pd.DataFrame, h: int, feats: List[str], ev: str, model=None):
    """Fit on the training rows and score `ev` rows; with `model` given, score that fitted model instead."""
    if model is None:
        model, m = tv._lh_fit_eval_temp(df, h, feats)
    else:
        m = tv.evaluate_temp(df, model.predict(df[feats]), f'{h}h',
                             extra_baselines={'harmonic_climatology': df['hclim_target'].to_numpy()})
    m = m[m['split'] == ev].copy()
    pred = model.predict(df[feats])
    mask = (df['split'] == ev).to_numpy()
    y = df['y'].to_numpy()
    base = {'persistence': df['t_now'].to_numpy(), 'same_hour_yesterday': df['t_same_hour_yday'].to_numpy(),
            'climatology': tv._train_lookup(df, y, [['target_month', 'target_hour'], ['target_hour'], []]),
            'harmonic_climatology': df['hclim_target'].to_numpy()}
    pooled = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL')].iloc[0]
    wk = _weeks(df.loc[mask, 'day'].to_numpy())
    lo, hi = tv.day_block_ci(wk, (pred[mask] - y[mask]) ** 2, (base[pooled['best_baseline']][mask] - y[mask]) ** 2)
    cellm = m[(m['cell'] != 'ALL') & (m['season'] == 'ALL')]
    err = pred[mask] - y[mask]
    seas = df.loc[mask, 'season'].to_numpy()
    summ = {'skill': float(pooled['skill']), 'best_baseline': pooled['best_baseline'],
            'ci_week_low': lo, 'ci_week_high': hi,
            'ci_day_low': float(pooled['skill_ci_low']), 'ci_day_high': float(pooled['skill_ci_high']),
            'rmse_model': float(pooled['rmse_model']), 'mae_model': float(pooled['mae_model']),
            'rmse_best_baseline': float(pooled[f"rmse_{pooled['best_baseline']}"]),
            'n': int(pooled['n']), 'n_days': int(pooled['n_days']),
            'cells_positive': int((cellm['skill'] > 0).sum()), 'n_cells': int(len(cellm)),
            'bias_model': float(err.mean())}
    diag = [{'season': s, 'bias_model': float(err[seas == s].mean()), 'n': int((seas == s).sum())} for s in np.unique(seas)]
    return model, m, summ, diag


def eval_rain(df: pd.DataFrame, ph, key: str, feats: List[str], ev: str, p_star: Optional[float] = None, model=None):
    """Fit on the training rows and score `ev` rows; with `model` given, score that fitted model instead."""
    from src.models.rain_model import WET_MM, TwoPartRainModel
    if model is None:
        tr = df[df['split'] == 'train']
        model = TwoPartRainModel(key, feature_names=feats).fit(tr[feats], tr['y'].to_numpy(), tr['season'].to_numpy())
    scored = tv.twopart_scored_rows(df, model.predict(df[feats], df['season'].to_numpy()), key, extra_prob_baselines=ph)
    se = scored[scored['split'] == ev]
    m = tv.evaluate_twopart(se, key, p_star=p_star)
    thr0 = tv._twopart_events(key)[0]
    pooled = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL')].iloc[0]
    best = pooled[f'bss_ge{thr0}_best_baseline']
    col = {'climatology': 'pc', 'persistence': 'pp', 'conditional_climatology': 'pk', 'harmonic_climatology': 'ph'}[best]
    o = se[f'o_{thr0}'].to_numpy()
    wk = _weeks(se['day'].to_numpy())
    lo, hi = tv.day_block_ci(wk, (se[f'pm_{thr0}'].to_numpy() - o) ** 2, (se[f'{col}_{thr0}'].to_numpy() - o) ** 2)
    clo, chi = tv.day_block_ci(wk, se['crps_model'].to_numpy(), se['crps_clim'].to_numpy())
    cellm = m[(m['cell'] != 'ALL') & (m['season'] == 'ALL')]
    heavy_days = int(se.loc[se['y'] >= tv.TWOPART_HEAVY_MM, 'day'].nunique()) if key == tv.RAIN_OM_ACCUM_KEY else None
    summ = {'skill': float(pooled[f'bss_ge{thr0}']), 'best_baseline': best, 'event_mm': thr0,
            'ci_week_low': lo, 'ci_week_high': hi,
            'ci_day_low': float(pooled['bss_ci_low']), 'ci_day_high': float(pooled['bss_ci_high']),
            'auc': float(pooled[f'auc_ge{thr0}']), 'crpss_clim': float(pooled['crpss_vs_clim']),
            'crpss_clim_ci_week_low': clo, 'crpss_clim_ci_week_high': chi,
            'crpss_zero': float(pooled['crpss_vs_zero']), 'crpss_persistence': float(pooled['crpss_vs_persistence']),
            'mae_model_median': float(pooled['mae_model_median']), 'rmse_model_mean': float(pooled['rmse_model_mean']),
            'mae_persistence': float(pooled['mae_persistence']), 'rmse_zero': float(pooled['rmse_zero']),
            'events': int(o.sum()), 'event_days': int(se.loc[o == 1, 'day'].nunique()),
            'heavy_days_ge64.5': heavy_days, 'prob_threshold': float(pooled['prob_threshold']),
            'pod': float(pooled['pod']), 'far': float(pooled['far']), 'csi': float(pooled['csi']),
            'n': int(pooled['n']), 'n_days': int(pooled['n_days']),
            'cells_positive': int((cellm[f'bss_ge{thr0}'] > 0).sum()), 'n_cells': int(len(cellm)),
            'mean_p_wet': float(se[f'pm_{WET_MM}'].mean()), 'obs_wet_freq': float(se[f'o_{WET_MM}'].mean()),
            'mean_pred_mm': float(se['model_mean'].mean()), 'mean_obs_mm': float(se['y'].mean())}
    seas = se['season'].to_numpy()
    diag = [{'season': s, 'mean_p_wet': float(se.loc[seas == s, f'pm_{WET_MM}'].mean()),
             'obs_wet_freq': float(se.loc[seas == s, f'o_{WET_MM}'].mean()),
             'mean_pred_mm': float(se.loc[seas == s, 'model_mean'].mean()),
             'mean_obs_mm': float(se.loc[seas == s, 'y'].mean()), 'n': int((seas == s).sum())} for s in np.unique(seas)]
    return model, m, summ, diag


# ── rules ───────────────────────────────────────────────────────────────────
def select_variant(cv: pd.DataFrame, n_features: Dict[str, int]) -> Tuple[str, Dict[str, Any]]:
    """Eligible: skill > 0 in >= MIN_POSITIVE_FOLDS folds and mean > 0; pick the highest mean (ties: fewer features).
    None eligible -> best mean, flagged ineligible (cannot pass)."""
    agg = cv.groupby('variant')['skill'].agg(mean='mean', positive=lambda s: int((s > 0).sum()))
    agg['nf'] = [n_features[v] for v in agg.index]
    eligible = agg[(agg['positive'] >= MIN_POSITIVE_FOLDS) & (agg['mean'] > 0)]
    pool = eligible if len(eligible) else agg
    best = pool.sort_values(['mean', 'nf'], ascending=[False, True]).index[0]
    return best, {'eligible': bool(len(eligible)), 'mean_cv_skill': float(pool.loc[best, 'mean']),
                  'positive_folds': int(pool.loc[best, 'positive']), 'n_folds': int(cv['fold'].nunique())}


def validation_pass(variable: str, s: Dict[str, Any], eligible: bool) -> Dict[str, Any]:
    checks = {'selection_eligible': eligible, 'skill_positive': s['skill'] > 0,
              'week_ci_low_positive': np.isfinite(s['ci_week_low']) and s['ci_week_low'] > 0,
              'cells_positive': s['cells_positive'] >= MIN_POSITIVE_CELLS}
    if variable == 'rainfall':
        checks['enough_events'] = s['events'] >= MIN_EVENTS and s['event_days'] >= MIN_EVENT_DAYS
    passed = all(bool(v) for v in checks.values())
    out = {'passed': passed, **{k: bool(v) for k, v in checks.items()}}
    if variable == 'rainfall':
        out['amount_passed'] = bool(passed and s['crpss_clim'] > 0 and np.isfinite(s['crpss_clim_ci_week_low'])
                                    and s['crpss_clim_ci_week_low'] > 0)
    return out


# ── main experiment ─────────────────────────────────────────────────────────
def _frames_for(variable, frames, target, splits):
    if variable == 'temperature':
        return tv._lh_temp_frame(_series(frames, 'temperature_2m_c'), frames, target, splits), None
    return tv._lh_rain_frame(_series(frames, 'precipitation_mm'), frames, target, splits)


def _variants(variable, df):
    if variable == 'temperature':
        return tv._lh_temp_variants()
    return tv._lh_rain_variants([c for c in df.columns if c.startswith('is_')])


def _evaluate(variable, df, ph, target, feats, ev, p_star=None):
    if variable == 'temperature':
        return eval_temp(df, target, feats, ev)
    return eval_rain(df, ph, target, feats, ev, p_star)


def run_experiment(save: bool = True, install_guard: bool = True) -> Dict[str, Any]:
    from src.collectors import open_meteo
    from src.collectors.open_meteo_history import HISTORY_DEV_FILE, HISTORY_MANIFEST, UNSEAL_ENV, load_history
    if os.environ.get(UNSEAL_ENV):
        raise RuntimeError(f'{UNSEAL_ENV} is set - refusing to run (2024 must stay sealed)')
    crit_path = os.path.join(ROOT, CRITERIA)
    if not os.path.exists(crit_path):
        raise FileNotFoundError('pre-declared selection criteria missing')
    if install_guard:
        sys.addaudithook(sealed_open_guard)
    started = pd.Timestamp.now(tz='UTC').isoformat()
    crit_sha = _sha256(crit_path)
    dev = load_history(include_sealed=False)                       # manifest + hash verified
    yrs = sorted(dev['time_utc'].dt.year.unique().tolist())
    if yrs != list(range(2017, 2024)):
        raise RuntimeError(f'unexpected development years {yrs}')
    cells = grid_cells(dev)
    frames = cell_frames(cells, dev)
    targets = [('temperature', h, f'{h}h') for h in TEMP_HORIZONS] + [('rainfall', k, k) for k in RAIN_KEYS]

    # 1) selection folds
    cv_rows, cv_detail, diag_rows = [], [], []
    for fold, splits in FOLDS.items():
        for variable, target, key in targets:
            df, ph = _frames_for(variable, frames, target, splits)
            for name, feats in _variants(variable, df).items():
                _, m, s, dg = _evaluate(variable, df, ph, target, feats, 'validate')
                cv_rows.append({'variable': variable, 'horizon': key, 'fold': fold, 'variant': name, 'n_features': len(feats), **s})
                cv_detail.append(m.assign(variable=variable, fold=fold, variant=name))
                diag_rows += [{'period': fold, 'variable': variable, 'horizon': key, 'variant': name, **d} for d in dg]
            print(f'{fold} {variable} {key} done', flush=True)
    cv = pd.DataFrame(cv_rows)

    # 2) freeze selections (folds only)
    selections = {}
    for variable, target, key in targets:
        sub = cv[(cv['variable'] == variable) & (cv['horizon'] == key)]
        nf = sub.groupby('variant')['n_features'].first().to_dict()
        sel, info = select_variant(sub, nf)
        p_star = float(sub[sub['variant'] == sel]['prob_threshold'].median()) if variable == 'rainfall' else None
        selections[(variable, key)] = {'selected': sel, **info, 'p_star_from_folds': p_star}
    frozen_at = pd.Timestamp.now(tz='UTC').isoformat()

    # 3) validation 2023 and 4) refit 2017-2023 + secondary 2025-26 robustness
    local = open_meteo.load_local()['hourly']
    rob_frames = cell_frames(cells, dev, local)
    val_metrics, rob_metrics, summary, models = [], [], [], {}
    for variable, target, key in targets:
        sel = selections[(variable, key)]
        names = list(dict.fromkeys([sel['selected'], REFERENCE[variable]]))
        dfv, phv = _frames_for(variable, frames, target, VALIDATION)
        dfr, phr = _frames_for(variable, rob_frames, target, ROBUST)
        for name in names:
            feats = _variants(variable, dfv)[name]
            _, mv, sv, dgv = _evaluate(variable, dfv, phv, target, feats, 'validate', sel['p_star_from_folds'])
            model_r, mr, sr, dgr = _evaluate(variable, dfr, phr, target, feats, 'robust', sel['p_star_from_folds'])
            val_metrics.append(mv.assign(variable=variable, variant=name))
            rob_metrics.append(mr.assign(variable=variable, variant=name))
            diag_rows += [{'period': 'V2023', 'variable': variable, 'horizon': key, 'variant': name, **d} for d in dgv]
            diag_rows += [{'period': 'R2025_26', 'variable': variable, 'horizon': key, 'variant': name, **d} for d in dgr]
            role = 'selected' if name == sel['selected'] else 'production_features_reference'
            if name == sel['selected']:
                gate = validation_pass(variable, sv, sel['eligible'])
                models[(variable, key)] = model_r
            else:
                gate = {}
            summary.append({'variable': variable, 'horizon': key, 'variant': name, 'role': role,
                            **{f'cv_{k}': v for k, v in sel.items() if k != 'selected'},
                            **{f'val_{k}': v for k, v in sv.items()}, **{f'rob_{k}': v for k, v in sr.items()},
                            **{f'gate_{k}': v for k, v in gate.items()},
                            'recommend_promotion_review': bool(gate.get('passed') and sr['skill'] > 0) if gate else None})
        print(f'validation/robustness {variable} {key} done', flush=True)
    summary = pd.DataFrame(summary)
    result = {'cv': cv, 'cv_detail': pd.concat(cv_detail, ignore_index=True), 'selections': selections,
              'validation': pd.concat(val_metrics, ignore_index=True), 'robust': pd.concat(rob_metrics, ignore_index=True),
              'summary': summary, 'diagnostics': pd.DataFrame(diag_rows), 'cells': cells, 'models': models}
    with open(os.path.join(ROOT, HISTORY_MANIFEST), encoding='utf-8') as f:
        man = json.load(f)
    run_meta = {'started_utc': started, 'selection_frozen_utc': frozen_at, 'finished_utc': pd.Timestamp.now(tz='UTC').isoformat(),
                'criteria_file': CRITERIA, 'criteria_sha256': crit_sha,
                'dev_file': os.path.relpath(HISTORY_DEV_FILE, ROOT), 'dev_sha256_manifest': man['files']['dev']['sha256'],
                'dev_sha256_verified': True, 'dev_years': yrs, 'sealed_file_opened': False,
                'robustness_source': 'data/external/open_meteo_chennai_hourly_2025_2026.csv (already examined; secondary only)',
                'robustness_file_sha256': _sha256(open_meteo.LOCAL_FILE), 'folds': FOLDS, 'validation': VALIDATION,
                'robust': ROBUST, 'n_grid_cells': int(len(cells))}
    result['run_meta'] = run_meta
    result['report'] = build_report(cv, summary, result['validation'], result['robust'], result['diagnostics'],
                                    cells, run_meta)
    print(result['report'])
    if save:
        _save(result)
    return result


def _save(r: Dict[str, Any]) -> None:
    prod = {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS}
    assert not {os.path.normpath(p) for p in OUTPUTS} & prod
    assert all('experiments' in p for p in OUTPUTS)
    os.makedirs(os.path.join(ROOT, OUT_MODELS), exist_ok=True)
    os.makedirs(os.path.join(ROOT, OUT_RESULTS), exist_ok=True)
    joblib.dump({'models': r['models'], 'selections': r['selections'], 'cells': r['cells'],
                 'summary': r['summary'], 'run_meta': r['run_meta'],
                 'status': 'EXPERIMENT - refit on 2017-2023, not promoted, not used by the live pipeline; '
                           'not scored on 2024 (sealed)',
                 'note': 'Live use would additionally require storing the harmonic/encoding coefficients.'},
                os.path.join(ROOT, ARTIFACT))
    r['cv'].to_csv(os.path.join(ROOT, CV_CSV), index=False)
    r['cv_detail'].to_csv(os.path.join(ROOT, CV_DETAIL_CSV), index=False)
    r['validation'].to_csv(os.path.join(ROOT, VAL_CSV), index=False)
    r['robust'].to_csv(os.path.join(ROOT, ROB_CSV), index=False)
    r['summary'].to_csv(os.path.join(ROOT, SUMMARY_CSV), index=False)
    r['diagnostics'].to_csv(os.path.join(ROOT, DIAG_CSV), index=False)
    r['cells'].to_csv(os.path.join(ROOT, CELLS_CSV), index=False)
    with open(os.path.join(ROOT, RUN_MANIFEST), 'w', encoding='utf-8') as f:
        json.dump(r['run_meta'], f, indent=2, default=str)
    with open(os.path.join(ROOT, REPORT), 'w', encoding='utf-8') as f:
        f.write(r['report'])
    print('Saved:', ', '.join(OUTPUTS))

# ── report ──────────────────────────────────────────────────────────────────
def _f(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 'NaN'
    return repr(x) if np.isfinite(x) else 'NaN'


def _skill_col(m: pd.DataFrame) -> pd.Series:
    """Headline skill per metrics row: MSE skill (temperature) or Brier skill of the gating event (rainfall)."""
    sk = m['skill'].copy() if 'skill' in m else pd.Series(np.nan, index=m.index)
    rain = m['variable'] == 'rainfall'
    sk[rain] = m.loc[rain, 'bss_ge0.1']
    tot = m['horizon'] == tv.RAIN_OM_ACCUM_KEY
    sk[tot] = m.loc[tot, 'bss_ge2.5']
    return sk


def _breakdown(m: pd.DataFrame, sel: pd.DataFrame, by: str) -> str:
    m = m.merge(sel, on=['variable', 'horizon', 'variant'])
    m = m[(m['cell'] != 'ALL') & (m['season'] == 'ALL')] if by == 'cell' else m[(m['cell'] == 'ALL') & (m['season'] != 'ALL')]
    m = m.assign(sk=_skill_col(m), target=m['variable'] + ' ' + m['horizon'])
    return m.pivot_table(index=by, columns='target', values='sk').to_string(float_format=_f)


def build_report(cv: pd.DataFrame, summ: pd.DataFrame, val: pd.DataFrame, rob: pd.DataFrame, diag: pd.DataFrame,
                 cells: pd.DataFrame, meta: Dict[str, Any]) -> str:
    sel = summ[summ['role'] == 'selected'][['variable', 'horizon', 'variant']]
    L = ['# Long-horizon experiment on the 2017-2023 history (6 h / 24 h)', '',
         'EXPERIMENT ONLY - nothing promoted; production availability unchanged. All skill values are unrounded.', '',
         '## 1. Dataset and leakage checks', '',
         f"- Development data `{meta['dev_file']}`, loaded with `load_history(include_sealed=False)`; manifest SHA-256 "
         f"`{meta['dev_sha256_manifest']}` verified; years {meta['dev_years']}.",
         '- 2024: never opened (an audit hook refusing any open of the sealed file was active for the whole run; '
         'every frame checked to hold no 2024 values; unseal variable not set; no access log).',
         f"- {meta['n_grid_cells']} unique Open-Meteo grid cells, one series each (zones sharing a cell are not counted "
         'separately; the current zone-1 request is not used - it duplicates the zone 2/3 cell; the coastal reference is its own cell).',
         '- Rows: origin and target in the same period; climatologies, encodings and harmonic fits use training rows only '
         '(leave-one-out inside training).',
         f"- Criteria `{meta['criteria_file']}` (SHA-256 `{meta['criteria_sha256']}`) written before the run; "
         f"run started {meta['started_utc']}, selection frozen {meta['selection_frozen_utc']} before any 2023 scoring.",
         f"- Secondary check source: {meta['robustness_source']} (SHA-256 `{meta['robustness_file_sha256']}`).", '',
         '| cell | grid lat | grid lon | representative | zones / locations in cell |', '|---|---|---|---|---|']
    for _, c in cells.iterrows():
        L.append(f"| {c['cell']} | {c['grid_latitude']} | {c['grid_longitude']} | {c['rep_location_key']} | {c['locations']} |")
    L += ['', '## 2. Selection (folds 2019-2022, expanding from 2017)', '',
          'Eligible: skill > 0 in >= 3 of 4 folds and mean > 0; selected: highest mean (ties: fewer features).',
          'Skill = MSE skill vs best baseline (temperature) or Brier skill of the gating event vs best baseline (rainfall).', '']
    piv = cv.pivot_table(index=['variable', 'horizon', 'variant'], columns='fold', values='skill')
    folds = list(piv.columns)
    piv['mean'] = piv[folds].mean(axis=1)
    piv['positive_folds'] = (piv[folds] > 0).sum(axis=1)
    L += ['```', piv.to_string(float_format=_f), '```', '']
    L += ['## 3-4. 2023 validation (selected configuration vs production-input features, identical rows)', '',
          '| target | variant | role | fold eligible (+/4) | 2023 skill | 7-day-block 95% CI | day-block 95% CI | best baseline | cells > 0 | passes |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for _, s in summ.iterrows():
        L.append(f"| {s['variable']} {s['horizon']} | {s['variant']} | {s['role']} | "
                 f"{s['cv_eligible']} ({s['cv_positive_folds']}) | {_f(s['val_skill'])} | "
                 f"[{_f(s['val_ci_week_low'])}, {_f(s['val_ci_week_high'])}] | [{_f(s['val_ci_day_low'])}, {_f(s['val_ci_day_high'])}] | "
                 f"{s['val_best_baseline']} | {int(s['val_cells_positive'])}/{int(s['val_n_cells'])} | "
                 f"{s['gate_passed'] if s['role'] == 'selected' else '-'} |")
    t = summ[summ['variable'] == 'temperature']
    L += ['', 'Temperature errors on 2023 (deg C):', '', '| target | variant | RMSE model | RMSE best baseline | MAE model | mean error |',
          '|---|---|---|---|---|---|']
    for _, s in t.iterrows():
        L.append(f"| {s['horizon']} | {s['variant']} | {_f(s['val_rmse_model'])} | {_f(s['val_rmse_best_baseline'])} | "
                 f"{_f(s['val_mae_model'])} | {_f(s['val_bias_model'])} |")
    r = summ[summ['variable'] == 'rainfall']
    L += ['', 'Rainfall on 2023 (event = >= 0.1 mm/h for hourly rates, >= 2.5 mm for the 24 h total; events are cell-hours or '
          'overlapping cell-windows over 8 cells, so distinct days are the meaningful count):', '',
          '| target | variant | events | event days | AUC | CRPS skill vs clim [7-day CI] | vs zero | vs persistence | '
          'POD / FAR / CSI (threshold from folds) | mean pred / obs mm | amount passes |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, s in r.iterrows():
        L.append(f"| {s['horizon']} | {s['variant']} | {int(s['val_events'])} | {int(s['val_event_days'])} | {_f(s['val_auc'])} | "
                 f"{_f(s['val_crpss_clim'])} [{_f(s['val_crpss_clim_ci_week_low'])}, {_f(s['val_crpss_clim_ci_week_high'])}] | "
                 f"{_f(s['val_crpss_zero'])} | {_f(s['val_crpss_persistence'])} | "
                 f"{_f(s['val_pod'])} / {_f(s['val_far'])} / {_f(s['val_csi'])} (p >= {s['val_prob_threshold']}) | "
                 f"{_f(s['val_mean_pred_mm'])} / {_f(s['val_mean_obs_mm'])} | "
                 f"{s['gate_amount_passed'] if s['role'] == 'selected' else '-'} |")
    tot = summ[(summ['horizon'] == tv.RAIN_OM_ACCUM_KEY) & (summ['role'] == 'selected')]
    if len(tot):
        L += ['', f"Heavy rain (24 h total >= {tv.TWOPART_HEAVY_MM} mm, any cell): {int(tot['val_heavy_days_ge64.5'].iloc[0])} "
              f"distinct origin days in 2023 and {int(tot['rob_heavy_days_ge64.5'].iloc[0])} in 2025-26 - below the "
              f"{HEAVY_MIN_DAYS}-day minimum, so no heavy-rain skill is claimed; heavy-rain alerts stay disabled."]
    L += ['', '## 5. Seasonal and grid-cell performance (selected configurations)', '',
          '2023 by season (all cells):', '', '```', _breakdown(val, sel, 'season'), '```', '',
          '2023 by unique grid cell (all seasons):', '', '```', _breakdown(val, sel, 'cell'), '```', '']
    L += ['## 6. Secondary robustness 2025-01-01..2026-10-07 (ALREADY EXAMINED data - NOT an untouched test; '
          'refit on 2017-2023, frozen configuration, no tuning)', '',
          '| target | variant | role | skill | 7-day-block 95% CI | best baseline | cells > 0 | events (days) | recommend review |',
          '|---|---|---|---|---|---|---|---|---|']
    for _, s in summ.iterrows():
        ev = f"{int(s['rob_events'])} ({int(s['rob_event_days'])})" if s['variable'] == 'rainfall' else '-'
        L.append(f"| {s['variable']} {s['horizon']} | {s['variant']} | {s['role']} | {_f(s['rob_skill'])} | "
                 f"[{_f(s['rob_ci_week_low'])}, {_f(s['rob_ci_week_high'])}] | {s['rob_best_baseline']} | "
                 f"{int(s['rob_cells_positive'])}/{int(s['rob_n_cells'])} | {ev} | "
                 f"{s['recommend_promotion_review'] if s['role'] == 'selected' else '-'} |")
    L += ['', '2025-26 by season (selected):', '', '```', _breakdown(rob, sel, 'season'), '```', '',
          '2025-26 by unique grid cell (selected):', '', '```', _breakdown(rob, sel, 'cell'), '```', '']
    d = diag.merge(sel, on=['variable', 'horizon', 'variant'])
    order = list(FOLDS) + ['V2023', 'R2025_26']
    dt = d[d['variable'] == 'temperature'].pivot_table(index=['horizon', 'season'], columns='period', values='bias_model')
    dr = d[d['variable'] == 'rainfall'].assign(ratio=lambda x: x['mean_pred_mm'] / x['mean_obs_mm'])
    dr = dr.pivot_table(index=['horizon', 'season'], columns='period', values='ratio')
    L += ['## 7. Distribution-shift diagnostics (stability-audit follow-up)', '',
          'Temperature mean error (model - observed, deg C), selected configuration:', '',
          '```', dt[[c for c in order if c in dt]].to_string(float_format=_f), '```', '',
          'Rainfall mean predicted / mean observed amount, selected configuration:', '',
          '```', dr[[c for c in order if c in dr]].to_string(float_format=_f), '```', '']
    return '\n'.join(L) + '\n'


def rebuild_report() -> str:
    """Regenerate report.md from the saved outputs (no refitting, no new scoring)."""
    def rd(p):
        return pd.read_csv(os.path.join(ROOT, p), float_precision='round_trip')
    with open(os.path.join(ROOT, RUN_MANIFEST), encoding='utf-8') as f:
        meta = json.load(f)
    rep = build_report(rd(CV_CSV), rd(SUMMARY_CSV), rd(VAL_CSV), rd(ROB_CSV), rd(DIAG_CSV), rd(CELLS_CSV), meta)
    with open(os.path.join(ROOT, REPORT), 'w', encoding='utf-8') as f:
        f.write(rep)
    return rep


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Long-horizon experiment on the 2017-2023 history (experiment outputs only)')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--rebuild-report', action='store_true', help='regenerate report.md from saved outputs only')
    a = ap.parse_args()
    if a.rebuild_report:
        print(rebuild_report())
    else:
        run_experiment(save=not a.no_save)
