"""
Final independent evaluation on the sealed 2024 year — PREPARATION + GUARDED SINGLE RUN. EXPERIMENT ONLY.

Frozen configurations (selected on 2019-2022 folds, validated on 2023, refit on 2017-2023):
  temperature +6 h / +24 h  T4_lookup_clim_full            (long_horizon_history refit artifact)
  rainfall occurrence +6 h  R3_harmonic_prob_weather_cond  (same artifact)
  rainfall occurrence +24 h R4_lookup_full                 (same artifact)
  rainfall 24 h TOTAL       C2 hurdle model, reference A0  (rain24_total refit artifact)

Commands
  --preflight         (default) no 2024 access: loads every frozen artifact, rebuilds the feature frames from
                      2017-2023 only, refits in memory and checks the saved models are reproduced exactly,
                      reports the protocol status and writes a preparation record.
  --freeze-protocol   locks protocol + artifacts + criteria + code by SHA-256; refused while the protocol still
                      contains a 'PENDING DECISION' or a lock already exists.
  --run --reason ...  the single final evaluation. Refused unless the lock exists and every hash still matches,
                      the reason is non-empty, URBANPULSE_UNSEAL_2024 carries the unseal token, the preflight
                      reproduction passes, and no final result exists yet. Only then is load_history(
                      include_sealed=True, reason) called (which logs the access).
Writes only under results/experiments/final_eval_2024.
"""

import hashlib
import json
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.collectors import open_meteo_history as omh
from src.models import long_horizon_history as lh
from src.models import rain24_total_experiment as rx
from src.models import train_and_validate_hmm as tv

ROOT = lh.ROOT
OUT = os.path.join('results', 'experiments', 'final_eval_2024')
PROTOCOL = os.path.join(OUT, 'protocol_v1.md')
LOCK = os.path.join(OUT, 'protocol_lock.json')
PREP_RECORD = os.path.join(OUT, 'preparation_record.json')
FINAL_RESULTS = os.path.join(OUT, 'final_results.json')
FINAL_METRICS = os.path.join(OUT, 'final_metrics_by_group.csv')
FINAL_REPORT = os.path.join(OUT, 'final_report.md')

TRAIN = ('2017-01-01 00:00', '2023-12-31 23:00')
TEST = ('2024-01-01 00:00', '2024-12-31 23:00')
FINAL_SPLITS = {'train': TRAIN, 'test': TEST}
PREFLIGHT_SPLITS = {'train': TRAIN}

CONFIGS = [
    {'id': 'temperature_6h', 'family': 'lh', 'variable': 'temperature', 'target': 6, 'key': '6h',
     'variant': 'T4_lookup_clim_full', 'p_star': None},
    {'id': 'temperature_24h', 'family': 'lh', 'variable': 'temperature', 'target': 24, 'key': '24h',
     'variant': 'T4_lookup_clim_full', 'p_star': None},
    {'id': 'rain_occurrence_6h', 'family': 'lh', 'variable': 'rainfall', 'target': '6h', 'key': '6h',
     'variant': 'R3_harmonic_prob_weather_cond', 'p_star': 0.275},
    {'id': 'rain_occurrence_24h', 'family': 'lh', 'variable': 'rainfall', 'target': '24h', 'key': '24h',
     'variant': 'R4_lookup_full', 'p_star': 0.25},
    {'id': 'rain_total_24h', 'family': 'c2', 'variable': 'rainfall', 'target': '24h_total', 'key': '24h_total',
     'variant': 'C2', 'reference': 'A0', 'p_star': None},
]
LOCKED_FILES = {
    'protocol': PROTOCOL,
    'artifact_long_horizon_history': lh.ARTIFACT,
    'artifact_rain24_total': rx.ARTIFACT,
    'summary_long_horizon_history': lh.SUMMARY_CSV,
    'summary_rain24_total': rx.SUMMARY_JSON,
    'criteria_long_horizon_history': lh.CRITERIA,
    'criteria_rain24_total': rx.CRITERIA,
    'history_manifest': os.path.relpath(omh.HISTORY_MANIFEST, ROOT),
    'code_final_eval_2024': os.path.join('src', 'models', 'final_eval_2024.py'),
    'code_long_horizon_history': os.path.join('src', 'models', 'long_horizon_history.py'),
    'code_rain24_total_experiment': os.path.join('src', 'models', 'rain24_total_experiment.py'),
    'code_train_and_validate_hmm': os.path.join('src', 'models', 'train_and_validate_hmm.py'),
    'code_rain_model': os.path.join('src', 'models', 'rain_model.py'),
    'code_temp_model': os.path.join('src', 'models', 'temp_model.py'),
    'code_encoding': os.path.join('src', 'models', 'encoding.py'),
    'code_open_meteo_history': os.path.join('src', 'collectors', 'open_meteo_history.py'),
}
PENDING_RE = re.compile(r'PENDING DECISION (D\d+)')
REPRO_RTOL, REPRO_ATOL = 1e-9, 1e-9


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def sha256(path: str) -> str:
    with open(_abs(path), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── 1. C2 compatibility loader (scoped; no sys.modules changes) ─────────────
C2_MAIN_CLASSES = frozenset({'A0Existing', 'DirectTweedie', 'HurdleTotal', 'RatioDistribution', 'AggregatedHourly'})
C2_MODULE = 'src.models.rain24_total_experiment'


def _remap_unpickler_class():
    from joblib import numpy_pickle as jnp

    class MainRemapUnpickler(jnp.NumpyUnpickler):
        """joblib unpickler that resolves an explicit allow-list of `__main__.<name>` globals from `target`."""
        allowed: frozenset = frozenset()
        target: str = ''

        def find_class(self, module, name):
            if module == '__main__':
                if name not in self.allowed:
                    raise pickle.UnpicklingError(f'refusing to resolve __main__.{name} (not in the allow-list)')
                module = self.target
            return super().find_class(module, name)
    return MainRemapUnpickler, jnp


def load_with_main_remap(path: str, target: str = C2_MODULE, allowed=C2_MAIN_CLASSES):
    """Load a joblib file whose classes were pickled from a script run as __main__, resolving only the
    allowed class names from their real module. Nothing global is modified."""
    cls, jnp = _remap_unpickler_class()
    path = _abs(path)
    with open(path, 'rb') as f:
        with jnp._validate_fileobject_and_memmap(f, path, None) as (fobj, _):
            u = cls(path, fobj, True)
            u.allowed, u.target = frozenset(allowed), target
            return u.load()


def load_with_main_alias(path: str):
    """Reference method (verification only): temporarily expose the classes on __main__, load, then remove them."""
    import joblib
    main = sys.modules['__main__']
    added = [n for n in C2_MAIN_CLASSES if not hasattr(main, n)]
    for n in added:
        setattr(main, n, getattr(rx, n))
    try:
        return joblib.load(_abs(path))
    finally:
        for n in added:
            delattr(main, n)


def load_artifacts() -> Dict[str, Any]:
    """Frozen artifacts. Raises (blocker) if a frozen model is missing or of an unexpected type."""
    import joblib
    from src.models.rain_model import TwoPartRainModel
    from src.models.temp_model import HourlyTempModel
    lha = joblib.load(_abs(lh.ARTIFACT))
    c2a = load_with_main_remap(rx.ARTIFACT)
    for cfg in CONFIGS:
        if cfg['family'] == 'lh':
            k = (cfg['variable'], cfg['key'])
            sel = lha['selections'][k]['selected']
            if sel != cfg['variant']:
                raise RuntimeError(f"BLOCKER {cfg['id']}: artifact selection {sel} != frozen {cfg['variant']}")
            want = HourlyTempModel if cfg['variable'] == 'temperature' else TwoPartRainModel
            if not isinstance(lha['models'][k], want):
                raise RuntimeError(f"BLOCKER {cfg['id']}: unexpected model type {type(lha['models'][k])}")
    if not isinstance(c2a['models'].get('C2'), rx.HurdleTotal) or not isinstance(c2a['models'].get('A0'), rx.A0Existing):
        raise RuntimeError('BLOCKER rain_total_24h: C2/A0 not found in the rain24_total artifact')
    if c2a['summary']['validated_candidate'] != 'C2' or not c2a['summary']['validation_gate']['passed']:
        raise RuntimeError('BLOCKER rain_total_24h: artifact summary does not record a validated C2')
    summ = pd.read_csv(_abs(lh.SUMMARY_CSV), float_precision='round_trip')
    for cfg in CONFIGS:
        if cfg['family'] == 'lh' and cfg['variable'] == 'rainfall':
            row = summ[(summ['variable'] == 'rainfall') & (summ['horizon'] == cfg['key']) & (summ['role'] == 'selected')].iloc[0]
            if row['variant'] != cfg['variant'] or float(row['cv_p_star_from_folds']) != cfg['p_star']:
                raise RuntimeError(f"BLOCKER {cfg['id']}: summary threshold/variant differ from the frozen config")
    return {'lh': lha, 'c2': c2a}


# ── 2. frames (2017-2023 + 2024, December 2023 as lead-in) ──────────────────
def final_cell_frames(cells: pd.DataFrame, dev: pd.DataFrame, sealed: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per unique grid cell: one continuous hourly frame 2017..2024 (same representative locations as the
    experiments), so features at the start of 2024 use December 2023 history. No other years allowed."""
    frames = {}
    boundary = (pd.Timestamp('2023-12-31 23:00', tz='UTC'), pd.Timestamp('2024-01-01 00:00', tz='UTC'))
    for _, c in cells.iterrows():
        parts = [d[d['location_key'] == c['rep_location_key']].set_index('time_utc')[lh.VARS] for d in (dev, sealed)]
        g = pd.concat(parts).sort_index()
        g = g[~g.index.duplicated()]
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq='h'))
        yrs = set(g.index.year)
        if not yrs <= set(range(2017, 2025)):
            raise RuntimeError(f'unexpected years in frame {c["cell"]}: {sorted(yrs)}')
        if not all(t in g.index and g.loc[t, 'precipitation_mm'] == g.loc[t, 'precipitation_mm'] for t in boundary):
            raise RuntimeError(f'no continuous 2023/2024 boundary in frame {c["cell"]} (lead-in history missing)')
        frames[c['cell']] = g
    return frames


# ── 3. scoring with the frozen models (existing evaluation utilities) ───────
def score_lh(cfg: Dict[str, Any], frames, model, splits, ev: str) -> Dict[str, Any]:
    df, ph = lh._frames_for(cfg['variable'], frames, cfg['target'], splits)
    feats = lh._variants(cfg['variable'], df)[cfg['variant']]
    have = list(getattr(model, 'features', None) or model.feature_names)
    if have != list(feats):
        raise RuntimeError(f"BLOCKER {cfg['id']}: frozen model features differ from the rebuilt feature list")
    if cfg['variable'] == 'temperature':
        _, m, s, _ = lh.eval_temp(df, cfg['target'], feats, ev, model=model)
    else:
        _, m, s, _ = lh.eval_rain(df, ph, cfg['key'], feats, ev, cfg['p_star'], model=model)
    return {'metrics': m, 'summary': s, 'gate': lh.validation_pass(cfg['variable'], s, True)}


def score_c2(frames, models: Dict[str, Any], splits, ev: str) -> Dict[str, Any]:
    df = rx.make_split(rx.build_base(frames), splits)
    fs = rx.feature_sets(df)
    if list(models['C2'].feats) != fs['ext'] or list(models['A0'].feats) != fs['prod']:
        raise RuntimeError('BLOCKER rain_total_24h: frozen C2/A0 features differ from the rebuilt feature lists')
    res = rx.evaluate(df, ev, {'A0': models['A0'], 'C2': models['C2']}, detail=True)
    return {'metrics': res['rows'], 'reliability': res['reliability'], 'gate': rx.validation_gate(res['rows'], 'C2')}


# ── 4. preflight: reproduce the frozen models from 2017-2023 only ───────────
def _params(model) -> Dict[str, Any]:
    if hasattr(model, 'coef_') and hasattr(model, 'resid_std_'):                          # HourlyTempModel
        return {'coef': np.asarray(model.coef_), 'resid_std': model.resid_std_}
    out = {'coef': np.asarray(model.coef_), 'intercept': model.intercept_, 'mean': np.asarray(model.mean_),
           'scale': np.asarray(model.scale_), 'gamma_overall': np.asarray(model.gamma_overall or (np.nan, np.nan))}
    out.update({f'gamma_{k}': np.asarray(v) for k, v in sorted(model.gamma_by_season.items())})
    return out


def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a.keys() == b.keys() and all(np.allclose(a[k], b[k], rtol=REPRO_RTOL, atol=REPRO_ATOL, equal_nan=True) for k in a)


def preflight(artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """No 2024 access. Rebuild 2017-2023 frames, refit each frozen configuration in memory and compare with
    the saved model (parameters / predictions). Nothing is saved except the preparation record."""
    from src.models.rain_model import TwoPartRainModel
    from src.models.temp_model import HourlyTempModel
    art = artifacts or load_artifacts()
    dev = omh.load_history(include_sealed=False)
    if dev['time_utc'].dt.year.max() >= 2024:
        raise RuntimeError('development data contain 2024 rows')
    cells = lh.grid_cells(dev)
    frames = lh.cell_frames(cells, dev)                                       # asserts no 2024 values
    out = {}
    for cfg in CONFIGS:
        if cfg['family'] == 'lh':
            df, _ = lh._frames_for(cfg['variable'], frames, cfg['target'], PREFLIGHT_SPLITS)
            feats = lh._variants(cfg['variable'], df)[cfg['variant']]
            tr = df[df['split'] == 'train']
            saved = art['lh']['models'][(cfg['variable'], cfg['key'])]
            if cfg['variable'] == 'temperature':
                fresh = HourlyTempModel(cfg['target'], feature_names=feats).fit(tr[feats], tr['y'].to_numpy())
            else:
                fresh = TwoPartRainModel(cfg['key'], feature_names=feats).fit(tr[feats], tr['y'].to_numpy(), tr['season'].to_numpy())
            same_feats = list(getattr(saved, 'features', None) or saved.feature_names) == list(feats)
            out[cfg['id']] = {'reproduced': bool(same_feats and _same(_params(saved), _params(fresh))),
                              'features_match': same_feats, 'n_train_rows': int(len(tr)), 'n_features': len(feats)}
        else:
            df = rx.make_split(rx.build_base(frames), PREFLIGHT_SPLITS)
            fs = rx.feature_sets(df)
            tr = df[df['split'] == 'train']
            sample = tr.iloc[::97]
            rec = {'n_train_rows': int(len(tr)), 'n_compared_rows': int(len(sample))}
            ok = True
            for name in ('A0', 'C2'):
                fresh = rx.make_candidate(name, fs).fit(tr)
                ps, pf = art['c2']['models'][name].predict(sample), fresh.predict(sample)
                same = all(np.allclose(ps[k], pf[k], rtol=1e-7, atol=1e-9) for k in ('mean', 'median', 'p25', 'Q'))
                rec[f'{name}_reproduced'] = bool(same)
                ok = ok and same
            c2 = art['c2']['models']['C2']
            rec['C2_no_amount_correction'] = bool(not c2.calibrate and not c2.recent and all(v == 1.0 for v in c2.factor.values()))
            ok = ok and rec['C2_no_amount_correction']
            rec['reproduced'] = bool(ok)
            out[cfg['id']] = rec
    return {'checked_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'data': 'development 2017-2023 only (2024 not opened)',
            'results': out, 'all_reproduced': all(v['reproduced'] for v in out.values())}


# ── 5. protocol lock ────────────────────────────────────────────────────────
def pending_decisions(path: Optional[str] = None) -> List[str]:
    with open(_abs(path or PROTOCOL), encoding='utf-8') as f:
        return sorted(set(PENDING_RE.findall(f.read())))


def current_hashes() -> Dict[str, str]:
    return {k: sha256(p) for k, p in LOCKED_FILES.items()}


def freeze_protocol() -> Dict[str, Any]:
    if os.path.exists(_abs(LOCK)):
        raise FileExistsError('protocol lock already exists - a frozen protocol is never re-frozen')
    pend = pending_decisions()
    if pend:
        raise RuntimeError(f'protocol not frozen: pending decisions {pend}')
    lock = {'frozen': True, 'frozen_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'protocol_version': 'v1',
            'configs': CONFIGS, 'final_splits': FINAL_SPLITS, 'sha256': current_hashes()}
    with open(_abs(LOCK), 'w', encoding='utf-8') as f:
        json.dump(lock, f, indent=2, default=str)
    return lock


def verify_lock() -> Dict[str, Any]:
    if not os.path.exists(_abs(LOCK)):
        raise PermissionError('protocol is not frozen (no lock) - the sealed year stays closed')
    with open(_abs(LOCK), encoding='utf-8') as f:
        lock = json.load(f)
    if lock.get('frozen') is not True:
        raise PermissionError('protocol lock is not marked frozen')
    if pending_decisions():
        raise PermissionError('protocol contains pending decisions')
    now = current_hashes()
    bad = sorted(k for k, v in lock['sha256'].items() if now.get(k) != v) + sorted(set(now) - set(lock['sha256']))
    if bad:
        raise PermissionError(f'locked files changed since freezing: {bad}')
    if json.loads(json.dumps(CONFIGS, default=str)) != lock['configs']:
        raise PermissionError('frozen configurations changed since freezing')
    return lock


# ── 6. guarded single run ───────────────────────────────────────────────────
_SEALED_STATE = {'allowed': False, 'hook': False}


def sealed_guard(event: str, args: tuple) -> None:
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    if '2024_SEALED' in os.fsdecode(args[0]) and not _SEALED_STATE['allowed']:
        raise PermissionError('final_eval_2024: the sealed file may only be opened inside the approved run')


def _install_guard():
    if not _SEALED_STATE['hook']:
        sys.addaudithook(sealed_guard)
        _SEALED_STATE['hook'] = True


def run_final(reason: str, env: Optional[Dict[str, str]] = None, save: bool = True) -> Dict[str, Any]:
    env = os.environ if env is None else env
    lock = verify_lock()                                                      # 1. frozen protocol + hashes
    if not reason or not reason.strip():                                      # 2. reason
        raise PermissionError('a non-empty --reason is required')
    if env.get(omh.UNSEAL_ENV) != omh.UNSEAL_TOKEN:                           # 3. explicit unseal guard
        raise PermissionError(f'{omh.UNSEAL_ENV} is not set to the unseal token')
    if any(os.path.exists(_abs(p)) for p in (FINAL_RESULTS, FINAL_METRICS, FINAL_REPORT)):
        raise FileExistsError('final results already exist - the final evaluation runs once')
    _install_guard()
    art = load_artifacts()
    pf = preflight(art)                                                       # 4. frozen models reproduced
    if not pf['all_reproduced']:
        raise RuntimeError(f"BLOCKER: frozen models not reproduced: {pf['results']}")
    _SEALED_STATE['allowed'] = True
    try:
        full = omh.load_history(include_sealed=True, reason=reason.strip())  # logs the access
    finally:
        _SEALED_STATE['allowed'] = False
    yr = full['time_utc'].dt.year
    dev, sealed = full[yr < 2024], full[yr == 2024]
    cells = lh.grid_cells(dev)
    frames = final_cell_frames(cells, dev, sealed)
    results, rows = {}, []
    for cfg in CONFIGS:
        if cfg['family'] == 'lh':
            r = score_lh(cfg, frames, art['lh']['models'][(cfg['variable'], cfg['key'])], FINAL_SPLITS, 'test')
            results[cfg['id']] = {'summary': r['summary'], 'gate': r['gate']}
            rows.append(r['metrics'].assign(config=cfg['id']))
        else:
            r = score_c2(frames, art['c2']['models'], FINAL_SPLITS, 'test')
            a = r['metrics']
            results[cfg['id']] = {'summary': a[a['group'] == 'ALL'].set_index('candidate').to_dict('index'), 'gate': r['gate']}
            rows.append(a.assign(config=cfg['id']))
    out = {'run_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'reason': reason.strip(), 'lock': lock,
           'preflight': pf, 'results': results}
    if save:
        pd.concat(rows, ignore_index=True).to_csv(_abs(FINAL_METRICS), index=False)
        with open(_abs(FINAL_RESULTS), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
        with open(_abs(FINAL_REPORT), 'w', encoding='utf-8') as f:
            f.write(_final_report(out))
    return out


def _final_report(out: Dict[str, Any]) -> str:
    L = ['# Final independent evaluation on 2024 (single run)', '', f"Run {out['run_utc']} · reason: {out['reason']}",
         f"Protocol frozen {out['lock']['frozen_utc']} (SHA-256 {out['lock']['sha256']['protocol']}).", '',
         '| configuration | passes | checks |', '|---|---|---|']
    for k, v in out['results'].items():
        g = {kk: vv for kk, vv in v['gate'].items() if kk != 'passed'}
        L.append(f"| {k} | {v['gate']['passed']} | {json.dumps(g, default=str)} |")
    return '\n'.join(L) + '\n'


def write_preparation_record(pf: Dict[str, Any]) -> Dict[str, Any]:
    rec = {'written_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'sealed_2024_opened': False,
           'protocol_pending_decisions': pending_decisions(), 'protocol_frozen': os.path.exists(_abs(LOCK)),
           'sha256': current_hashes(), 'configs': CONFIGS, 'preflight': pf}
    os.makedirs(_abs(OUT), exist_ok=True)
    with open(_abs(PREP_RECORD), 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, default=str)
    return rec


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Final 2024 evaluation (preparation by default; sealed run is guarded)')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--preflight', action='store_true', help='default: no 2024 access')
    g.add_argument('--freeze-protocol', action='store_true')
    g.add_argument('--run', action='store_true', help='the single approved final evaluation (opens 2024)')
    ap.add_argument('--reason', default='')
    a = ap.parse_args()
    if a.freeze_protocol:
        print(json.dumps(freeze_protocol(), indent=2, default=str))
    elif a.run:
        print(json.dumps(run_final(a.reason)['results'], indent=2, default=str))
    else:
        _install_guard()
        _SEALED_STATE['allowed'] = False
        rec = write_preparation_record(preflight())
        print(json.dumps({k: rec[k] for k in ('protocol_pending_decisions', 'protocol_frozen', 'preflight')}, indent=2, default=str))
