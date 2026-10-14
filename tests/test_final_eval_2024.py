"""
Final 2024 evaluation preparation: scoped C2 compatibility loader, final frames with December-2023 lead-in,
training-only climatologies, reuse of the existing scorers with frozen models, protocol locking, and the sealed
access guard. Synthetic data only; the sealed file is never opened (load_history is replaced by a tripwire).
"""

import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collectors import open_meteo_history as omh
from src.models import final_eval_2024 as fe
from src.models import long_horizon_history as lh
from src.models import rain24_total_experiment as rx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CELLS = [f'gc{i}' for i in range(1, 9)]


def _frame(start, hours, seed):
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    rng = np.random.default_rng(seed)
    ih = idx.tz_convert('Asia/Kolkata').hour.to_numpy()
    return pd.DataFrame({'temperature_2m_c': 28 + 3 * np.sin(2 * np.pi * ih / 24) + rng.normal(0, 0.6, hours),
                         'relative_humidity_2m_pct': 70 + rng.normal(0, 5, hours),
                         'precipitation_mm': np.where(rng.random(hours) < 0.12, rng.gamma(0.7, 2.0, hours), 0.0),
                         'wind_speed_10m_kmh': 10 + rng.random(hours), 'wind_direction_10m_deg': rng.random(hours) * 360},
                        index=idx)


@pytest.fixture(scope='module')
def rx_df():
    frames = {c: _frame('2019-01-01', 24 * 400, i) for i, c in enumerate(CELLS)}
    splits = {'train': ('2019-01-01 00:00', '2019-10-31 23:00'), 'test': ('2019-11-01 00:00', '2020-02-03 23:00')}
    return rx.make_split(rx.build_base(frames), splits)


# ── compatibility loader ─────────────────────────────────────────────────────
def _dump_as_main(obj, path, monkeypatch):
    """Pickle `obj` as if its classes lived in a script run as __main__ (how the C2 artifact was written)."""
    main = sys.modules['__main__']
    for n in fe.C2_MAIN_CLASSES:
        cls = getattr(rx, n)
        monkeypatch.setattr(cls, '__module__', '__main__')
        monkeypatch.setattr(main, n, cls, raising=False)
    joblib.dump(obj, path)
    monkeypatch.undo()


def test_remap_loader_resolves_main_pickles_without_touching_sys_modules(rx_df, tmp_path, monkeypatch):
    fs = rx.feature_sets(rx_df)
    tr, te = rx_df[rx_df['split'] == 'train'], rx_df[rx_df['split'] == 'test']
    models = {'A0': rx.make_candidate('A0', fs).fit(tr), 'C2': rx.make_candidate('C2', fs).fit(tr)}
    p = tmp_path / 'c2.joblib'
    _dump_as_main({'models': models}, str(p), monkeypatch)
    assert b'__main__' in p.read_bytes()
    with pytest.raises(AttributeError):
        joblib.load(str(p))                                                 # the original failure
    main_before = {n: hasattr(sys.modules['__main__'], n) for n in fe.C2_MAIN_CLASSES}
    a = fe.load_with_main_remap(str(p))
    b = fe.load_with_main_alias(str(p))
    assert {n: hasattr(sys.modules['__main__'], n) for n in fe.C2_MAIN_CLASSES} == main_before   # nothing left behind
    for name in ('A0', 'C2'):
        assert type(a['models'][name]) is type(models[name]) is type(b['models'][name])
        pa, pb, po = (m['models'][name].predict(te) if isinstance(m, dict) else m.predict(te) for m in (a, b, models[name]))
        for k in ('mean', 'median', 'p25', 'Q'):
            np.testing.assert_array_equal(pa[k], pb[k])
            np.testing.assert_array_equal(pa[k], po[k])


def test_remap_loader_refuses_classes_outside_the_allow_list(tmp_path, monkeypatch):
    Stranger = type('Stranger', (), {'__module__': '__main__'})
    monkeypatch.setattr(sys.modules['__main__'], 'Stranger', Stranger, raising=False)
    p = tmp_path / 'x.joblib'
    joblib.dump(Stranger(), str(p))
    monkeypatch.undo()
    with pytest.raises(Exception, match='allow-list'):
        fe.load_with_main_remap(str(p))


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, rx.ARTIFACT)), reason='no C2 artifact')
def test_real_c2_artifact_matches_the_alias_reference(rx_df):
    a, b = fe.load_with_main_remap(rx.ARTIFACT), fe.load_with_main_alias(rx.ARTIFACT)
    assert isinstance(a['models']['C2'], rx.HurdleTotal) and isinstance(a['models']['A0'], rx.A0Existing)
    te = rx_df[rx_df['split'] == 'test']
    for name in ('A0', 'C2'):
        ma, mb = a['models'][name], b['models'][name]
        assert ma.feats == mb.feats and len(ma.feats) == (37 if name == 'C2' else 17)
        pa, pb = ma.predict(te), mb.predict(te)
        for k in ('mean', 'median', 'p25', 'Q'):
            np.testing.assert_array_equal(pa[k], pb[k])
    np.testing.assert_array_equal(a['models']['C2'].l2.coef_, b['models']['C2'].l2.coef_)
    assert a['summary']['validated_candidate'] == 'C2'


# ── frames, lead-in, leakage ─────────────────────────────────────────────────
def _hist_rows(frames, years_mask):
    rows = []
    for key, f in frames.items():
        g = f[years_mask(f.index)].reset_index().rename(columns={'index': 'time_utc'})
        g['location_key'] = key
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def test_final_frames_are_continuous_and_use_december_2023_as_lead_in():
    frames = {'zone_02': _frame('2023-11-01', 24 * 120, 1), 'zone_04': _frame('2023-11-01', 24 * 120, 2)}
    dev = _hist_rows(frames, lambda i: i.year < 2024)
    sealed = _hist_rows(frames, lambda i: i.year == 2024)
    cells = pd.DataFrame({'cell': ['gc1', 'gc2'], 'rep_location_key': ['zone_02', 'zone_04']})
    out = fe.final_cell_frames(cells, dev, sealed)
    f = out['gc1']
    assert f.index.min().year == 2023 and f.index.max().year == 2024 and f.notna().all().all()
    feats = rx.cell_features(f)
    t0 = pd.Timestamp('2024-01-01 00:00', tz='UTC')
    p = f['precipitation_mm']
    assert np.isfinite(feats.loc[t0, 'log_rain_7d'])                         # needs 168 h of December 2023
    assert feats.loc[t0, 'log_rain_7d'] == pytest.approx(np.log1p(p.loc[t0 - pd.Timedelta(hours=167):t0].sum()))
    with pytest.raises(RuntimeError, match='boundary'):
        fe.final_cell_frames(cells, dev, sealed[sealed['time_utc'] >= pd.Timestamp('2024-01-02', tz='UTC')])


def test_climatology_features_ignore_2024_targets():
    frames = {c: _frame('2023-01-01', 24 * 540, i) for i, c in enumerate(CELLS[:2])}
    base = rx.build_base(frames)
    a = rx.make_split(base, fe.FINAL_SPLITS)
    b2 = base.copy()
    b2.loc[b2.index >= pd.Timestamp('2024-01-01', tz='UTC'), 'y'] = 500.0
    b = rx.make_split(b2, fe.FINAL_SPLITS)
    np.testing.assert_allclose(a[rx.CLIM_FEATS].to_numpy(), b[rx.CLIM_FEATS].to_numpy())
    test = a[a['split'] == 'test']
    assert len(test) and test.index.min() >= pd.Timestamp('2024-01-01', tz='UTC')
    assert (test.index + pd.Timedelta(hours=24) <= pd.Timestamp('2024-12-31 23:00', tz='UTC')).all()


# ── scoring reuses the existing utilities with frozen models ─────────────────
def _same_summary(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and np.isnan(a[k]):
            assert isinstance(b[k], float) and np.isnan(b[k]), k
        else:
            assert a[k] == b[k], k


def test_frozen_model_scoring_equals_the_existing_fit_and_score_path():
    frames = {c: _frame('2018-01-01', 24 * 500, i) for i, c in enumerate(CELLS[:3])}
    splits = {'train': ('2018-01-01 00:00', '2018-12-31 23:00'), 'test': ('2019-01-01 00:00', '2019-05-14 23:00')}
    df, _ = lh._frames_for('temperature', frames, 6, splits)
    feats = lh._variants('temperature', df)['T4_lookup_clim_full']
    m1, met1, s1, _ = lh.eval_temp(df, 6, feats, 'test')
    _, met2, s2, _ = lh.eval_temp(df, 6, feats, 'test', model=m1)
    _same_summary(s1, s2)
    pd.testing.assert_frame_equal(met1.reset_index(drop=True), met2.reset_index(drop=True))
    dfr, ph = lh._frames_for('rainfall', frames, '6h', splits)
    fr = lh._variants('rainfall', dfr)['R3_harmonic_prob_weather_cond']
    r1, _, rs1, _ = lh.eval_rain(dfr, ph, '6h', fr, 'test', 0.275)
    _, _, rs2, _ = lh.eval_rain(dfr, ph, '6h', fr, 'test', 0.275, model=r1)
    _same_summary(rs1, rs2)
    cfg = dict(fe.CONFIGS[0])
    out = fe.score_lh(cfg, frames, m1, splits, 'test')
    _same_summary(out['summary'], s1)
    assert set(out['gate']) >= {'passed', 'skill_positive', 'week_ci_low_positive', 'cells_positive'}
    bad = dict(cfg, variant='T0_production')
    with pytest.raises(RuntimeError, match='BLOCKER'):
        fe.score_lh(bad, frames, m1, splits, 'test')                       # never silently substitutes


# ── protocol lock ────────────────────────────────────────────────────────────
@pytest.fixture
def locked_env(tmp_path, monkeypatch):
    files = {}
    for k, p in fe.LOCKED_FILES.items():
        dst = tmp_path / k
        if k == 'protocol':
            dst.write_text('# protocol\nall decided\n', encoding='utf-8')
        else:
            dst.write_bytes(b'content of ' + k.encode())
        files[k] = str(dst)
    monkeypatch.setattr(fe, 'LOCKED_FILES', files)
    monkeypatch.setattr(fe, 'PROTOCOL', files['protocol'])
    monkeypatch.setattr(fe, 'LOCK', str(tmp_path / 'lock.json'))
    for n in ('FINAL_RESULTS', 'FINAL_METRICS', 'FINAL_REPORT'):
        monkeypatch.setattr(fe, n, str(tmp_path / n))
    return files


def test_freeze_refused_while_decisions_are_pending(locked_env):
    with open(locked_env['protocol'], 'a', encoding='utf-8') as f:
        f.write('PENDING DECISION D1 - pass rule\n')
    assert fe.pending_decisions() == ['D1']
    with pytest.raises(RuntimeError, match='pending'):
        fe.freeze_protocol()
    assert not os.path.exists(fe.LOCK)


def test_lock_detects_changed_protocol_artifact_or_config(locked_env, monkeypatch):
    lock = fe.freeze_protocol()
    assert lock['frozen'] and fe.verify_lock()['sha256'] == lock['sha256']
    with pytest.raises(FileExistsError):
        fe.freeze_protocol()                                                # never re-frozen
    with open(locked_env['artifact_rain24_total'], 'ab') as f:
        f.write(b'x')
    with pytest.raises(PermissionError, match='artifact_rain24_total'):
        fe.verify_lock()


def test_lock_detects_changed_configuration(locked_env, monkeypatch):
    fe.freeze_protocol()
    monkeypatch.setattr(fe, 'CONFIGS', [dict(c, p_star=0.3) if c['id'] == 'rain_occurrence_6h' else c for c in fe.CONFIGS])
    with pytest.raises(PermissionError, match='configurations changed'):
        fe.verify_lock()


def test_real_protocol_has_no_pending_decisions_and_any_lock_verifies():
    assert fe.pending_decisions() == []
    text = open(os.path.join(ROOT, fe.PROTOCOL), encoding='utf-8').read()
    for phrase in ('ELIGIBLE FOR A SEPARATE PROMOTION REVIEW', 'No multiplicity adjustment', 'conditional_climatology'):
        assert phrase in text
    if os.path.exists(os.path.join(ROOT, fe.LOCK)):
        lock = fe.verify_lock()                                             # frozen hashes still match
        assert lock['frozen'] and lock['sha256']['protocol'] == fe.sha256(fe.PROTOCOL)


# ── sealed access guard ──────────────────────────────────────────────────────
@pytest.fixture
def tripwire(monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append((a, k))
        raise AssertionError('sealed data access attempted')
    monkeypatch.setattr(omh, 'load_history', boom)
    return calls


def test_run_refused_without_a_frozen_protocol(tripwire, monkeypatch, tmp_path):
    monkeypatch.setattr(fe, 'LOCK', str(tmp_path / 'missing.json'))
    with pytest.raises(PermissionError, match='not frozen'):
        fe.run_final('final evaluation', env={omh.UNSEAL_ENV: omh.UNSEAL_TOKEN}, save=False)
    assert tripwire == []


@pytest.mark.parametrize('reason,env', [('', {omh.UNSEAL_ENV: omh.UNSEAL_TOKEN}), ('   ', {omh.UNSEAL_ENV: omh.UNSEAL_TOKEN}),
                                        ('final evaluation', {}), ('final evaluation', {omh.UNSEAL_ENV: 'yes'})])
def test_run_refused_without_reason_or_unseal_token(reason, env, tripwire, locked_env):
    fe.freeze_protocol()
    with pytest.raises(PermissionError):
        fe.run_final(reason, env=env, save=False)
    assert tripwire == []


def test_run_refused_when_final_results_already_exist(tripwire, locked_env):
    fe.freeze_protocol()
    open(fe.FINAL_RESULTS, 'w').close()
    with pytest.raises(FileExistsError):
        fe.run_final('final evaluation', env={omh.UNSEAL_ENV: omh.UNSEAL_TOKEN}, save=False)
    assert tripwire == []


def test_audit_guard_blocks_the_sealed_file_unless_inside_the_run():
    assert fe._SEALED_STATE['allowed'] is False
    with pytest.raises(PermissionError):
        fe.sealed_guard('open', (omh.HISTORY_SEALED_FILE, 'rb', 0))
    fe.sealed_guard('open', (omh.HISTORY_DEV_FILE, 'rb', 0))
    fe._SEALED_STATE['allowed'] = True
    try:
        fe.sealed_guard('open', (omh.HISTORY_SEALED_FILE, 'rb', 0))
    finally:
        fe._SEALED_STATE['allowed'] = False


def test_configs_match_frozen_selections():
    ids = [c['id'] for c in fe.CONFIGS]
    assert ids == ['temperature_6h', 'temperature_24h', 'rain_occurrence_6h', 'rain_occurrence_24h', 'rain_total_24h']
    assert fe.FINAL_SPLITS == {'train': ('2017-01-01 00:00', '2023-12-31 23:00'), 'test': ('2024-01-01 00:00', '2024-12-31 23:00')}
    src = open(os.path.join(ROOT, 'src', 'models', 'final_eval_2024.py'), encoding='utf-8').read()
    assert src.count('include_sealed=True') == 2                            # docstring + the single guarded call
