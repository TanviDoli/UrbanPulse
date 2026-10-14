"""
Long-horizon (6 h / 24 h) experiment: harmonic climatologies are training-only, CV folds never touch
the test period, the pre-specified selection rule, backward-compatible split/baseline hooks, and
separation from production artifacts. Uses synthetic data; no model is fitted on project data here.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _frame(days=40, start='2025-11-10'):
    idx = pd.date_range(start, periods=24 * days, freq='h', tz='UTC')
    ist_h = idx.tz_convert('Asia/Kolkata').hour.to_numpy()
    df = pd.DataFrame({'cell': 'c1', 'y': 28 + 3 * np.sin(2 * np.pi * ist_h / 24), 't_now': 27.0}, index=idx)
    df['split'] = np.where(idx < pd.Timestamp('2025-12-01', tz='UTC'), 'train', 'validate')
    return df


def test_harmonic_design_shape_constant_and_periodicity():
    t = pd.DatetimeIndex([pd.Timestamp('2025-03-01 06:00', tz='UTC'), pd.Timestamp('2026-03-01 06:00', tz='UTC')])
    X = tv.harmonic_design(t)
    assert X.shape == (2, 13) and np.all(X[:, 0] == 1)
    assert np.allclose(X[0, 5:9], X[1, 5:9])                         # same IST hour -> same diurnal terms
    assert np.allclose(X[0], X[1], atol=0.02)                        # one year apart -> (almost) same row


def test_harmonic_climatologies_use_training_rows_only():
    df = _frame()
    tv._harmonic_temperature(df, 6)
    leak = _frame()
    leak.loc[leak['split'] == 'validate', 'y'] += 50.0               # validation targets must not matter
    tv._harmonic_temperature(leak, 6)
    np.testing.assert_allclose(df['hclim_target'], leak['hclim_target'])
    o = (df['y'] > 29).to_numpy()
    o_leak = o.copy()
    o_leak[(df['split'] == 'validate').to_numpy()] = True
    np.testing.assert_allclose(tv._harmonic_probability(df, o, 6), tv._harmonic_probability(df, o_leak, 6))
    const = tv._harmonic_probability(df, np.zeros(len(df), dtype=bool), 6)
    assert np.allclose(const, 0.0)                                    # one-class training -> constant


def test_cv_folds_never_include_the_test_period():
    test_start = pd.Timestamp(tv.RAIN_OM_SPLITS['test'][0], tz='UTC')
    for fold in tv.LH_FOLDS.values():
        for lo, hi in fold.values():
            assert pd.Timestamp(hi, tz='UTC') < test_start
        assert pd.Timestamp(fold['train'][1], tz='UTC') < pd.Timestamp(fold['validate'][0], tz='UTC')   # expanding, ordered
    assert pd.Timestamp(tv.LH_UNTOUCHED_START, tz='UTC') > pd.Timestamp(tv.RAIN_OM_SPLITS['test'][1], tz='UTC')


def test_selection_rule_is_the_pre_specified_one():
    cv = pd.DataFrame({'variant': ['a'] * 3 + ['b'] * 3 + ['c'] * 3,
                       'skill': [0.5, -0.9, -0.9,                    # highest single fold, but 1/3 positive
                                 0.1, 0.1, -0.05,                    # 2/3 positive, mean 0.05
                                 0.02, 0.03, 0.01]})                  # 3/3 positive, mean 0.02
    sel, info = tv._lh_select(cv)
    assert sel == 'b' and info['eligible'] and info['positive_folds'] == 2
    sel2, info2 = tv._lh_select(pd.DataFrame({'variant': ['x', 'x', 'x', 'y', 'y', 'y'],
                                             'skill': [-0.1, -0.2, 0.1, -0.3, -0.3, -0.3]}))
    assert sel2 == 'x' and info2['eligible'] is False                # none eligible -> best mean, flagged


def test_split_and_baseline_hooks_are_backward_compatible():
    idx = pd.date_range(pd.Timestamp(tv.RAIN_OM_SPLITS['validate'][0], tz='UTC') - pd.Timedelta(days=3),
                        periods=24 * 6, freq='h')
    cells = {'cell_1': pd.Series(np.where(np.arange(len(idx)) % 7 < 2, 1.0, 0.0), index=idx)}
    a = tv.twopart_rows(cells, '6h')
    b = tv.twopart_rows(cells, '6h', splits=tv.RAIN_OM_SPLITS)
    pd.testing.assert_frame_equal(a, b)
    custom = {'train': (str(idx[0].tz_localize(None)), str(idx[60].tz_localize(None))),
              'validate': (str(idx[61].tz_localize(None)), str(idx[-1].tz_localize(None)))}
    c = tv.twopart_rows(cells, '6h', splits=custom)
    assert set(c['split']) == {'train', 'validate'} and c[c['split'] == 'train'].index.max() <= idx[60] - pd.Timedelta(hours=6)
    df = _frame()
    df['y'] = df['y'] + 0.01 * np.arange(len(df))                   # trend: no simple baseline is perfect
    df['t_same_hour_yday'] = df['y'].shift(24).bfill()
    df['target_month'], df['target_hour'] = df.index.month, df.index.hour
    df['day'], df['season'] = df.index.strftime('%Y-%m-%d'), 'ne_monsoon'
    pred = df['y'].to_numpy() + 0.5
    base = tv.evaluate_temp(df, pred, '6h')
    perfect = tv.evaluate_temp(df, pred, '6h', extra_baselines={'harmonic_climatology': df['y'].to_numpy()})
    assert 'best_baseline' in base and (perfect['best_baseline'] == 'harmonic_climatology').all()
    assert (perfect['skill'].fillna(-1) <= 0).all()                  # a perfect extra baseline cannot be beaten


def test_experiment_outputs_are_separate_and_not_promoted():
    outs = {os.path.normpath(p) for p in (tv.LH_ARTIFACT, tv.LH_CV_CSV, tv.LH_FINAL_CSV, tv.LH_REPORT, tv.LH_SNAPSHOT)}
    assert not outs & {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS}
    assert all('experiments' in p for p in outs)
    art = os.path.join(ROOT, tv.LH_ARTIFACT)
    if os.path.exists(art):
        b = joblib.load(art)
        assert b['status'].startswith('EXPERIMENT')
        prod_t = joblib.load(os.path.join(ROOT, tv.TEMP_HOURLY_ARTIFACT))
        assert not prod_t['gate']['6h']['passed'] and not prod_t['gate']['24h']['passed']   # nothing promoted
