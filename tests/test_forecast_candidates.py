"""
3 h / 6 h / 24 h forecast candidate experiment: leakage-safe encodings, forecast-time features,
backward compatibility of production models, candidate/production path separation and saved
decisions. No model is fitted on project data here.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import live_forecaster as lf
from src.models import temp_model as tm
from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _enc_frame():
    # two cells; train rows in months 1-2, validation rows in month 3 and a sparse month 12
    rows = []
    for cell in ('a', 'b'):
        for m, split, n in ((1, 'train', 30), (2, 'train', 30), (12, 'train', 2), (3, 'validate', 10), (12, 'validate', 5)):
            for i in range(n):
                rows.append({'cell': cell, 'split': split, 'month': m, 'hour': i % 3})
    return pd.DataFrame(rows)


def test_train_encoding_is_train_only_leave_one_out_and_falls_back_on_sparse_groups():
    df = _enc_frame()
    rng = np.random.default_rng(0)
    v = rng.normal(size=len(df))
    levels = [['month'], []]
    enc = tv.train_encoding(df, v, levels, min_count=20)
    v_leak = v.copy()
    v_leak[(df['split'] == 'validate').to_numpy()] += 100.0
    np.testing.assert_allclose(enc, tv.train_encoding(df, v_leak, levels, min_count=20))     # no validation leakage
    a = (df['cell'] == 'a').to_numpy()
    tr_m1 = a & (df['split'] == 'train').to_numpy() & (df['month'] == 1).to_numpy()
    i = np.flatnonzero(tr_m1)[0]
    assert enc[i] == pytest.approx((v[tr_m1].sum() - v[i]) / (tr_m1.sum() - 1))              # leave-one-out
    val_m1_like = a & (df['split'] == 'validate').to_numpy() & (df['month'] == 3).to_numpy()
    tr_a = a & (df['split'] == 'train').to_numpy()
    assert enc[np.flatnonzero(val_m1_like)[0]] == pytest.approx(v[tr_a].mean())             # unseen month -> overall
    sparse_val = a & (df['split'] == 'validate').to_numpy() & (df['month'] == 12).to_numpy()
    assert enc[np.flatnonzero(sparse_val)[0]] == pytest.approx(v[tr_a].mean())              # 2 train rows < 20 -> fallback
    pooled = tv.train_encoding(df, v, [[]], per_cell=False, loo=False)
    assert np.allclose(pooled, v[(df['split'] == 'train').to_numpy()].mean())


def test_temp_candidate_features_use_only_values_known_at_origin():
    idx = pd.date_range('2025-06-01', periods=24 * 6, freq='h', tz='UTC')
    k = np.arange(len(idx))
    frame = pd.DataFrame({'temperature_2m_c': 29 + 3 * np.sin(2 * np.pi * k / 24) + 0.01 * k,
                          'relative_humidity_2m_pct': 70 + 5 * np.cos(2 * np.pi * k / 24),
                          'wind_speed_10m_kmh': 10.0, 'wind_direction_10m_deg': 90.0}, index=idx)
    for h in tv.CAND_TEMP_HORIZONS:
        f = tm.temp_candidate_features(frame, h)
        t = 100
        changed = frame.copy()
        changed.iloc[t + 1:] = 999.0
        pd.testing.assert_frame_equal(f.iloc[:t + 1], tm.temp_candidate_features(changed, h).iloc[:t + 1])
        assert f['t_same_hour_2d'].iloc[t] == frame['temperature_2m_c'].iloc[t + h - 48]
        assert f['wind_u'].iloc[t] == pytest.approx(-10.0) and abs(f['wind_v'].iloc[t]) < 1e-9   # easterly wind


def test_production_temperature_models_keep_their_six_features():
    m = tm.HourlyTempModel(1)
    m.coef_ = np.arange(len(tm.TEMP_FEATURES), dtype=float)
    del m.feature_names                                       # like artifacts saved before the change
    X = pd.DataFrame(np.ones((2, len(tm.TEMP_FEATURES) + 3)), columns=tm.TEMP_FEATURES + ['x1', 'x2', 'x3'])
    assert m.features == tm.TEMP_FEATURES and np.allclose(m.predict(X), m.coef_.sum())
    prod = os.path.join(ROOT, tv.TEMP_HOURLY_ARTIFACT)
    if os.path.exists(prod):
        b = joblib.load(prod)
        promoted = set(b.get('encodings') or {})                 # 3 h after the controlled promotion
        for key, model in b['models'].items():
            if key in promoted:
                assert model.features == tm.TEMP_FEATURES + tm.TEMP_CANDIDATE_EXTRA
            else:
                assert model.features == tm.TEMP_FEATURES


def test_candidate_paths_are_separate_from_production_and_live_pipeline():
    cands = {os.path.normpath(p) for p in (tv.CAND_TEMP_ARTIFACT, tv.CAND_RAIN_ARTIFACT, tv.CAND_TEMP_METRICS,
                                            tv.CAND_RAIN_METRICS, tv.CAND_REPORT)}
    assert not cands & {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS}
    assert all(p.startswith(('models' + os.sep + 'candidates', 'results' + os.sep + 'experiments')) for p in cands)
    for live_path in (lf.RAIN_TWOPART_ARTIFACT, lf.TEMP_HOURLY_ARTIFACT, lf.RAIN_OM_ARTIFACT):
        assert 'candidates' not in live_path                 # the live pipeline never loads candidates


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, tv.CAND_RAIN_ARTIFACT)), reason='candidates not saved')
def test_saved_candidate_decisions_follow_the_promotion_rule():
    for path, passed_key, prod_key, beats_key in (
            (tv.CAND_TEMP_ARTIFACT, 'candidate_passed', 'production_passed', 'beats_production_on_both_splits'),
            (tv.CAND_RAIN_ARTIFACT, 'candidate_probability_passed', 'production_probability_passed',
             'beats_production_bss_on_both_splits')):
        b = joblib.load(os.path.join(ROOT, path))
        assert b['meta']['status'].startswith('CANDIDATE')
        for key, d in b['decision'].items():
            assert d['promotion_eligible'] == bool(d[passed_key] and (d[beats_key] or not d[prod_key]))
    rain = joblib.load(os.path.join(ROOT, tv.CAND_RAIN_ARTIFACT))
    assert rain['decision']['24h_total']['promotion_eligible'] is False      # 24 h total still fails
