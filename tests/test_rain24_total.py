"""
24 h rainfall total experiment: target/feature alignment, no future leakage, chronological splits, zero-rain
behaviour, non-negative predictions, unique-grid-cell evaluation, training-only baselines and calibration,
CRPS approximation, and isolation from production. Synthetic data only; no project data is used.
"""

import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import long_horizon_history as lh
from src.models import rain24_total_experiment as rx
from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _frame(n_hours=24 * 400, start='2019-01-01', seed=0, rain=None):
    n_hours = len(rain) if rain is not None else n_hours
    idx = pd.date_range(start, periods=n_hours, freq='h', tz='UTC')
    rng = np.random.default_rng(seed)
    p = rain if rain is not None else np.where(rng.random(n_hours) < 0.12, rng.gamma(0.7, 2.0, n_hours), 0.0)
    return pd.DataFrame({'precipitation_mm': p, 'temperature_2m_c': 28 + rng.normal(0, 1, n_hours),
                         'relative_humidity_2m_pct': 70 + rng.normal(0, 5, n_hours),
                         'wind_speed_10m_kmh': 10 + rng.random(n_hours), 'wind_direction_10m_deg': rng.random(n_hours) * 360},
                        index=idx)


SPLITS = {'train': ('2019-01-01 00:00', '2019-10-31 23:00'), 'validate': ('2019-11-01 00:00', '2020-02-03 23:00')}


def _df(seed=0, cells=('gc1', 'gc2')):
    frames = {c: _frame(seed=seed + i) for i, c in enumerate(cells)}
    return rx.make_split(rx.build_base(frames), SPLITS), frames


def test_target_is_next_24h_total_and_persistence_is_past_24h():
    f = _frame(rain=np.arange(24 * 30, dtype=float))
    out = rx.cell_features(f)
    t = 200
    p = f['precipitation_mm'].to_numpy()
    assert out['y'].iloc[t] == p[t + 1:t + 25].sum()
    assert out['persistence'].iloc[t] == p[t - 23:t + 1].sum()
    assert out['y_h1'].iloc[t] == p[t + 1] and out['y_h24'].iloc[t] == p[t + 24]
    assert np.isnan(out['y'].iloc[-24]) and not np.isnan(out['y'].iloc[-25])


def test_no_future_rainfall_or_weather_enters_features():
    f = _frame()
    a = rx.cell_features(f)
    g = f.copy()
    t0 = 24 * 200
    g.iloc[t0 + 1:] = g.iloc[t0 + 1:] * 3 + 5                         # change everything AFTER the origin
    b = rx.cell_features(g)
    feats = rx.PROD_FEATURES + rx.EXT_RAW + ['persistence', 'season', 'origin_month']
    pd.testing.assert_frame_equal(a.iloc[:t0 + 1][feats], b.iloc[:t0 + 1][feats])
    assert a['y'].iloc[t0] != b['y'].iloc[t0]                           # only the target sees the future


def test_hours_since_wet_does_not_bridge_missing_data():
    rain = np.zeros(24 * 20)
    rain[10] = 1.0
    f = _frame(n_hours=len(rain), rain=rain)
    f.iloc[100:150] = np.nan                                            # gap
    out = rx.cell_features(f)
    v = out['log_hours_since_wet'].to_numpy()
    assert np.isfinite(v[99]) and np.isnan(v[200])                      # unknown after the gap until the next wet hour


def test_chronological_split_keeps_whole_windows_inside_each_period():
    df, _ = _df()
    end = df.index + pd.Timedelta(hours=24)
    for sp, (lo, hi) in SPLITS.items():
        m = (df['split'] == sp).to_numpy()
        assert (df.index[m] >= pd.Timestamp(lo, tz='UTC')).all() and (end[m] <= pd.Timestamp(hi, tz='UTC')).all()
    tr_end = df.index[df['split'] == 'train'].max() + pd.Timedelta(hours=24)
    assert tr_end < df.index[df['split'] == 'validate'].min() + pd.Timedelta(hours=24)


def test_one_series_and_one_group_per_unique_grid_cell():
    df, _ = _df(cells=('gc1', 'gc2', 'gc3'))
    assert sorted(df['cell'].unique()) == ['gc1', 'gc2', 'gc3']
    assert not df.reset_index().duplicated(['time_utc', 'cell']).any()
    g = rx.groups(df[df['split'] == 'validate'])
    assert sorted(k for k in g if k.startswith('cell=')) == ['cell=gc1', 'cell=gc2', 'cell=gc3']


def test_baselines_and_climatology_features_use_training_rows_only():
    df, _ = _df()
    b1 = rx.baselines(df, 'validate')
    df2 = df.copy()
    df2.loc[df2['split'] == 'validate', 'y'] = 999.0                   # validation targets must not matter
    b2 = rx.baselines(df2, 'validate')
    for name in ('climatology', 'conditional_climatology'):
        np.testing.assert_allclose(b1[name]['Q'], b2[name]['Q'])
        np.testing.assert_allclose(b1[name]['p25'], b2[name]['p25'])
    base = rx.build_base({'gc1': _frame(seed=1)})
    base2 = base.copy()
    v = rx.make_split(base, SPLITS)
    base2.loc[v.index[v['split'] == 'validate'], 'y'] = 999.0
    v2 = rx.make_split(base2, SPLITS)
    np.testing.assert_allclose(v[rx.CLIM_FEATS].to_numpy(), v2[rx.CLIM_FEATS].to_numpy())


@pytest.mark.parametrize('name', rx.CANDIDATES)
def test_predictions_are_non_negative_and_monotone(name):
    df, _ = _df()
    fs = rx.feature_sets(df)
    tr, e = df[df['split'] == 'train'], df[df['split'] == 'validate']
    pr = rx.make_candidate(name, fs).fit(tr).predict(e)
    assert np.isfinite(pr['Q']).all() and (pr['Q'] >= 0).all()
    assert (np.diff(pr['Q'], axis=1) >= -1e-9).all()                    # quantiles non-decreasing
    assert (pr['mean'] >= 0).all() and (pr['median'] >= 0).all()
    assert ((pr['p25'] >= 0) & (pr['p25'] <= 1)).all()


@pytest.mark.parametrize('name', ['A0', 'B1', 'C1', 'C2_cal'])
def test_all_dry_training_gives_zero_forecasts(name):
    frames = {'gc1': _frame(rain=np.zeros(24 * 400))}
    df = rx.make_split(rx.build_base(frames), SPLITS)
    fs = rx.feature_sets(df)
    pr = rx.make_candidate(name, fs).fit(df[df['split'] == 'train']).predict(df[df['split'] == 'validate'])
    assert np.allclose(pr['Q'], 0) and np.allclose(pr['mean'], 0) and np.allclose(pr['p25'], 0, atol=1e-6)


def test_calibration_factor_is_fitted_on_training_years_only():
    frames = {'gc1': _frame(n_hours=24 * 900, start='2017-01-01')}
    base = rx.build_base(frames)
    splits = {'train': ('2017-01-01 00:00', '2018-12-31 23:00'), 'validate': ('2019-01-01 00:00', '2019-06-30 23:00')}
    df = rx.make_split(base, splits)
    fs = rx.feature_sets(df)
    m1 = rx.make_candidate('C2_cal', fs).fit(df[df['split'] == 'train'])
    df2 = df.copy()
    df2.loc[df2['split'] == 'validate', 'y'] *= 10                    # validation amounts must not matter
    m2 = rx.make_candidate('C2_cal', fs).fit(df2[df2['split'] == 'train'])
    assert m1.factor == m2.factor
    assert all(rx.CAL_CLIP[0] <= v <= rx.CAL_CLIP[1] for v in m1.factor.values())


def test_crps_quantile_approximation():
    y = np.array([0.0, 1.0, 5.0])
    pt = np.repeat(np.array([[2.0], [2.0], [2.0]]), len(rx.TAUS), axis=1)
    np.testing.assert_allclose(rx.crps_from_quantiles(pt, y), np.abs(2.0 - y))   # point forecast -> |q - y|
    from src.models.rain_model import crps_zero_inflated_gamma
    p, a, s = 0.6, 0.8, 3.0
    u = np.clip((rx.TAUS - 0.4) / 0.6, 0, 1)
    Q = np.where(rx.TAUS <= 0.4, 0.0, 0.1 + stats.gamma.ppf(u, a, scale=s))[None, :].repeat(3, axis=0)
    exact = crps_zero_inflated_gamma(y, np.full(3, p), np.full(3, a), np.full(3, s), 0.1)
    np.testing.assert_allclose(rx.crps_from_quantiles(Q, y), exact, rtol=0.05, atol=0.03)


def test_selection_rule():
    folds = ['F2019', 'F2020', 'F2021', 'F2022']
    rows = []
    for f, a0, b, c in zip(folds, [-0.05, 0.0, -0.1, -0.02], [0.1, 0.05, 0.08, -0.01], [0.3, -0.2, -0.1, 0.4]):
        rows += [{'fold': f, 'candidate': 'A0', 'crpss': a0, 'bss': 0.0, 'mae_median': 3.0},
                 {'fold': f, 'candidate': 'B1', 'crpss': b, 'bss': 0.01, 'mae_median': 2.9},
                 {'fold': f, 'candidate': 'C1', 'crpss': c, 'bss': 0.05, 'mae_median': 2.5}]
    s = rx.select(pd.DataFrame(rows), {'A0': 5, 'B1': 5, 'C1': 5})
    assert s['selected'] == 'B1' and not s['candidates']['C1']['eligible']       # C1: higher mean but 2/4 positive
    rows2 = [dict(r, crpss=-abs(r['crpss']) - 0.01) if r['candidate'] != 'A0' else r for r in rows]
    s2 = rx.select(pd.DataFrame(rows2), {'A0': 5, 'B1': 5, 'C1': 5})
    assert s2['selected'] is None and s2['best_mean_if_none'] in ('B1', 'C1')


def test_validation_gate_requires_every_condition():
    def rows(**over):
        base = {'crpss': 0.1, 'crpss_ci_low': 0.02, 'crps_improvement_vs_A0': 0.1, 'crps_improvement_vs_A0_ci_low': 0.01,
                'bss': 0.05, 'mse_skill': 0.05, 'ratio_pred_obs': 0.95, 'event_days': 150}
        base.update(over)
        r = [{'candidate': 'X', 'group': 'ALL', **base}]
        r += [{'candidate': 'X', 'group': f'season={s}', 'crpss': 0.05} for s in rx.SEASONS]
        r += [{'candidate': 'X', 'group': f'cell=gc{i}', 'crpss': 0.05} for i in range(1, 9)]
        return pd.DataFrame(r)
    assert rx.validation_gate(rows(), 'X')['passed']
    for bad in ({'crpss_ci_low': -0.01}, {'crps_improvement_vs_A0_ci_low': -0.001}, {'bss': -0.01},
                {'ratio_pred_obs': 0.79}, {'mse_skill': -0.01}, {'event_days': 99}):
        assert not rx.validation_gate(rows(**bad), 'X')['passed'], bad
    r = rows()
    r.loc[r['group'] == 'season=sw_monsoon', 'crpss'] = -0.001
    assert not rx.validation_gate(r, 'X')['passed']


def test_outputs_isolated_and_criteria_predeclared():
    prod = {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS} | {os.path.normpath(p) for p in lh.OUTPUTS}
    assert not {os.path.normpath(p) for p in rx.OUTPUTS} & prod
    assert all('rain24_total' in p and 'experiments' in p for p in rx.OUTPUTS)
    crit = ' '.join(open(os.path.join(ROOT, rx.CRITERIA), encoding='utf-8').read().split())
    for phrase in ('at least 3 of 4 folds', 'within [0.8, 1.25]', '> 0 in >= 6 of 8 cells', 'At least 100 distinct days'):
        assert phrase in crit
    src = inspect.getsource(rx)
    assert 'include_sealed=True' not in src and 'load_history(include_sealed=False)' in src
    from src.forecasting import live_forecaster
    from src.live import live_pipeline
    for mod in (live_pipeline, live_forecaster):
        assert 'rain24_total' not in inspect.getsource(mod)
