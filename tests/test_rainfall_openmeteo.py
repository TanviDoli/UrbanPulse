"""
Open-Meteo-trained rainfall forecasting: chronological splits, grid-cell mapping, baselines,
evaluation gate, live integration (hourly rate vs 24 h total, failed horizons unavailable) and
pipeline output. Models here are hand-built TemporalHMMs (parameters set directly, nothing fitted);
the local Open-Meteo file is only read.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alerts.alert_engine import compute_24h_rainfall_accumulation
from src.collectors import live_collect, open_meteo
from src.forecasting.live_forecaster import LiveForecaster, RAIN_FAILED_EVAL
from src.models import train_and_validate_hmm as tv
from src.models.hmm_model import TemporalHMM
from dashboard import data_loader as dl

ZONE_TO_CELL = {1: 'c1', 2: 'c2', 3: 'c2', 4: 'c3', 5: 'c3', 6: 'c3', 8: 'c3', 7: 'c4', 9: 'c5', 13: 'c5',
                10: 'c6', 11: 'c6', 12: 'c7', 14: 'c7', 15: 'c8'}


def _hmm(scale=1.0):
    m = TemporalHMM(n_components=3, variable='rainfall')
    m.means_ = np.array([0.0, 1.5, 8.0]) * scale
    m.variances_ = np.array([1e-3, 1.0, 16.0])
    m.transmat_ = np.array([[0.95, 0.04, 0.01], [0.30, 0.60, 0.10], [0.20, 0.30, 0.50]])
    m.startprob_ = np.array([0.80, 0.15, 0.05])
    m.is_fitted = True
    return m


def _bundle(passed):
    cells = sorted(set(ZONE_TO_CELL.values()))
    keys = ['1h', '3h', '6h', '24h', '24h_total']
    return {'models': {c: _hmm(1.0 + i / 10) for i, c in enumerate(cells)}, 'zone_to_cell': dict(ZONE_TO_CELL),
            'gate': {k: {'passed': k in passed, 'test_skill': 0.1 if k in passed else -0.2} for k in keys},
            'meta': {}}


def _forecaster(passed):
    fc = LiveForecaster()
    fc.rain_om = _bundle(passed)
    fc.rain_twopart = None                        # exercise the Open-Meteo HMM path explicitly
    return fc


def _precip(origin, wet=False):
    idx = pd.date_range(origin - pd.Timedelta(hours=71), origin, freq='h')
    vals = np.where(np.arange(len(idx)) > 60, 6.0, 0.0) if wet else np.zeros(len(idx))
    return {z: pd.Series(vals, index=idx) for z in range(1, 16)}


# ── data and splits ──────────────────────────────────────────────────────────
def test_splits_are_chronological_contiguous_and_inside_local_data():
    s = {k: tuple(pd.Timestamp(t, tz='UTC') for t in v) for k, v in tv.RAIN_OM_SPLITS.items()}
    assert s['train'][0] < s['train'][1] < s['validate'][0] < s['validate'][1] < s['test'][0] < s['test'][1]
    assert s['validate'][0] - s['train'][1] == pd.Timedelta(hours=1)
    assert s['test'][0] - s['validate'][1] == pd.Timedelta(hours=1)
    h = open_meteo.load_local()['hourly']
    assert h['time_utc'].min() == s['train'][0] and h['time_utc'].max() == s['test'][1]


def test_local_rain_series_map_15_zones_to_8_grid_cells():
    cells, z2c = tv.openmeteo_rain_cells(open_meteo.load_local()['hourly'])
    assert sorted(z2c) == list(range(1, 16)) and len(cells) == 8
    groups = sorted(sorted(z for z, c in z2c.items() if c == cell) for cell in cells)
    assert groups == sorted(sorted(z for z, c in ZONE_TO_CELL.items() if c == cell) for cell in set(ZONE_TO_CELL.values()))
    for s in cells.values():
        assert s.index.freq == 'h' and not s.isna().any() and (s >= 0).all()


# ── model helpers, baselines, evaluation, gate ───────────────────────────────
def test_vectorised_24h_total_matches_alert_engine_formula():
    m = _hmm()
    seq = np.array([0, 0, 0.5, 3.0, 8.0, 2.0, np.nan, 0.0, 0.2])
    mean, var = m.forecast_accumulation_sequence(seq, n_steps=24)
    gammas = m.filter_sequence(seq)
    for t in (0, 4, 6, 8):
        mu_ref, var_ref = compute_24h_rainfall_accumulation(gammas[t], m.transmat_, m.means_, m.variances_)
        assert mean[t] == pytest.approx(mu_ref) and var[t] == pytest.approx(var_ref)


def _series(n_days=5):
    start = pd.Timestamp(tv.RAIN_OM_SPLITS['validate'][0], tz='UTC') - pd.Timedelta(days=2)
    idx = pd.date_range(start, periods=24 * n_days, freq='h')
    return pd.Series(np.arange(len(idx)) % 5 * 1.0, index=idx)


def test_baselines_and_targets_are_causal_and_use_training_months_only():
    s = _series()
    tr = tv._split_mask(s.index, 'train')
    s_val_huge = s.copy()
    s_val_huge[~tr] = 1000.0                                     # validation values must not leak into climatology
    res = tv.rainfall_forecasts_and_baselines(_hmm(), s_val_huge, tr)
    clim_train = s_val_huge[tr].mean()
    assert np.nanmax(res['1h']['climatology']) == pytest.approx(clim_train)
    y = s_val_huge.to_numpy()
    assert res['3h']['target'][10] == y[13] and res['3h']['persistence'][10] == y[10]
    tot = res['24h_total']
    assert tot['target'][30] == pytest.approx(y[31:55].sum())
    assert tot['persistence'][30] == pytest.approx(y[7:31].sum())
    assert np.isnan(tot['target'][-1]) and np.isnan(tot['persistence'][0])


def test_evaluation_counts_only_origins_and_targets_inside_each_split():
    s = _series()
    m = tv.evaluate_rainfall_openmeteo({'c1': _hmm()}, {'c1': s})
    for split in ('train', 'validate'):
        n_split = int(tv._split_mask(s.index, split).sum())
        r = m[(m.cell == 'ALL') & (m.split == split)].set_index(['horizon', 'method'])
        assert r.loc[('24h_total', 'model'), 'count'] <= n_split - 24
        assert r.loc[('1h', 'model'), 'count'] <= n_split - 1
    assert set(m['method']) == {'model', 'persistence', 'climatology', 'zero'}
    assert set(m['horizon']) == {'1h', '3h', '6h', '24h', '24h_total'}


def _metrics(model_rmse, base_rmse):
    rows = []
    for split in ('validate', 'test'):
        for key in ('1h', '3h', '6h', '24h', '24h_total'):
            rows.append({'cell': 'ALL', 'split': split, 'horizon': key, 'method': 'model',
                         'mae': 0.1, 'rmse': model_rmse[split], 'count': 100})
            for meth, r in base_rmse.items():
                rows.append({'cell': 'ALL', 'split': split, 'horizon': key, 'method': meth,
                             'mae': 0.05, 'rmse': r, 'count': 100})
    return pd.DataFrame(rows)


def test_gate_requires_beating_the_best_baseline_on_validation_and_test():
    base = {'persistence': 1.0, 'climatology': 0.8, 'zero': 0.9}
    ok = tv.rainfall_horizon_gate(_metrics({'validate': 0.7, 'test': 0.75}, base))
    assert all(g['passed'] for g in ok.values())
    assert ok['1h']['validate_best_baseline'] == 'climatology' and ok['1h']['test_skill'] > 0
    bad_test = tv.rainfall_horizon_gate(_metrics({'validate': 0.7, 'test': 0.85}, base))
    assert not any(g['passed'] for g in bad_test.values())                # beats persistence only
    bad_val = tv.rainfall_horizon_gate(_metrics({'validate': 0.9, 'test': 0.7}, base))
    assert not any(g['passed'] for g in bad_val.values())


# ── live integration ─────────────────────────────────────────────────────────
def test_only_passing_horizons_are_published_hourly_vs_24h_total():
    origin = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    out = _forecaster({'1h', '24h_total'}).run_rainfall(_precip(origin, wet=True), origin)
    assert out['supported_horizons'] == ['1h', '24h'] and out['enabled'] is True and out['n_active'] == 15
    h = out['horizons']
    assert all(len(rows) == 15 for rows in h.values())
    for r in h['1h']:
        assert r['availability'] == 'MODEL_FORECAST' and r['mrf_pred'] is not None and r['accum_24h_mm'] is None
        assert r['quantity'] == 'hourly_rate_mm_per_h' and r['alert_status'] == 'NOT_APPLICABLE_HOURLY'
        assert r['target_time_utc'] == '2026-10-05T01:00:00+00:00'
    for key in ('3h', '6h'):
        for r in h[key]:
            assert r['availability'] == 'UNAVAILABLE' and r['alert_status'] == RAIN_FAILED_EVAL
            assert r['mrf_pred'] is None and r['exceedance_prob'] is None
            assert dl.get_risk_level(r['alert_status']) == 'Unavailable'
    for r in h['24h']:
        assert r['accum_24h_mm'] is not None and r['accum_quantity'] == 'total_next_24h_mm'
        assert r['mrf_pred'] is None                                     # +24 h hourly rate failed its gate
        assert r['alert_status'] in ('NORMAL', 'WATCH', 'WARNING') and 0 <= r['exceedance_prob'] <= 1
    by = {r['zone_id']: r for r in h['24h']}
    assert by[4]['accum_24h_mm'] == by[8]['accum_24h_mm'] and by[2]['accum_24h_mm'] == by[3]['accum_24h_mm']


def test_all_failed_horizons_give_no_rainfall_values():
    origin = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    out = _forecaster(set()).run_rainfall(_precip(origin), origin)
    assert out['supported_horizons'] == [] and out['enabled'] is False and out['n_active'] == 0
    for rows in out['horizons'].values():
        for r in rows:
            assert r['mrf_pred'] is None and r['accum_24h_mm'] is None and r['alert_status'] == RAIN_FAILED_EVAL


def test_missing_input_zone_is_unavailable_not_zero():
    origin = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    precip = _precip(origin)
    precip[7] = precip[7] * np.nan
    out = _forecaster({'1h', '24h_total'}).run_rainfall(precip, origin)
    r = {x['zone_id']: x for x in out['horizons']['1h']}[7]
    assert r['availability'] == 'UNAVAILABLE' and r['mrf_pred'] is None and r['alert_status'] == 'UNAVAILABLE_NO_DATA'


def _pipeline_output(tmp_path, monkeypatch, now):
    from src.live import live_pipeline
    t = pd.date_range(pd.Timestamp('2026-09-01', tz='UTC'), now.floor('h'), freq='h')
    pd.DataFrame([{'zone_id': z, 'time_utc': x, 'temperature_2m_c': 29.0, 'precipitation_mm': 0.4,
                   'relative_humidity_2m_pct': 70, 'wind_speed_10m_kmh': 10, 'wind_direction_10m_deg': 90,
                   'grid_latitude': 13.0, 'grid_longitude': 80.2, 'source': 'test'}
                  for z in range(1, 16) for x in t]).to_csv(tmp_path / 'om.csv', index=False)
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(tmp_path / 'none.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(tmp_path / 'om.csv'))
    monkeypatch.setattr(live_collect, 'load_status', lambda: {'open_meteo': {'ok': True, 'latest_record_utc': now.isoformat()}})
    out_path = tmp_path / 'pred.json'
    live_pipeline.run_once(fetch=False, now_utc=now, output_path=str(out_path), verbose=False,
                           forecaster=_forecaster({'1h', '24h_total'}))
    return out_path


def test_dashboard_shows_hourly_rate_vs_24h_total_and_failed_horizon(tmp_path, monkeypatch):
    apptest = pytest.importorskip('streamlit.testing.v1')
    import streamlit as st
    now = pd.Timestamp.now(tz='UTC').floor('h') + pd.Timedelta(minutes=5)      # current input at render time
    out_path = _pipeline_output(tmp_path, monkeypatch, now)
    monkeypatch.setattr(dl, 'PREDICTIONS_PATH', str(out_path))
    st.cache_data.clear()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    monkeypatch.chdir(root)
    at = apptest.AppTest.from_file(os.path.join(root, 'dashboard', 'app.py'), default_timeout=120)
    at.run()
    expected = {'1 Hour': 'expected rate in hour ending', '24 Hours': 'expected total over next 24 h',
                '3 Hours': 'did not beat a simple baseline'}
    for hz, text in expected.items():
        at.selectbox(key='horizon_dropdown').select(hz).run()
        assert not at.exception, [e.message for e in at.exception]
        assert text in ' '.join(m.value for m in at.markdown), hz
    st.cache_data.clear()


def test_pipeline_output_lists_only_evaluated_rainfall_horizons(tmp_path, monkeypatch):
    now = pd.Timestamp('2026-10-05 00:30', tz='UTC')
    text = _pipeline_output(tmp_path, monkeypatch, now).read_text(encoding='utf-8')
    assert 'NaN' not in text
    d = json.loads(text)
    m = d['meta']
    assert m['supported_horizons']['rainfall'] == ['1h', '24h'] and m['rainfall_forecast_enabled'] is True
    assert any('Open-Meteo reanalysis' in lim and 'enabled: 1h, 24h' in lim for lim in m['limitations'])
    assert all(z['rainfall_forecast_enabled'] and z['has_rainfall_model'] for z in d['zone_meta'])
    v = dl.forecast_view(d, 5, 'rainfall', '24h', now)
    assert v['forecast_freshness'] == 'CURRENT' and v['accum_24h_mm'] is not None and v['evaluated']
    assert dl.forecast_view(d, 5, 'rainfall', '3h', now)['level'] == 'Unavailable'


# ═════════════════════════════════════════════════════════════════════════════
# Two-part rainfall experiment (src/models/rain_model.py). No model is fitted here:
# TwoPartRainModel parameters are set by hand; data is synthetic unless stated.
# ═════════════════════════════════════════════════════════════════════════════
import inspect
import itertools

from scipy import integrate, stats

from src.models import rain_model as rm


def _twopart_model(key, feats, wet_coef=3.0, intercept=-2.0, gamma=(0.5, 2.0)):
    m = rm.TwoPartRainModel(key, feature_names=feats)
    m.mean_, m.scale_ = np.zeros(len(feats)), np.ones(len(feats))
    m.coef_ = np.zeros(len(feats))
    m.coef_[feats.index('wet_now')] = wet_coef
    m.intercept_ = intercept
    m.gamma_overall = gamma
    m.gamma_by_season = {'ne_monsoon': (0.8, 3.0)}
    return m


def _boundary_cells(days_before=4, days_after=4, n_cells=2):
    """Synthetic hourly series straddling the train/validate boundary; 3-hour bursts on every
    third day only, so both wet and fully dry 24 h windows occur."""
    start = pd.Timestamp(tv.RAIN_OM_SPLITS['validate'][0], tz='UTC') - pd.Timedelta(days=days_before)
    idx = pd.date_range(start, periods=24 * (days_before + days_after), freq='h')
    k = np.arange(len(idx))
    rain = np.where((k % 11 < 3) & ((k // 24) % 3 == 0), 2.0, 0.0)
    return {f'cell_{i + 1}': pd.Series(rain * (1 + i / 10), index=idx) for i in range(n_cells)}


def test_twopart_features_use_only_information_at_forecast_time():
    idx = pd.date_range('2025-06-01', periods=200, freq='h', tz='UTC')
    y = pd.Series(np.random.default_rng(0).gamma(0.3, 2.0, len(idx)) * (np.arange(len(idx)) % 5 == 0), index=idx)
    f1 = rm.build_features(y)
    t = 120
    y2 = y.copy()
    y2.iloc[t + 1:] = 99.0                                            # change the future only
    f2 = rm.build_features(y2)
    pd.testing.assert_frame_equal(f1.iloc[:t + 1], f2.iloc[:t + 1])
    assert np.isnan(f1['log_rain_24h'].iloc[22]) and np.isfinite(f1['log_rain_24h'].iloc[23])


def test_twopart_rows_respect_split_boundaries_and_separate_rate_from_total():
    cells = _boundary_cells()
    train_end = pd.Timestamp(tv.RAIN_OM_SPLITS['train'][1], tz='UTC')
    val_start = pd.Timestamp(tv.RAIN_OM_SPLITS['validate'][0], tz='UTC')
    for key, off in (('1h', 1), ('24h', 24), ('24h_total', 24)):
        df = tv.twopart_rows(cells, key)
        assert df[df['split'] == 'train'].index.max() == train_end - pd.Timedelta(hours=off)   # no target crossing
        assert df[df['split'] == 'validate'].index.min() == val_start
        assert set(df['split']) == {'train', 'validate'}
    s = cells['cell_1']
    rate, total = tv.twopart_rows(cells, '24h'), tv.twopart_rows(cells, '24h_total')
    t0 = rate[rate['cell'] == 'cell_1'].index[5]
    i = s.index.get_loc(t0)
    assert rate.loc[(rate['cell'] == 'cell_1') & (rate.index == t0), 'y'].item() == s.iloc[i + 24]   # rate at t+24
    assert total.loc[(total['cell'] == 'cell_1') & (total.index == t0), 'y'].item() == pytest.approx(
        s.iloc[i + 1:i + 25].sum())                                                                   # 24 h total
    assert tv._twopart_events('24h') == (0.1, 1.0) and tv._twopart_events('24h_total') == (2.5, 15.6)


def test_twopart_baselines_use_training_rows_only_with_fallbacks():
    df = tv.twopart_rows(_boundary_cells(), '1h')
    o = (df['y'].to_numpy() >= 0.1).astype(float)
    o_leak = o.copy()
    o_leak[(df['split'] == 'validate').to_numpy()] = 1.0                 # validation outcomes must not matter
    levels = [['target_month', 'target_hour'], ['target_hour'], []]
    a = tv._train_lookup(df, o, levels)
    np.testing.assert_allclose(a, tv._train_lookup(df, o_leak, levels))
    # A (December, hour) combination with no TRAINING rows falls back to the training hour-of-day mean.
    # (Some late-November UTC training targets are already 1 December in IST, so not every hour is missing.)
    c1 = (df['cell'] == 'cell_1').to_numpy()
    tr = (df['split'] == 'train').to_numpy() & c1
    m_, h = df['target_month'].to_numpy(), df['target_hour'].to_numpy()
    in_train = set(zip(m_[tr], h[tr]))
    dec_missing = [hh for hh in np.unique(h[(m_ == 12) & c1]) if (12, hh) not in in_train]
    assert dec_missing and np.isfinite(a[(m_ == 12)]).all()
    h0 = dec_missing[0]
    assert a[c1 & (m_ == 12) & (h == h0)][0] == pytest.approx(o[tr & (h == h0)].mean())


@pytest.mark.parametrize('y,p,a,s', [(0.0, 0.3, 0.7, 2.0), (0.05, 0.6, 1.5, 1.0), (0.1, 0.9, 0.4, 3.0),
                                     (4.2, 0.4, 0.8, 2.5), (30.0, 0.95, 2.0, 5.0), (2.0, 1.0, 1.2, 1.5),
                                     (3.0, 0.0, 0.7, 2.0)])
def test_crps_closed_form_matches_numerical_integration(y, p, a, s):
    loc = rm.WET_MM

    def cdf(x):
        return (1 - p) + p * stats.gamma.cdf(x - loc, a, scale=s)

    upper = max(y, loc) + stats.gamma.ppf(1 - 1e-12, a, scale=s) + 1.0
    pts = sorted({0.0, loc, y}) + [upper]
    num = sum(integrate.quad(lambda x: (cdf(x) - float(x >= y)) ** 2, lo, hi, limit=200)[0]
              for lo, hi in zip(pts[:-1], pts[1:]))
    assert rm.crps_zero_inflated_gamma(y, p, a, s) == pytest.approx(num, rel=1e-5, abs=1e-7)


def test_zero_rain_cases():
    y = np.array([0.0, 0.0, 3.0])
    assert np.allclose(rm.crps_zero_inflated_gamma(y, 0.0, np.nan, np.nan), y)    # certain-dry forecast = |y|
    assert rm.mixture_mean(0.0, 0.5, 2.0) == 0.0
    assert rm.mixture_median(np.array([0.3]), np.array([0.5]), np.array([2.0]))[0] == 0.0
    assert rm.prob_exceed(np.array([0.4]), np.array([0.5]), np.array([2.0]), rm.WET_MM)[0] == 0.4
    assert rm.gamma_moments(np.array([0.0, 0.0])) is None and rm.gamma_moments(np.array([])) is None
    idx = pd.date_range('2025-11-20', periods=24 * 20, freq='h', tz='UTC')
    df = tv.twopart_rows({'cell_1': pd.Series(0.0, index=idx)}, '1h')
    assert (df['y'] == 0).all() and np.isfinite(df[rm.FEATURES].to_numpy()).all()
    feats = rm.FEATURES + ['is_cell_1']
    scored = tv.twopart_scored_rows(df, _twopart_model('1h', feats).predict(df[feats], df['season'].to_numpy()), '1h')
    m = tv.evaluate_twopart(scored, '1h')
    pooled = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL')]
    assert (pooled['events_ge0.1'] == 0).all() and pooled['auc_ge0.1'].isna().all()
    assert pooled['bss_ge0.1'].isna().all()                          # perfect baselines -> skill undefined
    assert not any(g['probability_passed'] for g in tv.twopart_gate(m).values())    # undefined never passes


def test_brier_and_bss_scoring_known_values():
    g = pd.DataFrame({'day': ['d1', 'd1', 'd2', 'd2'], 'y': [0.0, 1.0, 0.0, 2.0], 'persistence': [0.0, 0.0, 1.0, 2.0],
                      'o_0.1': [0, 1, 0, 1.0], 'pm_0.1': [0.1, 0.8, 0.2, 0.9], 'pc_0.1': [0.5] * 4,
                      'pp_0.1': [0, 0, 1, 1.0], 'pk_0.1': [0.3, 0.3, 0.6, 0.6],
                      'o_1.0': [0, 1, 0, 1.0], 'pm_1.0': [0.1] * 4, 'pc_1.0': [0.5] * 4, 'pp_1.0': [0, 0, 1, 1.0],
                      'pk_1.0': [0.5] * 4, 'crps_model': [0.1, 0.2, 0.1, 0.3], 'crps_clim': [0.2] * 4,
                      'crps_zero': [0, 1, 0, 2.0], 'crps_persistence': [0, 1, 1, 0.0],
                      'model_mean': [0.1, 0.8, 0.2, 1.5], 'model_median': [0, 1, 0, 1.0]})
    s = tv._score_group(g, '1h', p_star=0.5)
    bm = np.mean((np.array([0.1, 0.8, 0.2, 0.9]) - [0, 1, 0, 1]) ** 2)            # 0.025
    assert s['brier_ge0.1_model'] == pytest.approx(bm)
    assert s['brier_ge0.1_climatology'] == pytest.approx(0.25)                    # lowest baseline Brier
    assert s['brier_ge0.1_conditional_climatology'] == pytest.approx(0.275)
    assert s['brier_ge0.1_persistence'] == pytest.approx(0.5)
    assert s['bss_ge0.1_best_baseline'] == 'climatology'
    assert s['bss_ge0.1'] == pytest.approx(1 - bm / 0.25)
    assert (s['pod'], s['far'], s['csi']) == (1.0, 0.0, 1.0)
    assert s['crpss_vs_clim'] == pytest.approx(1 - 0.175 / 0.2)
    assert s['mae_zero'] == pytest.approx(0.75) and s['n_days'] == 2


def _gate_metrics(vals):
    rows = []
    for key in tv.TWOPART_KEYS:
        thr0 = tv._twopart_events(key)[0]
        for split in ('validate', 'test'):
            bss, crpss = vals.get((key, split), (0.1, 0.1))
            rows.append({'horizon': key, 'split': split, 'cell': 'ALL', 'season': 'ALL', f'bss_ge{thr0}': bss,
                         f'bss_ge{thr0}_best_baseline': 'climatology', 'crpss_vs_clim': crpss})
    return pd.DataFrame(rows)


def test_twopart_gate_failed_horizons_and_unrounded_comparisons():
    g = tv.twopart_gate(_gate_metrics({
        ('1h', 'test'): (-0.001, 0.2),                     # probability fails on test -> amount fails too
        ('3h', 'validate'): (1e-9, 0.2),                   # tiny positive skill passes: no rounding
        ('6h', 'validate'): (0.0, 0.2),                    # exactly zero skill fails
        ('24h', 'test'): (0.2, -1e-9),                     # amount fails, probability passes
        ('24h_total', 'validate'): (np.nan, 0.2),          # missing score fails
    }))
    assert (g['1h']['probability_passed'], g['1h']['amount_passed']) == (False, False)
    assert (g['3h']['probability_passed'], g['3h']['amount_passed']) == (True, True)
    assert g['6h']['probability_passed'] is False
    assert (g['24h']['probability_passed'], g['24h']['amount_passed']) == (True, False)
    assert g['24h_total']['probability_passed'] is False and g['24h_total']['event_mm'] == 2.5


def test_day_block_ci_resamples_whole_days_across_cells():
    days = np.array(['d1', 'd1', 'd2', 'd2', 'd3', 'd3'])            # two cells per day
    ref = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0])
    assert tv.day_block_ci(days, ref, ref) == (0.0, 0.0)            # equal losses -> zero skill
    lo, hi = tv.day_block_ci(days, ref * 0.5, ref)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)    # uniform improvement
    model = np.array([0.0, 0.0, 2.0, 2.0, 1.0, 1.0])
    ci1 = tv.day_block_ci(days, model, ref)
    assert ci1 == tv.day_block_ci(days, model, ref)                 # deterministic (fixed seed)
    assert ci1[0] <= 1 - model.sum() / ref.sum() <= ci1[1]
    # every bootstrap replicate is a resample of WHOLE days (both cells of a day move together)
    possible = {1 - sum(model[days == d].sum() for d in c) / sum(ref[days == d].sum() for d in c)
                for c in itertools.combinations_with_replacement(['d1', 'd2', 'd3'], 3)}
    assert min(possible) <= ci1[0] and ci1[1] <= max(possible)


def test_hand_built_twopart_model_predictions():
    feats = list(rm.FEATURES)
    m = _twopart_model('1h', feats, wet_coef=4.0, intercept=-2.0)
    X = pd.DataFrame(np.zeros((2, len(feats))), columns=feats)
    X.loc[1, 'wet_now'] = 1.0
    out = m.predict(X, np.array(['ne_monsoon', 'winter']))
    assert out['p_wet'] == pytest.approx([1 / (1 + np.exp(2.0)), 1 / (1 + np.exp(-2.0))])
    assert (out['shape'][0], out['scale'][0]) == (0.8, 3.0)               # season-specific gamma
    assert (out['shape'][1], out['scale'][1]) == (0.5, 2.0)               # season without fit -> overall gamma
    assert out['median'][0] == 0.0 and out['median'][1] > rm.WET_MM       # dry-likely -> 0; wet-likely -> > 0.1
    assert out['mean'][1] == pytest.approx(out['p_wet'][1] * (rm.WET_MM + 0.5 * 2.0))


def test_twopart_evaluation_reports_split_cell_season_with_pooled_cis():
    cells = _boundary_cells()
    feats = rm.FEATURES + [f'is_{c}' for c in sorted(cells)]
    for key in ('1h', '24h_total'):
        df = tv.twopart_rows(cells, key)
        scored = tv.twopart_scored_rows(df, _twopart_model(key, feats).predict(df[feats], df['season'].to_numpy()), key)
        m = tv.evaluate_twopart(scored, key)
        assert set(m['split']) == {'train', 'validate'} and {'ALL', 'cell_1', 'cell_2'} <= set(m['cell'])
        assert {'ALL', 'ne_monsoon'} <= set(m['season'])
        pooled = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL')]
        assert pooled[['bss_ci_low', 'bss_ci_high', 'crpss_clim_ci_low', 'crpss_clim_ci_high']].notna().all().all()
        assert m.loc[(m['cell'] != 'ALL') | (m['season'] != 'ALL'), 'bss_ci_low'].isna().all()
        assert pooled['quantity'].iloc[0] == ('total_24h_mm' if key == '24h_total' else 'hourly_rate_mm_per_h')
        assert pooled.loc[pooled['split'] == 'validate', 'n'].item() == int((df['split'] == 'validate').sum())


def test_twopart_dry_run_cannot_write_and_heavy_rain_alerts_stay_disabled():
    assert tv.TWOPART_HEAVY_RAIN_ALERTS_ENABLED is False
    before_save, after_save = inspect.getsource(tv.train_rainfall_twopart).split('if save:')
    for w in ('joblib.dump', '.to_csv(', 'makedirs', '.save('):
        assert w not in before_save                                 # every write sits under `if save:`
    assert 'joblib.dump' in after_save and '.to_csv(' in after_save


# ── saved two-part artifact + live integration (1 h only) ────────────────────
TWOPART_PATH = os.path.join(os.path.dirname(__file__), '..', tv.TWOPART_ARTIFACT)
needs_twopart = pytest.mark.skipif(not os.path.exists(TWOPART_PATH), reason='two-part artifact not saved')


@needs_twopart
def test_saved_twopart_artifact_gate_and_heavy_rain_disabled():
    import joblib
    b = joblib.load(TWOPART_PATH)
    g = b['gate']
    assert g['1h']['probability_passed'] and g['1h']['amount_passed']
    assert g['3h']['probability_passed'] and g['3h']['amount_passed']       # promoted 3 h candidate
    assert not any(g[k]['probability_passed'] for k in ('6h', '24h', '24h_total'))
    assert b['meta']['heavy_rain_alerts_enabled'] is False and b['heavy_rain_events']['test']['distinct_days'] == 0
    assert sorted(b['zone_to_cell']) == list(range(1, 16)) and len(b['cells']) == 8


@needs_twopart
def test_live_twopart_publishes_only_1h_probability_and_amount():
    fc = LiveForecaster()
    assert fc.rain_twopart is not None and fc.rainfall_zones == list(range(1, 16)) and fc.rainfall_enabled
    origin = pd.Timestamp('2026-10-05 00:00', tz='UTC')
    precip = _precip(origin, wet=True)
    precip[7] = precip[7].copy()
    precip[7].iloc[-3] = np.nan                                     # incomplete last 24 h -> unavailable, not zero
    out = fc.run_rainfall(precip, origin)
    assert out['source'] == 'open_meteo_twopart' and out['supported_horizons'] == ['1h'] and out['enabled']
    h1 = {r['zone_id']: r for r in out['horizons']['1h']}
    for z, r in h1.items():
        if z == 7:
            assert r['availability'] == 'UNAVAILABLE' and r['prob_rain'] is None and r['mrf_pred'] is None
            continue
        assert r['availability'] == 'MODEL_FORECAST' and 0 <= r['prob_rain'] <= 1 and r['mrf_pred'] >= 0
        assert r['alert_status'] == 'NOT_APPLICABLE_HOURLY' and r['exceedance_prob'] is None
        assert r['target_time_utc'] == '2026-10-05T01:00:00+00:00' and r['accum_24h_mm'] is None
    assert h1[4]['prob_rain'] == h1[8]['prob_rain']                 # same grid cell -> identical values
    assert h1[1]['prob_rain'] > 0.5                                 # it is raining now -> rain likely next hour
    # promoted 3 h model needs the weather frames: without them it is unavailable, never filled
    assert all(r['alert_status'] == 'UNAVAILABLE_NO_DATA' and r['prob_rain'] is None for r in out['horizons']['3h'])
    for key in ('6h', '24h'):
        for r in out['horizons'][key]:
            assert r['alert_status'] == RAIN_FAILED_EVAL and r['prob_rain'] is None and r['mrf_pred'] is None
            assert r['accum_24h_mm'] is None and r['exceedance_prob'] is None
            assert r['heavy_rain_alert'] == 'NOT_EVALUATED_INSUFFICIENT_HEAVY_RAIN_EVENTS'


@needs_twopart
def test_pipeline_and_dashboard_with_twopart_rainfall(tmp_path, monkeypatch):
    apptest = pytest.importorskip('streamlit.testing.v1')
    import streamlit as st
    from src.live import live_pipeline
    now = pd.Timestamp.now(tz='UTC').floor('h') + pd.Timedelta(minutes=5)
    t = pd.date_range(now.floor('h') - pd.Timedelta(days=35), now.floor('h'), freq='h')
    pd.DataFrame([{'zone_id': z, 'time_utc': x, 'temperature_2m_c': 29.0, 'precipitation_mm': 0.4,
                   'relative_humidity_2m_pct': 70, 'wind_speed_10m_kmh': 10, 'wind_direction_10m_deg': 90,
                   'grid_latitude': 13.0, 'grid_longitude': 80.2, 'source': 'test'}
                  for z in range(1, 16) for x in t]).to_csv(tmp_path / 'om.csv', index=False)
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(tmp_path / 'none.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(tmp_path / 'om.csv'))
    monkeypatch.setattr(live_collect, 'load_status', lambda: {'open_meteo': {'ok': True, 'latest_record_utc': now.isoformat()}})
    out_path = tmp_path / 'pred.json'
    live_pipeline.run_once(fetch=False, now_utc=now, output_path=str(out_path), verbose=False)
    d = json.loads(out_path.read_text(encoding='utf-8'))
    assert d['meta']['supported_horizons']['rainfall'] == ['1h', '3h']        # 3 h promoted
    assert any('two-part model' in lim and 'NOT evaluated' in lim for lim in d['meta']['limitations'])
    assert not [a for a in d['alerts'] if a['variable'] == 'rainfall' and a['alert_status'] in ('WATCH', 'WARNING')]
    monkeypatch.setattr(dl, 'PREDICTIONS_PATH', str(out_path))
    st.cache_data.clear()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    monkeypatch.chdir(root)
    at = apptest.AppTest.from_file(os.path.join(root, 'dashboard', 'app.py'), default_timeout=120)
    at.run()
    at.selectbox(key='horizon_dropdown').select('1 Hour').run()
    assert not at.exception and 'P(rain ≥0.1 mm)' in ' '.join(m.value for m in at.markdown)
    at.selectbox(key='horizon_dropdown').select('3 Hours').run()
    assert 'P(rain ≥0.1 mm)' in ' '.join(m.value for m in at.markdown)
    at.selectbox(key='horizon_dropdown').select('6 Hours').run()
    assert 'did not beat a simple baseline' in ' '.join(m.value for m in at.markdown)
    st.cache_data.clear()
