"""
Long-horizon experiment on the 2017-2023 history: fold/validation periods never reach 2024, unique grid
cells (shared cells not double counted), the 2024 guards, the pre-declared selection and pass rules, the
fixed decision threshold, and separation from production artifacts. Synthetic data only; no model is
fitted on project data here.
"""

import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import long_horizon_history as lh
from src.models import train_and_validate_hmm as tv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _ts(s):
    return pd.Timestamp(s, tz='UTC')


def test_periods_are_expanding_from_2017_and_never_reach_2024():
    assert list(lh.FOLDS) == ['F2019', 'F2020', 'F2021', 'F2022']
    for name, f in lh.FOLDS.items():
        year = int(name[1:])
        assert f['train'][0].startswith('2017-01-01') and _ts(f['train'][1]) < _ts(f['validate'][0])
        assert _ts(f['validate'][0]).year == year and _ts(f['validate'][1]).year == year
    assert lh.VALIDATION['train'][1].startswith('2022-12-31') and lh.VALIDATION['validate'][0].startswith('2023-01-01')
    for p in list(lh.FOLDS.values()) + [lh.VALIDATION]:
        assert all(_ts(hi).year <= 2023 for _, hi in p.values())
    assert lh.ROBUST['train'][1].startswith('2023-12-31') and lh.ROBUST['robust'][0].startswith('2025-01-01')
    assert not any(str(v).startswith('2024') for p in lh.ROBUST.values() for v in p)


def _fake_dev():
    t = pd.date_range('2017-01-01', periods=48, freq='h', tz='UTC')
    spec = [('zone_01', 1, 13.181, 80.237), ('zone_01_coastal_original', 1, 13.251, 80.310), ('zone_02', 2, 13.181, 80.237),
            ('zone_03', 3, 13.181, 80.237), ('zone_04', 4, 13.11, 80.246), ('zone_05', 5, 13.11, 80.246)]
    rows = []
    for k, z, la, lo in spec:
        rows.append(pd.DataFrame({'time_utc': t, 'location_key': k, 'zone_id': z, 'grid_latitude': la, 'grid_longitude': lo,
                                  **{v: float(z) for v in lh.VARS}}))
    return pd.concat(rows, ignore_index=True)


def test_grid_cells_are_unique_and_zone1_current_is_never_a_representative():
    cells = lh.grid_cells(_fake_dev())
    assert len(cells) == 3                                             # 6 locations, 3 grid cells
    by = cells.set_index('rep_location_key')
    assert 'zone_01' not in by.index and 'zone_01_coastal_original' in by.index
    assert by.loc['zone_01_coastal_original', 'local_file_zone'] == 1
    assert by.loc['zone_02', 'locations'] == 'zone_01,zone_02,zone_03' and by.loc['zone_02', 'local_file_zone'] == 2
    assert by.loc['zone_04', 'locations'] == 'zone_04,zone_05'


def test_cell_frames_keep_2024_empty_and_refuse_2024_values():
    dev = _fake_dev()
    cells = lh.grid_cells(dev)
    local = pd.DataFrame({'time_utc': pd.date_range('2025-01-01', periods=24, freq='h', tz='UTC'), 'zone_id': 2,
                          **{v: 9.0 for v in lh.VARS}})
    local = pd.concat([local, local.assign(zone_id=1), local.assign(zone_id=4)], ignore_index=True)
    fr = lh.cell_frames(cells, dev, local)
    f = fr[cells.loc[cells['rep_location_key'] == 'zone_02', 'cell'].iloc[0]]
    assert f.loc[f.index.year == 2024].isna().all().all() and (f.index.year == 2024).sum() == 8784
    assert f.loc['2025-01-01'].notna().all().all()
    bad = {'x': f.copy()}
    bad['x'].loc[bad['x'].index.year == 2024, 'temperature_2m_c'] = 1.0
    with pytest.raises(RuntimeError, match='2024'):
        lh.assert_no_2024(bad)


def test_sealed_open_guard_blocks_only_the_sealed_file():
    from src.collectors.open_meteo_history import HISTORY_DEV_FILE, HISTORY_SEALED_FILE
    with pytest.raises(PermissionError):
        lh.sealed_open_guard('open', (HISTORY_SEALED_FILE, 'rb', 0))
    with pytest.raises(PermissionError):
        lh.sealed_open_guard('open', ('somewhere/open_meteo_chennai_hourly_2024_SEALED.csv.gz', 'r', 0))
    lh.sealed_open_guard('open', (HISTORY_DEV_FILE, 'rb', 0))         # allowed
    lh.sealed_open_guard('open', (3, 'rb', 0))                          # file descriptors ignored
    lh.sealed_open_guard('exec', (HISTORY_SEALED_FILE,))                 # other events ignored


def test_run_refuses_when_unseal_env_is_set(monkeypatch):
    from src.collectors.open_meteo_history import UNSEAL_ENV
    monkeypatch.setenv(UNSEAL_ENV, 'anything')
    with pytest.raises(RuntimeError, match='sealed'):
        lh.run_experiment(save=False, install_guard=False)


def test_module_never_unseals():
    src = inspect.getsource(lh)
    assert 'include_sealed=True' not in src and 'HISTORY_SEALED_FILE)' not in src.replace('abspath(HISTORY_SEALED_FILE)', '')
    assert 'load_history(include_sealed=False)' in src


def test_selection_rule_needs_three_positive_folds_and_positive_mean():
    cv = pd.DataFrame({'fold': ['F2019', 'F2020', 'F2021', 'F2022'] * 3,
                       'variant': ['a'] * 4 + ['b'] * 4 + ['c'] * 4,
                       'skill': [0.9, 0.9, -0.1, -0.1,            # highest mean, only 2 positive -> ineligible
                                 0.05, 0.05, 0.05, -0.01,         # 3 positive, mean 0.035
                                 0.02, 0.02, 0.02, 0.02]})        # 4 positive, mean 0.02
    sel, info = lh.select_variant(cv, {'a': 5, 'b': 9, 'c': 3})
    assert sel == 'b' and info['eligible'] and info['positive_folds'] == 3
    tie = pd.DataFrame({'fold': ['F2019', 'F2020', 'F2021', 'F2022'] * 2, 'variant': ['x'] * 4 + ['y'] * 4,
                        'skill': [0.1] * 8})
    assert lh.select_variant(tie, {'x': 10, 'y': 6})[0] == 'y'          # tie -> fewer features
    none = pd.DataFrame({'fold': ['F2019', 'F2020', 'F2021', 'F2022'], 'variant': ['z'] * 4, 'skill': [0.2, -0.1, -0.1, 0.1]})
    sel, info = lh.select_variant(none, {'z': 4})
    assert sel == 'z' and info['eligible'] is False


def test_validation_pass_rules():
    good = {'skill': 0.1, 'ci_week_low': 0.01, 'cells_positive': 6}
    assert lh.validation_pass('temperature', good, True)['passed']
    assert not lh.validation_pass('temperature', good, False)['passed']                       # ineligible
    assert not lh.validation_pass('temperature', {**good, 'ci_week_low': -0.001}, True)['passed']
    assert not lh.validation_pass('temperature', {**good, 'cells_positive': 5}, True)['passed']
    rain = {**good, 'events': 100, 'event_days': 30, 'crpss_clim': 0.05, 'crpss_clim_ci_week_low': 0.01}
    g = lh.validation_pass('rainfall', rain, True)
    assert g['passed'] and g['amount_passed']
    assert not lh.validation_pass('rainfall', {**rain, 'event_days': 29}, True)['passed']
    g2 = lh.validation_pass('rainfall', {**rain, 'crpss_clim_ci_week_low': -0.01}, True)
    assert g2['passed'] and not g2['amount_passed']
    assert not lh.validation_pass('rainfall', {**rain, 'ci_week_low': float('nan')}, True)['passed']


def test_week_blocks_group_seven_days():
    w = lh._weeks(np.array(['2017-01-02', '2017-01-08', '2017-01-09', '2023-12-31']))
    assert w[0] == w[1] and w[2] == w[0] + 1 and w[3] > w[2]


def test_evaluate_twopart_fixed_threshold_is_used_and_default_unchanged():
    rng = np.random.default_rng(0)
    n = 400
    o = (rng.random(n) < 0.3).astype(float)
    p = np.clip(o * 0.5 + rng.random(n) * 0.5, 0, 1)
    sc = pd.DataFrame({'cell': 'c1', 'split': 'validate', 'season': 'winter', 'day': np.repeat(np.arange(n // 10), 10).astype(str),
                       'y': o * 2.0, 'persistence': 0.0, 'o_0.1': o, 'pm_0.1': p, 'pc_0.1': 0.3, 'pp_0.1': 0.0, 'pk_0.1': 0.3,
                       'o_1.0': o, 'pm_1.0': p * 0.8, 'pc_1.0': 0.3, 'pp_1.0': 0.0, 'pk_1.0': 0.3,
                       'crps_model': 0.5, 'crps_clim': 0.6, 'crps_zero': 0.6, 'crps_persistence': 0.7,
                       'model_mean': 0.5, 'model_median': 0.0})
    default = tv.evaluate_twopart(sc, '6h')
    fixed = tv.evaluate_twopart(sc, '6h', p_star=0.85)
    assert default['prob_threshold'].iloc[0] == tv._choose_prob_threshold(p, o)
    assert (fixed['prob_threshold'] == 0.85).all()
    pd.testing.assert_series_equal(default['bss_ge0.1'], fixed['bss_ge0.1'])   # skill does not depend on it


def test_outputs_are_experiment_only_and_criteria_exist():
    prod = {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS}
    assert not {os.path.normpath(p) for p in lh.OUTPUTS} & prod
    assert all(os.path.normpath(p).split(os.sep)[1] == 'experiments' for p in lh.OUTPUTS)
    assert os.path.exists(os.path.join(ROOT, lh.CRITERIA))
    crit = ' '.join(open(os.path.join(ROOT, lh.CRITERIA), encoding='utf-8').read().split())
    assert 'at least 3 of the 4 folds' in crit and 'at least 6 of the 8 grid cells' in crit
    assert lh.MIN_POSITIVE_FOLDS == 3 and lh.MIN_POSITIVE_CELLS == 6 and (lh.MIN_EVENTS, lh.MIN_EVENT_DAYS) == (100, 30)


def test_experiment_is_not_imported_by_live_code():
    from src.forecasting import live_forecaster
    from src.live import live_pipeline
    for mod in (live_pipeline, live_forecaster):
        assert 'long_horizon_history' not in inspect.getsource(mod)
