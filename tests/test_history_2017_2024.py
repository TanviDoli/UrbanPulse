"""
Open-Meteo 2017-2024 history (isolated experiment data): request plan, validation, separate dev/sealed
files, the 2024 seal, both zone-1 representations, and per-year stability flags. No network: a fake
session generates archive-style responses.
"""

import inspect
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audit import history_stability as hs
from src.collectors import open_meteo_history as hist
from src.collectors.open_meteo import EXPECTED_UNITS, HOURLY_VARS


class FakeResp:
    def __init__(self, payload, status=200, headers=None):
        self._p, self.status_code, self.headers = payload, status, headers or {}

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f'HTTP {self.status_code}')

    def json(self):
        return self._p


class FakeSession:
    """Archive-like responses; optional per-call corruption to test validation."""

    def __init__(self, corrupt=None, shift_year=None):
        self.calls, self.corrupt, self.shift_year = [], corrupt or {}, shift_year

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        n = len(params['latitude'].split(','))
        t = pd.date_range(params['start_date'], pd.Timestamp(params['end_date']) + pd.Timedelta(hours=23), freq='h')
        k = np.arange(len(t))
        bump = np.where(t.year == self.shift_year, 2.0, 0.0) if self.shift_year else 0.0
        rng = np.random.default_rng(int(t[0].year))                   # realistic, deterministic year-to-year noise
        yr_off = rng.normal(0, 0.1, 1)[0]
        items = []
        for i, (la, lo) in enumerate(zip(params['latitude'].split(','), params['longitude'].split(','))):
            h = {'time': [x.strftime('%Y-%m-%dT%H:%M') for x in t],
                 'temperature_2m': list(28 + 3 * np.sin(2 * np.pi * k / 24) + bump + yr_off + i * 0.01
                                        + rng.normal(0, 0.5, len(t))),
                 'relative_humidity_2m': list(70 + rng.normal(0, 3, len(t))),
                 'precipitation': list(np.where(rng.random(len(t)) < 0.15, rng.gamma(0.8, 1.5, len(t)), 0.0)),
                 'wind_speed_10m': list(10 + 0 * k), 'wind_direction_10m': list(90 + 0 * k)}
            items.append({'latitude': float(la), 'longitude': float(lo), 'elevation': 5.0, 'utc_offset_seconds': 0,
                          'timezone': 'GMT', 'hourly': h, 'hourly_units': {'time': 'iso8601', **EXPECTED_UNITS}})
        c = self.corrupt.get(len(self.calls))
        if c == 'units':
            items[0]['hourly_units']['precipitation'] = 'inch'
        elif c == 'count':
            items = items[:-1]
        elif c == 'gap':
            items[0]['hourly']['time'] = items[0]['hourly']['time'][:-1]
        return FakeResp(items if n > 1 else items[0])


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(hist, 'HISTORY_DEV_FILE', str(tmp_path / 'dev.csv.gz'))
    monkeypatch.setattr(hist, 'HISTORY_SEALED_FILE', str(tmp_path / 'sealed.csv.gz'))
    monkeypatch.setattr(hist, 'HISTORY_MANIFEST', str(tmp_path / 'manifest.json'))
    monkeypatch.setattr(hist, 'SEALED_ACCESS_LOG', str(tmp_path / 'log' / 'access.jsonl'))
    monkeypatch.setattr(hist.time, 'sleep', lambda s: None)
    monkeypatch.delenv(hist.UNSEAL_ENV, raising=False)
    return tmp_path


def test_locations_keep_both_zone1_representations_separately():
    locs = hist.history_locations()
    assert len(locs) == 16 and [l['zone_id'] for l in locs].count(1) == 2
    cur = next(l for l in locs if l['location_key'] == 'zone_01')
    ref = next(l for l in locs if l['location_key'] == 'zone_01_coastal_original')
    assert not cur['is_reference'] and cur['cell_selection'] is None              # exactly as the live collector
    assert ref['is_reference'] and ref['cell_selection'] == 'nearest'
    assert (ref['lat'], ref['lon']) == (13.251318, 80.31022) and ref['expected_cell'] == (13.251318, 80.31022)


def test_build_makes_nine_requests_and_writes_dev_and_sealed_separately(paths):
    s = FakeSession()
    man = hist.build_history(session=s, coastal_check={'passed': True})
    assert len(s.calls) == 9
    yearly, ref = s.calls[:8], s.calls[8]
    assert [c['start_date'] for c in yearly] == [f'{y}-01-01' for y in range(2017, 2025)]
    assert all(len(c['latitude'].split(',')) == 15 and 'cell_selection' not in c for c in yearly)
    assert ref['cell_selection'] == 'nearest' and (ref['start_date'], ref['end_date']) == ('2017-01-01', '2024-12-31')
    assert all(c['models'] == 'ecmwf_ifs' and c['timezone'] == 'GMT' for c in s.calls)
    dev = pd.read_csv(hist.HISTORY_DEV_FILE, parse_dates=['time_utc'])
    sealed = pd.read_csv(hist.HISTORY_SEALED_FILE, parse_dates=['time_utc'])
    assert sorted(dev['time_utc'].dt.year.unique()) == list(range(2017, 2024))
    assert sorted(sealed['time_utc'].dt.year.unique()) == [2024]
    assert set(dev['location_key']) == set(sealed['location_key']) and dev['location_key'].nunique() == 16
    assert man['files']['dev']['sha256'] == hist._sha256(hist.HISTORY_DEV_FILE)
    assert man['files']['sealed']['sha256'] == hist._sha256(hist.HISTORY_SEALED_FILE)
    assert set(man['structure_by_year']['2024']) == {'rows', 'hours_per_location', 'missing_share', 'grid_cells'}   # structure only
    with pytest.raises(FileExistsError):
        hist.build_history(session=FakeSession())                     # never overwrites


@pytest.mark.parametrize('bad', ['units', 'count', 'gap'])
def test_validation_failure_writes_nothing(paths, bad):
    with pytest.raises(ValueError):
        hist.build_history(session=FakeSession(corrupt={3: bad}))
    assert not any(os.path.exists(p) for p in (hist.HISTORY_DEV_FILE, hist.HISTORY_SEALED_FILE, hist.HISTORY_MANIFEST))


def test_2024_stays_sealed_unless_explicitly_unsealed_and_access_is_logged(paths, monkeypatch):
    hist.build_history(session=FakeSession(), coastal_check={'passed': True})
    dev = hist.load_history()
    assert dev['time_utc'].dt.year.max() == 2023
    with pytest.raises(PermissionError):
        hist.load_history(include_sealed=True, reason='peek')        # no unseal token
    monkeypatch.setenv(hist.UNSEAL_ENV, hist.UNSEAL_TOKEN)
    with pytest.raises(PermissionError):
        hist.load_history(include_sealed=True)                        # no reason
    full = hist.load_history(include_sealed=True, reason='final evaluation, configuration frozen')
    assert full['time_utc'].dt.year.max() == 2024
    log = [json.loads(l) for l in open(hist.SEALED_ACCESS_LOG, encoding='utf-8')]
    assert len(log) == 1 and log[0]['sealed_sha256'] == hist._sha256(hist.HISTORY_SEALED_FILE)


def test_coastal_reference_check_compares_cell_and_values():
    s = FakeSession()
    resp = s.get(None, params={'latitude': '13.251318', 'longitude': '80.31022', 'start_date': '2025-01-01',
                               'end_date': '2025-01-31'}).json()
    t = pd.to_datetime(resp['hourly']['time'], utc=True)
    local = pd.DataFrame({'zone_id': 1, 'time_utc': t, **{c: resp['hourly'][v] for v, c in hist.CANONICAL.items()}})
    ok = hist.verify_coastal_reference(session=FakeSession(), local_file_hourly=local)
    assert ok['passed'] and ok['cell_matches']
    local2 = local.copy()
    local2['temperature_2m_c'] += 0.5
    assert not hist.verify_coastal_reference(session=FakeSession(), local_file_hourly=local2)['passed']


def test_stability_flags_a_regime_shift_without_dropping_years(paths):
    hist.build_history(session=FakeSession(shift_year=2021), coastal_check={'passed': True})
    dev = hist.load_history()
    stats = hs.yearly_stats(dev)
    assert len(stats) == 16 * 7                                        # every location-year kept
    for c in ('missing_temperature', 'temp_mean', 'rh_mean', 'wet_hour_freq', 'mean_wet_amount', 'annual_total_mm',
              'wind_mean', 'wind_dir_mean_deg', 'temp_diurnal_amplitude', 'temp_peak_ist_hour', 'rain_peak_ist_hour'):
        assert c in stats
    flags = hs.flag_regime_changes(stats)
    summary = hs.summarize_flags(flags, 16)
    top = summary.iloc[0]
    assert top['year'] == '2021' and top['statistic'].startswith('temp') and top['share_of_locations'] == 1.0
    assert not flags[(flags['statistic'] == 'temp_mean') & (flags['year'] != '2021') & (flags['test'] == 'leave_one_year_out')].shape[0]


def test_history_code_is_isolated_from_the_live_pipeline():
    from src.forecasting import live_forecaster
    from src.live import live_pipeline
    for mod in (live_pipeline, live_forecaster):
        assert 'open_meteo_history' not in inspect.getsource(mod)
    src = inspect.getsource(hs.run_stability)
    assert 'include_sealed=False' in src and 'SEALED' not in src.replace('include_sealed', '')


# --- safety review: atomicity, CLI request order, seal, stability isolation -------------------------

OUTPUTS = ('HISTORY_DEV_FILE', 'HISTORY_SEALED_FILE', 'HISTORY_MANIFEST')


def _outputs():
    return [getattr(hist, n) for n in OUTPUTS]


def _nothing_published_and_no_tmp(tmp_path):
    assert not any(os.path.exists(p) for p in _outputs())
    assert not list(tmp_path.glob('*.tmp'))


def _local_jan2025():
    """Local-file zone 1 rows identical to what the fake coastal request returns for January 2025."""
    resp = FakeSession().get(None, params={'latitude': '13.251318', 'longitude': '80.31022',
                                           'start_date': '2025-01-01', 'end_date': '2025-01-31'}).json()
    t = pd.to_datetime(resp['hourly']['time'], utc=True)
    return pd.DataFrame({'zone_id': 1, 'time_utc': t, **{c: resp['hourly'][v] for v, c in hist.CANONICAL.items()}})


def test_failure_while_writing_temp_files_publishes_nothing(paths, monkeypatch):
    (paths / 'unrelated.txt').write_text('keep')
    monkeypatch.setattr(hist, '_manifest', lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        hist.build_history(session=FakeSession(), coastal_check={'passed': True})
    _nothing_published_and_no_tmp(paths)
    assert (paths / 'unrelated.txt').read_text() == 'keep'


def test_failure_during_publish_rolls_back_already_published_files(paths, monkeypatch):
    (paths / 'unrelated.txt').write_text('keep')
    real, n = os.replace, {'k': 0}

    def flaky(src, dst):
        n['k'] += 1
        if n['k'] == 3:                                                # dev + sealed published, manifest fails
            raise OSError('locked')
        return real(src, dst)
    monkeypatch.setattr(hist.os, 'replace', flaky)
    with pytest.raises(OSError):
        hist.build_history(session=FakeSession(), coastal_check={'passed': True})
    _nothing_published_and_no_tmp(paths)
    assert (paths / 'unrelated.txt').read_text() == 'keep'


@pytest.mark.parametrize('which', ['final', 'tmp'])
def test_pre_existing_output_or_tmp_refuses_before_any_request_and_is_untouched(paths, which):
    target = hist.HISTORY_SEALED_FILE + ('.tmp' if which == 'tmp' else '')
    with open(target, 'w') as f:
        f.write('pre-existing')
    s = FakeSession()
    with pytest.raises(FileExistsError):
        hist.build_history(session=s, coastal_check={'passed': True})
    assert s.calls == [] and open(target).read() == 'pre-existing'
    s2 = FakeSession()
    with pytest.raises(FileExistsError):
        hist.main(['--build'], session=s2, local_file_hourly=_local_jan2025())
    assert s2.calls == [] and open(target).read() == 'pre-existing'


def test_load_history_requires_manifest_and_matching_hashes(paths, monkeypatch):
    hist.build_history(session=FakeSession(), coastal_check={'passed': True})
    with open(hist.HISTORY_SEALED_FILE, 'ab') as f:
        f.write(b'tampered')
    monkeypatch.setenv(hist.UNSEAL_ENV, hist.UNSEAL_TOKEN)
    with pytest.raises(RuntimeError, match='sealed'):
        hist.load_history(include_sealed=True, reason='final evaluation')
    with open(hist.HISTORY_DEV_FILE, 'ab') as f:
        f.write(b'tampered')
    with pytest.raises(RuntimeError, match='development'):
        hist.load_history()
    os.remove(hist.HISTORY_MANIFEST)
    with pytest.raises(FileNotFoundError):
        hist.load_history()


def test_cli_build_makes_exactly_ten_requests_in_order_and_does_not_unseal(paths):
    s = FakeSession()
    assert hist.main(['--build'], session=s, local_file_hourly=_local_jan2025()) == 0
    assert len(s.calls) == 10
    first, yearly, last = s.calls[0], s.calls[1:9], s.calls[9]
    assert first['cell_selection'] == 'nearest' and (first['start_date'], first['end_date']) == ('2025-01-01', '2025-01-31')
    assert len(first['latitude'].split(',')) == 1
    assert [(c['start_date'], c['end_date']) for c in yearly] == [(f'{y}-01-01', f'{y}-12-31') for y in range(2017, 2025)]
    assert all(len(c['latitude'].split(',')) == 15 and 'cell_selection' not in c for c in yearly)
    assert last['cell_selection'] == 'nearest' and (last['start_date'], last['end_date']) == ('2017-01-01', '2024-12-31')
    assert all(c['models'] == 'ecmwf_ifs' and c['timezone'] == 'GMT' for c in s.calls)
    assert json.load(open(hist.HISTORY_MANIFEST))['coastal_reference_check']['passed'] is True
    assert not os.path.exists(hist.SEALED_ACCESS_LOG)                 # building never unseals


def test_cli_build_stops_after_failed_coastal_check_writing_nothing(paths):
    local = _local_jan2025()
    local['temperature_2m_c'] += 0.5
    s = FakeSession()
    assert hist.main(['--build'], session=s, local_file_hourly=local) == 2
    assert len(s.calls) == 1
    _nothing_published_and_no_tmp(paths)


def test_stability_never_reads_2024_even_with_unseal_token_set(paths, monkeypatch):
    hist.build_history(session=FakeSession(), coastal_check={'passed': True})
    with open(hist.HISTORY_SEALED_FILE, 'wb') as f:
        f.write(b'\x00not a csv')                                     # unreadable sealed file
    monkeypatch.setenv(hist.UNSEAL_ENV, hist.UNSEAL_TOKEN)             # even an unsealed environment
    opened, real_read = [], pd.read_csv

    def spy(path, *a, **k):
        opened.append(os.path.normpath(str(path)))
        return real_read(path, *a, **k)
    monkeypatch.setattr(pd, 'read_csv', spy)
    from src.collectors import open_meteo
    loc = _local_jan2025().assign(zone_id=1)
    loc = pd.concat([loc.assign(zone_id=z) for z in range(1, 16)], ignore_index=True)
    monkeypatch.setattr(open_meteo, 'load_local', lambda *a, **k: {'hourly': loc})
    res = hs.run_stability(include_2025_reference=True, out_dir=str(paths / 'out'))
    assert set(res['stats']['year']) == {str(y) for y in range(2017, 2024)} | {'2025_ref'}
    assert os.path.normpath(hist.HISTORY_SEALED_FILE) not in opened
    assert not os.path.exists(hist.SEALED_ACCESS_LOG)


def test_stability_rejects_a_dev_frame_containing_2024(paths, monkeypatch):
    bad = pd.DataFrame({'location_key': 'zone_02', 'time_utc': pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC'),
                        **{c: 1.0 for c in hist.CANONICAL.values()}})
    monkeypatch.setattr(hist, 'load_history', lambda include_sealed=False, reason=None: bad)
    with pytest.raises(RuntimeError, match='unexpected years'):
        hs.run_stability(include_2025_reference=False)


# --- rate-limit handling ------------------------------------------------------------------------------

class ThrottlingSession(FakeSession):
    """plan: {n: [(status, headers) | Exception, ...]} - failures returned BEFORE the n-th successful request
    (1-based; with the CLI, n=1 is the coastal check and n=6 is 2021)."""

    def __init__(self, plan, **kw):
        super().__init__(**kw)
        self.plan, self.attempts = {k: list(v) for k, v in plan.items()}, []

    def get(self, url, params=None, timeout=None):
        self.attempts.append(dict(params))
        queue = self.plan.get(len(self.calls) + 1)
        if queue:
            f = queue.pop(0)
            if isinstance(f, Exception):
                raise f
            return FakeResp(None, f[0], f[1])
        return super().get(url, params, timeout)


@pytest.fixture
def sleeps(monkeypatch):
    rec = []
    monkeypatch.setattr(hist.time, 'sleep', rec.append)
    monkeypatch.setattr(hist, 'REQUEST_PAUSE_S', 0.5)               # distinguishable from backoff waits
    monkeypatch.setattr(hist.random, 'uniform', lambda a, b: b)      # deterministic: maximum jitter
    return rec


def test_429_while_fetching_2021_recovers_without_repeating_the_coastal_check(paths, sleeps):
    s = ThrottlingSession({6: [(429, {}), (503, {})]})
    assert hist.main(['--build'], session=s, local_file_hourly=_local_jan2025()) == 0
    assert len(s.calls) == 10 and len(s.attempts) == 12
    assert [a['start_date'] for a in s.attempts].count('2025-01-01') == 1          # coastal check ran once
    assert [a['start_date'] for a in s.attempts].count('2021-01-01') == 3          # 2 failures + success
    assert [a['start_date'] for a in s.attempts].count('2020-01-01') == 1          # earlier years not re-fetched
    assert [w for w in sleeps if w != hist.REQUEST_PAUSE_S] == [15.0, 30.0]                         # backoff 15, 30 s (max jitter)
    assert all(os.path.exists(p) for p in _outputs())
    man = json.load(open(hist.HISTORY_MANIFEST))
    assert man['files']['dev']['sha256'] == hist._sha256(hist.HISTORY_DEV_FILE)


def test_retry_after_is_respected_in_seconds_and_http_date(paths, sleeps):
    s = ThrottlingSession({2: [(429, {'Retry-After': '120'})]})
    hist.build_history(session=s, coastal_check={'passed': True})
    assert 120.0 <= [w for w in sleeps if w != hist.REQUEST_PAUSE_S][0] <= 121.0                     # waits what the server asked
    assert hist._retry_after_seconds('Wed, 21 Oct 2015 07:28:00 GMT') == 0.0       # date in the past -> 0
    future = (pd.Timestamp.now(tz='UTC') + pd.Timedelta(seconds=90)).strftime('%a, %d %b %Y %H:%M:%S GMT')
    assert 80 <= hist._retry_after_seconds(future) <= 90
    assert hist._retry_after_seconds(None) is None and hist._retry_after_seconds('soon') is None


def test_excessive_retry_after_aborts_immediately_writing_nothing(paths, sleeps):
    s = ThrottlingSession({3: [(429, {'Retry-After': '3600'})]})
    with pytest.raises(hist.RetriesExhausted, match='3600'):
        hist.build_history(session=s, coastal_check={'passed': True})
    assert len(s.attempts) == 3
    _nothing_published_and_no_tmp(paths)


def test_exhausted_retries_fail_cleanly_and_keep_atomicity(paths, sleeps):
    (paths / 'unrelated.txt').write_text('keep')
    import requests
    s = ThrottlingSession({5: [(429, {}), (502, {}), requests.ConnectionError('reset'), (500, {}),
                               requests.Timeout('slow'), (429, {})]})
    with pytest.raises(hist.RetriesExhausted, match='HTTP 429 persisted after 6 attempts'):
        hist.build_history(session=s, coastal_check={'passed': True})
    assert len(s.attempts) == 4 + hist.MAX_ATTEMPTS
    assert [w for w in sleeps if w != hist.REQUEST_PAUSE_S] == [15.0, 30.0, 60.0, 120.0, 240.0]     # bounded, no sleep after last
    _nothing_published_and_no_tmp(paths)
    assert (paths / 'unrelated.txt').read_text() == 'keep'
    hist.build_history(session=FakeSession(), coastal_check={'passed': True})       # a clean re-run is possible
    assert all(os.path.exists(p) for p in _outputs())


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_permanent_client_errors_are_not_retried(paths, sleeps, status):
    s = ThrottlingSession({2: [(status, {'Retry-After': '1'})]})
    with pytest.raises(RuntimeError, match=f'HTTP {status}'):
        hist.build_history(session=s, coastal_check={'passed': True})
    assert len(s.attempts) == 2 and [w for w in sleeps if w != hist.REQUEST_PAUSE_S] == []
    _nothing_published_and_no_tmp(paths)


def test_rate_limited_coastal_check_retries_then_stops_cleanly(paths, sleeps):
    s = ThrottlingSession({1: [(429, {})] * hist.MAX_ATTEMPTS})
    with pytest.raises(hist.RetriesExhausted):
        hist.main(['--build'], session=s, local_file_hourly=_local_jan2025())
    assert len(s.attempts) == hist.MAX_ATTEMPTS and s.calls == []
    _nothing_published_and_no_tmp(paths)


def test_pause_between_successful_requests_is_60_seconds(paths, monkeypatch):
    rec = []
    monkeypatch.setattr(hist.time, 'sleep', rec.append)
    assert hist.REQUEST_PAUSE_S == 60.0
    hist.main(['--build'], session=FakeSession(), local_file_hourly=_local_jan2025())
    assert rec == [60.0] * 8                                          # between the 9 build requests, no retries
