"""
Scheduler output path: every scheduled cycle must write to the --output file (e.g. the demo file)
and never to data/live/current_predictions.json. No network: collection is stubbed.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collectors import live_collect
from src.live import live_pipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CURRENT = os.path.join(ROOT, 'data', 'live', 'current_predictions.json')
DEMO_ARG = os.path.join('data', 'live', 'demo_predictions.json')


class StopLoop(Exception):
    pass


def _stop_after(n, calls):
    def fake_sleep(seconds):
        calls.append(seconds)
        if len(calls) >= n:
            raise StopLoop()
    return fake_sleep


def test_cli_scheduler_passes_demo_output_to_every_cycle(monkeypatch):
    runs, sleeps = [], []
    monkeypatch.setattr(live_pipeline, 'LiveForecaster', lambda: 'fc')
    monkeypatch.setattr(live_pipeline, 'run_once', lambda **kw: runs.append(kw))
    monkeypatch.setattr(live_pipeline.time, 'sleep', _stop_after(3, sleeps))
    with pytest.raises(StopLoop):
        live_pipeline.main(['--output', DEMO_ARG, '--interval', '3600'])
    assert len(runs) == 3 and sleeps == [3600, 3600, 3600]
    for kw in runs:
        assert kw['output_path'] == os.path.abspath(DEMO_ARG) != CURRENT
        assert kw['fetch'] is True and kw['scheduled'] is True and kw['forecaster'] == 'fc'


def test_cli_once_and_default_paths_are_unchanged(monkeypatch):
    runs = []
    monkeypatch.setattr(live_pipeline, 'run_once', lambda **kw: runs.append(kw))
    live_pipeline.main(['--once', '--no-fetch', '--output', DEMO_ARG])
    assert runs[-1] == {'fetch': False, 'output_path': os.path.abspath(DEMO_ARG)}
    sched = []
    monkeypatch.setattr(live_pipeline, 'run_scheduler', lambda interval, output_path: sched.append((interval, output_path)))
    live_pipeline.main([])                                           # no --output: previous default kept
    assert sched == [(live_pipeline.REFRESH_INTERVAL_SECONDS, os.path.abspath(live_pipeline.OUTPUT_PATH))]


def test_scheduled_cycle_writes_demo_file_and_never_current_predictions(tmp_path, monkeypatch):
    before = (hashlib.sha256(open(CURRENT, 'rb').read()).hexdigest(), os.path.getmtime(CURRENT)) \
        if os.path.exists(CURRENT) else None
    now = pd.Timestamp.now(tz='UTC').floor('h') + pd.Timedelta(minutes=5)
    t = pd.date_range(now.floor('h') - pd.Timedelta(days=35), now.floor('h'), freq='h')
    ist_h = t.tz_convert('Asia/Kolkata').hour.to_numpy()
    pd.DataFrame([{'zone_id': z, 'time_utc': x, 'temperature_2m_c': 29 + 3 * np.sin(2 * np.pi * (h - 9) / 24),
                   'precipitation_mm': 0.0, 'relative_humidity_2m_pct': 70, 'wind_speed_10m_kmh': 10,
                   'wind_direction_10m_deg': 90, 'grid_latitude': 13.0, 'grid_longitude': 80.2, 'source': 'test'}
                  for z in range(1, 16) for x, h in zip(t, ist_h)]).to_csv(tmp_path / 'om.csv', index=False)
    status = {'open_meteo': {'ok': True, 'latest_record_utc': now.floor('h').isoformat()},
              'openaq': {'ok': False, 'error': 'stubbed in test'}}
    monkeypatch.setattr(live_collect, 'run_collection', lambda **kw: status)      # no network
    monkeypatch.setattr(live_collect, 'OPENAQ_FILE', str(tmp_path / 'none.csv'))
    monkeypatch.setattr(live_collect, 'OPEN_METEO_FILE', str(tmp_path / 'om.csv'))
    sleeps = []
    monkeypatch.setattr(live_pipeline.time, 'sleep', _stop_after(1, sleeps))
    demo = tmp_path / 'demo_predictions.json'
    with pytest.raises(StopLoop):
        live_pipeline.run_scheduler(3600, output_path=str(demo))
    d = json.loads(demo.read_text(encoding='utf-8'))
    assert d['meta']['run_mode'] == 'scheduler' and d['meta']['next_update_utc'] is not None
    assert len(d['zone_meta']) == 15 and sleeps == [3600]
    after = (hashlib.sha256(open(CURRENT, 'rb').read()).hexdigest(), os.path.getmtime(CURRENT)) \
        if os.path.exists(CURRENT) else None
    assert after == before                                           # current_predictions.json untouched
    assert not os.path.exists(CURRENT + '.tmp')
