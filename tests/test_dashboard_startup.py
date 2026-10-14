"""
Dashboard startup regression tests: every name dashboard/app.py imports from
dashboard.data_loader exists, and each page renders without an exception from the
existing data/live/current_predictions.json (read only; no collection is triggered).
"""

import ast
import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, 'dashboard', 'app.py')


def _imported_names(module_name):
    tree = ast.parse(open(APP, encoding='utf-8').read())
    return [a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == module_name
            for a in n.names]


def test_app_imports_exist_in_data_loader():
    names = _imported_names('dashboard.data_loader')
    assert 'aqi_observation' in names
    dl = importlib.import_module('dashboard.data_loader')
    assert [n for n in names if not hasattr(dl, n)] == []


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, 'data', 'live', 'current_predictions.json')),
                    reason='no live output file')
@pytest.mark.parametrize('page', ['Risk Map', 'Alert Center', 'Data & Coverage'])
def test_dashboard_page_renders_without_exception(page, monkeypatch):
    apptest = pytest.importorskip('streamlit.testing.v1')
    monkeypatch.chdir(ROOT)
    at = apptest.AppTest.from_file(APP, default_timeout=120)
    at.session_state['page'] = page
    at.run()
    assert [e.message for e in at.exception] == []


def test_predictions_path_can_point_to_a_separate_demo_output(monkeypatch, tmp_path):
    dl = importlib.import_module('dashboard.data_loader')
    default = dl.PREDICTIONS_PATH
    demo = tmp_path / 'demo_predictions.json'
    demo.write_text('{"meta": {"pipeline_version": "demo"}}', encoding='utf-8')
    monkeypatch.setenv('URBANPULSE_PREDICTIONS_PATH', str(demo))
    try:
        importlib.reload(dl)
        assert dl.PREDICTIONS_PATH == os.path.abspath(str(demo))
        assert dl.load_predictions() == {'meta': {'pipeline_version': 'demo'}}
    finally:
        monkeypatch.delenv('URBANPULSE_PREDICTIONS_PATH')
        importlib.reload(dl)
    assert dl.PREDICTIONS_PATH == default and default.endswith(os.path.join('data', 'live', 'current_predictions.json'))
