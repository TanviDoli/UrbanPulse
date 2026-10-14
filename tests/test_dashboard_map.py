"""
Dashboard map tests: zone polygons/colours/tooltips/popups, map configuration, click-to-select
without a rerun loop, and Quick Horizon Switch <-> dropdown sync.

st_folium is replaced in-process by a recorder that captures the real folium.Map built by
dashboard/app.py and returns a configurable component value — like the browser, it keeps
returning the last click on every rerun. Reads data/live/current_predictions.json only.
"""

import json
import os
import sys

import folium
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, 'dashboard', 'app.py')
GEOJSON = os.path.join(ROOT, 'data', 'raw', 'chennai_zones_2025.geojson')
LIVE_JSON = os.path.join(ROOT, 'data', 'live', 'current_predictions.json')

apptest = pytest.importorskip('streamlit.testing.v1')
from dashboard import data_loader as dl

pytestmark = pytest.mark.skipif(not os.path.exists(LIVE_JSON), reason='no live output file')

NAME_TO_ZONE = {
    'THIRUVOTTRIYUR': 1, 'MANALI': 2, 'MADHAVARAM': 3, 'TONDIARPET': 4, 'ROYAPURAM': 5,
    'THIRU-VIKA-NAGAR': 6, 'AMBATTUR': 7, 'ANNA NAGAR': 8, 'TEYNAMPET': 9, 'KODAMBAKKAM': 10,
    'VALASARVAKKAM': 11, 'ALANDUR': 12, 'ADYAR': 13, 'PERUNKUDI': 14, 'SHOZHANGANALLUR': 15,
}
MAX_RENDERS = 10


class FakeMap:
    """Stands in for streamlit_folium.st_folium; returns `value` on every call."""

    def __init__(self):
        self.value = None
        self.figs = []

    def __call__(self, fig, *args, **kwargs):
        self.figs.append(fig)
        if len(self.figs) > MAX_RENDERS:
            raise RuntimeError('rerun loop: map rendered more than %d times' % MAX_RENDERS)
        return self.value

    def click(self, zone_id, label, lat=13.0, lng=80.2):
        self.value = {'last_object_clicked_tooltip': f'Zone {zone_id}: {label}\nStale',
                      'last_active_drawing': None, 'last_object_clicked': {'lat': lat, 'lng': lng}}


@pytest.fixture
def app(monkeypatch):
    import streamlit_folium
    fake = FakeMap()
    monkeypatch.setattr(streamlit_folium, 'st_folium', fake)
    monkeypatch.chdir(ROOT)
    at = apptest.AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at, fake


def _zone_layers(fig):
    out = {}
    for child in fig._children.values():
        if isinstance(child, folium.GeoJson):
            feat = child.data['features'][0] if 'features' in child.data else child.data
            zid = NAME_TO_ZONE[feat['properties']['name_of_th'].upper().strip()]
            out[zid] = (child, feat)
    return out


# ── zones, colours, tooltips, popups, configuration ─────────────────────────
def test_map_draws_all_15_zones_unchanged_with_risk_colours(app):
    at, fake = app
    fig = fake.figs[-1]
    layers = _zone_layers(fig)
    assert sorted(layers) == list(range(1, 16))
    source = {NAME_TO_ZONE[f['properties']['name_of_th'].upper().strip()]: f
              for f in json.load(open(GEOJSON, encoding='utf-8'))['features']}
    data = dl.load_predictions()
    now = pd.Timestamp.now(tz='UTC')
    risks = dl.get_zone_risk_map(data, now)
    for zid, (layer, feat) in layers.items():
        assert feat['geometry'] == source[zid]['geometry']                 # polygon not altered
        style = layer.style_function(feat)
        assert style['fillColor'] == dl.RISK_LEVEL_COLOR[risks[zid]]       # existing colouring logic
        is_est = dl.zone_aqi_basis(data, zid) == 'MODEL_ESTIMATE'
        assert (style['dashArray'] == '6 4') == is_est                     # estimate outline kept
        tips = [c for c in layer._children.values() if isinstance(c, folium.Tooltip)]
        assert tips and f'Zone {zid}: {dl.ZONE_MAP[zid]}' in tips[0].text  # tooltip used by click parsing
        assert any(isinstance(c, folium.Popup) for c in layer._children.values())   # popup kept
    # The committed output is stale: no zone may be painted as low risk (green).
    assert dl.RISK_LEVEL_COLOR['Low'] not in {l.style_function(f)['fillColor'] for l, f in layers.values()}


def test_map_configuration_labels_and_controls(app):
    at, fake = app
    fig = fake.figs[-1]
    markers = [c for c in fig._children.values() if isinstance(c, folium.Marker)]
    assert len(markers) == 15                                              # one label per zone
    assert fig.options.get('zoom_control', fig.options.get('zoomControl', True)) is not False
    tiles = [c for c in fig._children.values() if isinstance(c, folium.TileLayer)]
    assert len(tiles) == 1 and tiles[0].tiles.startswith('https://') and tiles[0].options.get('attribution')


# ── click-to-select, no rerun loop ───────────────────────────────────────────
def test_map_click_selects_zone_once_without_rerun_loop(app):
    at, fake = app
    fake.figs.clear()
    fake.click(7, 'Ambattur')
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state['selected_zone'] == 7
    assert at.selectbox(key='zone_dropdown').value == 7                    # dropdown agrees with map
    assert len(fake.figs) <= 3


def test_dropdown_change_is_not_reverted_by_stale_map_click(app):
    at, fake = app
    fake.click(7, 'Ambattur')
    at.run()
    fake.figs.clear()
    at.selectbox(key='zone_dropdown').select(3).run()                      # map still reports zone 7
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state['selected_zone'] == 3
    assert len(fake.figs) <= 3


def test_repeat_click_on_same_zone_after_dropdown_change_selects_it(app):
    at, fake = app
    fake.click(7, 'Ambattur', lat=13.10)
    at.run()
    at.selectbox(key='zone_dropdown').select(3).run()
    fake.click(7, 'Ambattur', lat=13.11)                                   # new click, same zone
    at.run()
    assert at.session_state['selected_zone'] == 7 and at.selectbox(key='zone_dropdown').value == 7


def test_selected_zone_survives_page_switch(app):
    at, fake = app
    at.selectbox(key='zone_dropdown').select(5).run()
    at.button(key='nav_Alert Center').click().run()
    at.button(key='nav_Risk Map').click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state['selected_zone'] == 5 and at.selectbox(key='zone_dropdown').value == 5


# ── Quick Horizon Switch <-> dropdown ────────────────────────────────────────
def test_quick_horizon_buttons_update_dropdown_and_stick(app):
    at, fake = app
    for label in ('1 Hour', '24 Hours', '3 Hours'):
        fake.figs.clear()                                                  # loop guard is per interaction
        at.button(key=f'hz_btn_{label}').click().run()
        assert not at.exception, [e.message for e in at.exception]
        assert at.session_state['horizon'] == label
        assert at.selectbox(key='horizon_dropdown').value == label
        assert len(fake.figs) <= 3
    fake.figs.clear()
    at.selectbox(key='horizon_dropdown').select('Current').run()
    assert at.session_state['horizon'] == 'Current'
    fake.figs.clear()
    at.button(key='hz_btn_6 Hours').click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state['horizon'] == '6 Hours'
    md = ' '.join(m.value for m in at.markdown)
    assert 'Forecast: data time + 6 hours' in md                           # the cards follow the button
