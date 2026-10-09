"""
UrbanPulse Chennai — Live Environmental Intelligence Dashboard
===============================================================
Phase 3: Full redesign consuming data/live/current_predictions.json
produced by the Phase 2 HMM+MRF live pipeline.

Layout:
  LEFT SIDEBAR  |  CHENNAI MAP (center)  |  ZONE DETAIL  |  ALERT CENTER
  ──────────────────────────────────────────────────────────────────────
                         CHENNAI OVERVIEW SUMMARY

Data flow:
  data/live/current_predictions.json  (written by src/live/live_pipeline.py)
      -> dashboard/data_loader.py     (parsing / typed accessors)
      -> dashboard/app.py             (display only — NO model calls)
"""

import os
import sys
import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dashboard.data_loader import (
    load_predictions,
    get_meta,
    get_zone_meta_map,
    get_predictions_for_zone,
    get_all_zones_snapshot,
    get_risk_level,
    get_zone_risk_map,
    get_dominant_risk_for_zone,
    get_summary_stats,
    get_alerts,
    format_ist_timestamp,
    get_countdown_minutes,
    generate_zone_insights,
    zone_aqi_basis,
    get_source_status,
    data_is_stale,
    STATUS_EXPLANATION,
    ZONE_MAP,
    RISK_LEVEL_COLOR,
    RISK_LEVEL_BG,
)

GEOJSON_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "chennai_zones_2025.geojson")

GEOJSON_NAME_MAP = {
    'THIRUVOTTRIYUR': 1, 'MANALI': 2, 'MADHAVARAM': 3, 'TONDIARPET': 4,
    'ROYAPURAM': 5, 'THIRU-VIKA-NAGAR': 6, 'AMBATTUR': 7, 'ANNA NAGAR': 8,
    'TEYNAMPET': 9, 'KODAMBAKKAM': 10, 'VALASARVAKKAM': 11, 'ALANDUR': 12,
    'ADYAR': 13, 'PERUNKUDI': 14, 'SHOZHANGANALLUR': 15,
}

HORIZON_OPTIONS = ["Current", "1 Hour", "3 Hours", "6 Hours", "24 Hours"]
HORIZON_KEY_MAP = {
    "Current": "current",
    "1 Hour":  "1h",
    "3 Hours": "3h",
    "6 Hours": "6h",
    "24 Hours": "24h",
}

SEVERITY_ICON = {
    "High":        "triangle-exclamation",
    "Critical":    "circle-exclamation",
    "Moderate":    "circle-info",
    "Low":         "circle-check",
    "Unmonitored": "circle-question",
    "Unavailable": "circle-xmark",
}

VAR_ICON = {
    "AQI":         "🍃",
    "rainfall":    "🌧️",
    "temperature": "🌡️",
}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UrbanPulse Chennai",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

/* ── Global dark background ── */
.stApp { background: #0b0f1a; }
.main .block-container {
    padding: 0.75rem 1rem 0.5rem 1rem !important;
    max-width: 100% !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.18) !important;
    padding: 0 !important;
    min-width: 220px !important;
    max-width: 240px !important;
}
section[data-testid="stSidebar"] > div:first-child { padding: 1rem 1rem !important; }
[data-testid="stSidebarNavItems"] { display: none; }

/* ── Cards ── */
.up-card {
    background: #111827;
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
.up-card-sm {
    background: #111827;
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4b5563;
    margin: 0 0 0.4rem 0;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Overview cards (bottom strip) ── */
.overview-strip {
    display: flex;
    gap: 0.65rem;
    margin-top: 0.6rem;
}
.ov-card {
    flex: 1;
    background: #111827;
    border: 1px solid rgba(59,130,246,0.14);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.ov-val { font-size: 1.45rem; font-weight: 800; color: #e5e7eb; margin: 0; line-height: 1.2; }
.ov-lbl { font-size: 0.62rem; font-weight: 600; color: #6b7280; text-transform: uppercase;
           letter-spacing: 0.08em; margin: 0.15rem 0 0 0; }

/* ── Sidebar nav button ── */
.nav-btn {
    display: block;
    width: 100%;
    padding: 0.5rem 0.7rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #9ca3af;
    cursor: pointer;
    margin-bottom: 0.2rem;
    border: none;
    background: transparent;
    text-align: left;
    transition: background 0.15s;
}
.nav-btn-active {
    background: rgba(59,130,246,0.15);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,0.25);
}

/* ── Legend dot ── */
.leg-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Zone detail env cards ── */
.env-cards { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
.env-card {
    flex: 1;
    background: #1a2235;
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px;
    padding: 0.65rem 0.6rem;
    text-align: center;
}
.env-icon { font-size: 1.1rem; color: #60a5fa; margin-bottom: 0.2rem; }
.env-val { font-size: 1.05rem; font-weight: 700; color: #e5e7eb; }
.env-lbl { font-size: 0.58rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.07em; }

/* ── Prediction cards ── */
.pred-cards { display: flex; gap: 0.5rem; margin: 0.4rem 0; }
.pred-card {
    flex: 1;
    background: #1a2235;
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px;
    padding: 0.65rem 0.6rem;
    text-align: center;
    min-height: 90px;
}
.pred-icon { font-size: 1.0rem; margin-bottom: 0.15rem; }
.pred-val { font-size: 1.05rem; font-weight: 700; color: #e5e7eb; line-height: 1.2; }
.pred-unit { font-size: 0.6rem; color: #9ca3af; }
.pred-status { font-size: 0.6rem; font-weight: 700; margin-top: 0.2rem; border-radius: 8px; padding: 1px 6px; }
.pred-na { font-size: 0.65rem; color: #6b7280; font-style: italic; }

/* ── Alert cards ── */
.alert-card {
    background: #111827;
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 10px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #f97316;
}
.alert-card-moderate { border-left-color: #f59e0b; }
.alert-card-high     { border-left-color: #f97316; }
.alert-card-critical { border-left-color: #ef4444; }

/* ── Insight items ── */
.insight-item {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 0.3rem;
    line-height: 1.4;
}

/* ── Timestamp display ── */
.ts-block { font-size: 0.72rem; color: #6b7280; line-height: 1.6; }
.ts-block strong { color: #9ca3af; font-weight: 600; }

/* ── Countdown box ── */
.countdown-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    text-align: center;
    font-size: 0.7rem;
    color: #6b7280;
}
.countdown-box .mins { font-size: 1.3rem; font-weight: 800; color: #60a5fa; }

/* ── Header bar ── */
.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #111827;
    border: 1px solid rgba(59,130,246,0.14);
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 0.75rem;
}
.dash-title { font-size: 1.1rem; font-weight: 800; color: #e5e7eb; margin: 0; }
.dash-sub { font-size: 0.7rem; color: #4b5563; margin: 0; }
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 5px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* ── Scrollable alert panel ── */
.alert-scroll { max-height: 480px; overflow-y: auto; padding-right: 2px; }
.alert-scroll::-webkit-scrollbar { width: 4px; }
.alert-scroll::-webkit-scrollbar-track { background: #111827; }
.alert-scroll::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 2px; }

/* ── Zone chip ── */
.zone-chip {
    display: inline-block;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px;
    padding: 1px 9px;
    font-size: 0.65rem;
    color: #60a5fa;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* ── Unavailable notice ── */
.unavail-notice {
    background: rgba(75,85,99,0.12);
    border: 1px dashed rgba(75,85,99,0.35);
    border-radius: 8px;
    padding: 0.4rem 0.65rem;
    font-size: 0.68rem;
    color: #6b7280;
    font-style: italic;
    text-align: center;
}

/* ── Column divider fix for Streamlit ── */
[data-testid="stHorizontalBlock"] > div { min-width: 0; }

/* Column padding removal */
[data-testid="column"] { padding: 0 0.25rem !important; }

/* Remove default Streamlit padding */
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = 14   # default: Perungudi
if "horizon" not in st.session_state:
    st.session_state.horizon = "Current"
if "page" not in st.session_state:
    st.session_state.page = "Risk Map"
if "last_data_mtime" not in st.session_state:
    st.session_state.last_data_mtime = 0

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING (with file-modification-time cache invalidation)
# ══════════════════════════════════════════════════════════════════════════
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "live", "current_predictions.json")

@st.cache_data(ttl=60, show_spinner=False)
def _load_cached(mtime: float):
    return load_predictions()

def get_data():
    try:
        mtime = os.path.getmtime(PREDICTIONS_PATH)
    except OSError:
        mtime = 0
    return _load_cached(mtime), mtime

data, data_mtime = get_data()

# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _parse_ts(raw):
    if not raw:
        return None
    try:
        t = pd.Timestamp(raw)
        return t.tz_localize("UTC") if t.tzinfo is None else t
    except Exception:
        return None


def risk_badge(level: str, small: bool = False) -> str:
    color = RISK_LEVEL_COLOR.get(level, "#6b7280")
    bg    = RISK_LEVEL_BG.get(level, "rgba(107,114,128,0.12)")
    sz    = "0.62rem" if small else "0.7rem"
    return (
        f'<span class="risk-badge" style="background:{bg};color:{color};'
        f'border:1px solid {color}44;font-size:{sz};">{level.upper()}</span>'
    )

def _safe_float(v) -> Optional[float]:
    if v is None: return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def _aqi_label(v: Optional[float]) -> str:
    if v is None: return ""
    if v <= 50:  return "Good"
    if v <= 100: return "Satisfactory"
    if v <= 200: return "Moderate"
    if v <= 300: return "Poor"
    if v <= 400: return "Very Poor"
    return "Severe"

def _rf_label(v: Optional[float]) -> str:
    if v is None: return ""
    if v <= 0:    return "Dry"
    if v < 2.5:   return "Light rain"
    if v < 7.5:   return "Moderate rain"
    if v < 64.5:  return "Heavy rain"
    return "Very heavy rain"

def _var_pred_display(data, zone_id, hz_key, variable) -> tuple:
    """Return (value_str, unit_str, risk_level, note) for display. Missing data is never shown as a value."""
    if hz_key == "current":
        zm = get_zone_meta_map(data).get(zone_id, {})
        if variable == "AQI":
            v = _safe_float(zm.get("cpcb_aqi_computed"))
            if v is not None:
                age = _safe_float(zm.get("aqi_age_hours"))
                when = f" {age:.0f} h ago" if age is not None and age >= 2 else ""
                return (f"{v:.0f}", "AQI", "", f"{_aqi_label(v)} · measured{when}")
            state = zm.get("aqi_state")
            if state == "MEASURED_STALE":
                last = _safe_float(zm.get("last_measured_aqi"))
                return (None, None, None, f"Station data stale (last {last:.0f} AQI, {zm.get('aqi_age_hours')} h ago)"
                        if last is not None else "Station data stale")
            return (None, None, None, "No air-quality station" if state == "NO_STATION" else "No live station data")
        if variable == "rainfall":
            v = _safe_float(zm.get("rainfall_mm"))
            if v is None:
                return (None, None, None, "Weather data unavailable")
            return (f"{v:.1f}", "mm/h", "", f"{_rf_label(v)} · reanalysis")
        if variable == "temperature":
            v = _safe_float(zm.get("temp_c_current"))
            if v is None:
                return (None, None, None, "Weather data unavailable")
            return (f"{v:.1f}", "°C", "", "2 m air · reanalysis")
    else:
        if variable == "temperature" and hz_key in ("1h", "3h", "6h"):
            return (None, None, None, "No sub-daily temperature model")
        zp = get_predictions_for_zone(data, zone_id, variable, hz_key)
        if zp is None:
            return (None, None, None, "Not produced")
        status = zp.get("alert_status", "")
        if zp.get("availability") == "UNAVAILABLE":
            return (None, None, None, STATUS_EXPLANATION.get(status, "Unavailable"))
        level = get_risk_level(status) if status in ("NORMAL", "WATCH", "WARNING") else ""
        if variable == "AQI":
            v = _safe_float(zp.get("mrf_pred"))
            basis = "estimate" if zp.get("availability") == "MODEL_ESTIMATE" else "forecast"
            return (f"{v:.0f}" if v is not None else "—", "AQI", level, f"{_aqi_label(v)} · {basis}")
        if variable == "rainfall":
            if hz_key == "24h":
                v = _safe_float(zp.get("accum_24h_mm"))
                return (f"{v:.1f}" if v is not None else "—", "mm/24h", level, "24 h total · reanalysis-driven")
            v = _safe_float(zp.get("mrf_pred"))
            return (f"{v:.1f}" if v is not None else "—", "mm/h", "", "hourly · reanalysis-driven")
        if variable == "temperature":
            v = _safe_float(zp.get("next_day_tmean_pred_c"))
            return (f"{v:.1f}" if v is not None else "—", "°C", "", f"Daily mean {zp.get('target_day_ist', '')} · low skill")
    return (None, None, None, "Unavailable")



def pred_card_html(icon: str, label: str, val_str, unit_str, level: str, note: str) -> str:
    color = RISK_LEVEL_COLOR.get(level, "#9ca3af") if level else "#9ca3af"
    bg    = RISK_LEVEL_BG.get(level, "transparent") if level else "transparent"
    if val_str is None:
        inner = f'<div class="pred-na">{note}</div>'
    else:
        badge = (f'<div class="pred-status" style="background:{bg};color:{color};">'
                 f'{note}</div>') if note and level else (
                 f'<div class="pred-unit">{note}</div>' if note else "")
        inner = (
            f'<div class="pred-val">{val_str}'
            f'<span class="pred-unit"> {unit_str}</span></div>'
            f'{badge}'
        )
    return f"""
<div class="pred-card">
  <div class="pred-icon" style="color:{color};">{icon}</div>
  <div class="env-lbl">{label}</div>
  {inner}
</div>"""


# ══════════════════════════════════════════════════════════════════════════
# LEFT SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / Brand
    st.markdown("""
<div style="padding:0.5rem 0 1rem 0; border-bottom:1px solid rgba(59,130,246,0.12); margin-bottom:0.9rem;">
  <div style="display:flex;align-items:center;gap:0.5rem;">
    <div style="width:30px;height:30px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);
         border-radius:8px;display:flex;align-items:center;justify-content:center;
         font-size:16px;flex-shrink:0;">&#127758;</div>
    <div>
      <div style="font-size:1.0rem;font-weight:800;color:#e5e7eb;letter-spacing:-0.02em;">UrbanPulse</div>
      <div style="font-size:0.62rem;color:#4b5563;font-weight:500;letter-spacing:0.05em;">CHENNAI</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sec-label">Navigation</div>', unsafe_allow_html=True)

    for pg_name, pg_icon in [("Risk Map", "&#128506;"), ("Alert Center", "&#128680;"), ("Data & Coverage", "&#128202;")]:
        is_active = st.session_state.page == pg_name
        style = "nav-btn nav-btn-active" if is_active else "nav-btn"
        if st.button(f"{pg_name}", key=f"nav_{pg_name}", use_container_width=True):
            st.session_state.page = pg_name
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Timestamps
    st.markdown('<div class="sec-label">Data Status</div>', unsafe_allow_html=True)
    if data:
        meta = get_meta(data)
        srcs = get_source_status(data)
        stale = data_is_stale(data)
        ts_gen = format_ist_timestamp(meta.get("generated_at_utc_parsed"))
        ts_ok  = format_ist_timestamp(_parse_ts(meta.get("last_successful_update_utc")))
        mins   = get_countdown_minutes(data)
        gen_age_h = None
        if meta.get("generated_at_utc_parsed") is not None:
            gen_age_h = (pd.Timestamp.now(tz="UTC") - meta["generated_at_utc_parsed"]).total_seconds() / 3600
        dot = "#ef4444" if stale else "#22c55e"
        src_lines = ""
        for name, label in (("openaq", "Air quality (OpenAQ, measured)"), ("open_meteo", "Weather (Open-Meteo, reanalysis)")):
            sinfo = srcs.get(name, {})
            latest = format_ist_timestamp(_parse_ts(sinfo.get("latest_record_utc")))
            age = sinfo.get("data_age_hours")
            flag = ("<span style='color:#f87171;'>STALE</span>" if sinfo.get("stale") else
                    "<span style='color:#22c55e;'>current</span>")
            err = (f"<br><span style='color:#f87171;'>Last fetch failed: {sinfo.get('error')}</span>"
                   if sinfo.get("ok") is False else "")
            src_lines += (f"<br><strong>{label}</strong><br>latest data: {latest}"
                          f"{f' ({age:.0f} h old)' if age is not None else ''} · {flag}{err}")
        overdue = (f"<br><span style='color:#f87171;'>Pipeline output is {gen_age_h:.1f} h old (overdue)</span>"
                   if gen_age_h is not None and gen_age_h > 2 * (meta.get("refresh_interval_seconds") or 3600) / 3600 else "")
        st.markdown(f"""
<div class="ts-block">
  <span class="live-dot" style="background:{dot};"></span><strong>{'Data partly stale' if stale else 'Live'}</strong><br>
  <strong>Last successful update:</strong> {ts_ok}<br>
  <strong>Generated:</strong> {ts_gen}{overdue}
  {src_lines}
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="countdown-box" style="margin-top:0.6rem;">
  <div style="font-size:0.6rem;color:#4b5563;margin-bottom:0.15rem;">NEXT SCHEDULED UPDATE</div>
  <div class="mins">{format_ist_timestamp(meta.get("next_update_utc_parsed"))}</div>
  <div style="font-size:0.58rem;color:#374151;">in {mins if mins is not None else '--'} min · hourly cycle</div>
</div>
""", unsafe_allow_html=True)

        if st.button("Refresh now", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.markdown('<div class="ts-block" style="color:#f87171;">Pipeline output not found or unreadable.<br>Run: <code>python -m src.live.live_pipeline --once</code></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk Legend
    st.markdown('<div class="sec-label">Risk Level</div>', unsafe_allow_html=True)
    for lvl, col in [("Low", "#22c55e"), ("Moderate", "#f59e0b"), ("High", "#f97316")]:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#9ca3af;padding:3px 0;">'
            f'<span class="leg-dot" style="background:{col};"></span>{lvl}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="font-size:0.75rem;color:#9ca3af;padding:3px 0;">'
        '<span class="leg-dot" style="background:#374151;"></span>Unavailable (no evaluated forecast)</div>'
        '<div style="font-size:0.75rem;color:#9ca3af;padding:3px 0;">'
        '<span class="leg-dot" style="background:transparent;border:2px dashed #e5e7eb;"></span>'
        'Dashed outline = AQI is an MRF estimate (no usable monitor)</div>'
        '<div style="font-size:0.62rem;color:#4b5563;padding-top:4px;">Low/Moderate/High = UrbanPulse decision '
        'rule on P(threshold crossed): &lt;20%, 20–60%, ≥60%. Thresholds: CPCB AQI &gt; 200; IMD heavy rain ≥ 64.5 mm/24 h.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════
if data is None:
    st.markdown("""
<div class="up-card" style="text-align:center;padding:3rem;">
  <div style="font-size:2rem;margin-bottom:1rem;">&#9888;</div>
  <div style="font-size:1.1rem;color:#e5e7eb;font-weight:700;margin-bottom:0.5rem;">Waiting for UrbanPulse Data</div>
  <div style="font-size:0.8rem;color:#6b7280;">
    The live prediction file was not found.<br>
    Run the pipeline first:<br><br>
    <code style="background:#1a2235;padding:4px 10px;border-radius:6px;color:#60a5fa;">
      python -m src.live.live_pipeline --once
    </code>
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Derived data ──────────────────────────────────────────────────────────────
zone_risks     = get_zone_risk_map(data)
zone_meta_map  = get_zone_meta_map(data)
summary_stats  = get_summary_stats(data)
active_alerts  = get_alerts(data, active_only=True)
meta           = get_meta(data)
sel_zone       = st.session_state.selected_zone
sel_horizon    = st.session_state.horizon
hz_key         = HORIZON_KEY_MAP[sel_horizon]

# ══════════════════════════════════════════════════════════════════════════
# PAGE: RISK MAP
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Risk Map":

    # Header bar
    n_active = len(active_alerts)
    alert_str = f"{n_active} Active Alert{'s' if n_active != 1 else ''}"
    sev_colors = {"High": "#f97316", "Moderate": "#f59e0b", "Critical": "#ef4444"}
    dominant_sev_color = "#22c55e"
    if any(a.get("severity") == "HIGH" for a in active_alerts):
        dominant_sev_color = "#f97316"
    if any(a.get("severity") == "CRITICAL" for a in active_alerts):
        dominant_sev_color = "#ef4444"

    ts_display = format_ist_timestamp(meta.get("timestamp_utc_parsed"))
    st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">Chennai Environmental Intelligence</div>
    <div class="dash-sub">HMM + MRF &nbsp;|&nbsp; AQI data as of {ts_display}{' &nbsp;|&nbsp; <span style="color:#f87171;">some sources stale</span>' if data_is_stale(data) else ''}</div>
  </div>
  <div style="display:flex;align-items:center;gap:0.75rem;">
    <div style="background:rgba(249,115,22,0.12);border:1px solid {dominant_sev_color}44;
         border-radius:20px;padding:3px 12px;font-size:0.7rem;font-weight:700;color:{dominant_sev_color};">
      {alert_str}
    </div>
    <div style="font-size:0.7rem;color:#4b5563;">
      <span class="live-dot" style="background:{'#ef4444' if data_is_stale(data) else '#22c55e'};"></span>{'Stale' if data_is_stale(data) else 'Live'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Three-column layout: MAP | ZONE DETAIL | ALERTS ──────────────────
    col_map, col_detail, col_alerts = st.columns([5, 2.5, 2.2], gap="small")

    # ════════════════════════════════════════════════════
    # MAP COLUMN
    # ════════════════════════════════════════════════════
    with col_map:
        # Load GeoJSON
        try:
            with open(GEOJSON_PATH, encoding="utf-8") as f:
                geojson_data = json.load(f)
        except Exception:
            st.error("Could not load GeoJSON.")
            st.stop()

        m = folium.Map(
            location=[13.075, 80.235],
            zoom_start=12,
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr="CartoDB DarkMatter",
            control_scale=True,
        )

        # Draw zones
        for feat in geojson_data["features"]:
            raw_name = feat["properties"].get("name_of_th", "").upper().strip()
            zid = GEOJSON_NAME_MAP.get(raw_name)
            if zid is None:
                continue

            zlabel   = ZONE_MAP.get(zid, f"Zone {zid}")
            risk_lvl = zone_risks.get(zid, "Unavailable")
            is_est   = zone_aqi_basis(data, zid) == "MODEL_ESTIMATE"
            fill_col = RISK_LEVEL_COLOR.get(risk_lvl, "#6b7280")
            is_sel   = (zid == sel_zone)

            zm = zone_meta_map.get(zid, {})
            cur_aqi  = _safe_float(zm.get("cpcb_aqi_computed"))
            cur_rf   = _safe_float(zm.get("rainfall_mm"))
            cur_temp = _safe_float(zm.get("temp_c_current"))

            aqi_str  = (f"{cur_aqi:.0f}" if cur_aqi is not None else
                        {"MEASURED_STALE": "stale", "NO_STATION": "no station"}.get(zm.get("aqi_state"), "N/A"))
            rf_str   = f"{cur_rf:.1f} mm/h" if cur_rf is not None else "N/A"
            temp_str = f"{cur_temp:.1f} °C" if cur_temp is not None else "N/A"
            est_note = ("<div style='margin-top:4px;font-size:10px;color:#9ca3af;'>AQI risk is an MRF estimate "
                        "(no usable monitor)</div>" if is_est else "")

            popup_html = f"""
<div style="font-family:Inter,sans-serif;padding:8px;min-width:190px;
     background:#111827;color:#e5e7eb;border-radius:10px;font-size:12px;">
  <div style="font-weight:700;font-size:13px;color:#e5e7eb;margin-bottom:5px;">
    Zone {zid}: {zlabel}
  </div>
  <div style="background:{RISK_LEVEL_BG.get(risk_lvl,'rgba(107,114,128,0.12)')};
       border:1px solid {fill_col}55;border-radius:6px;
       padding:2px 8px;margin-bottom:7px;font-size:11px;
       font-weight:700;color:{fill_col};display:inline-block;">
    {risk_lvl.upper()}
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:11px;">
    <tr><td style="color:#6b7280;padding:1.5px 0;">AQI (measured)</td><td style="font-weight:600;text-align:right;">{aqi_str}</td></tr>
    <tr><td style="color:#6b7280;padding:1.5px 0;">Rain (reanalysis)</td><td style="font-weight:600;text-align:right;">{rf_str}</td></tr>
    <tr><td style="color:#6b7280;padding:1.5px 0;">Temp (reanalysis)</td><td style="font-weight:600;text-align:right;">{temp_str}</td></tr>
  </table>{est_note}
  <div style="margin-top:6px;font-size:10px;color:#4b5563;text-align:center;">Click to select zone</div>
</div>"""

            folium.GeoJson(
                feat,
                style_function=lambda x, c=fill_col, sel=is_sel, est=is_est: {
                    "fillColor": c,
                    "color": "#60a5fa" if sel else ("#e5e7eb" if est else "#1e3a5f"),
                    "dashArray": "6 4" if est else None,
                    "weight": 3 if sel else (2 if est else 1.2),
                    "fillOpacity": 0.72 if sel else 0.55,
                    "opacity": 1 if sel else 0.8,
                },
                tooltip=folium.Tooltip(
                    f"<b style='font-family:Inter;'>Zone {zid}: {zlabel}</b>"
                    f"<br><span style='color:{fill_col};font-weight:700;'>{risk_lvl}</span>",
                    style="font-family:Inter,sans-serif;font-size:12px;background:#111827;color:#e5e7eb;border:none;"
                ),
                popup=folium.Popup(popup_html, max_width=210),
            ).add_to(m)

            # Add zone label
            try:
                from shapely.geometry import shape as shp
                centroid = shp(feat["geometry"]).centroid
                folium.Marker(
                    location=[centroid.y, centroid.x],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-family:Inter,sans-serif;font-size:9px;
                            font-weight:700;color:#e5e7eb;
                            text-shadow:0 0 4px #0b0f1a,0 0 4px #0b0f1a;
                            white-space:nowrap;text-align:center;
                            transform:translate(-50%,-50%);">{zlabel}</div>""",
                        icon_size=(100, 20),
                        icon_anchor=(50, 10),
                    ),
                    tooltip=folium.Tooltip(f"<b style='font-family:Inter;'>Zone {zid}: {zlabel}</b><br><span style='color:{fill_col};font-weight:700;'>{risk_lvl}</span>")
                ).add_to(m)
            except Exception:
                pass

        # Map output — capture click events
        map_output = st_folium(
            m,
            width=None,
            height=460,
            returned_objects=["last_object_clicked_tooltip", "last_active_drawing"],
            use_container_width=True,
            key="main_map",
        )

        # Handle zone click via tooltip text or active drawing
        if map_output:
            clicked_id = None
            if map_output.get("last_object_clicked_tooltip"):
                tooltip_text = map_output["last_object_clicked_tooltip"]
                try:
                    zone_part = tooltip_text.split("Zone ")[-1].split(":")[0].strip()
                    cid = int(zone_part)
                    if 1 <= cid <= 15:
                        clicked_id = cid
                except (ValueError, IndexError):
                    pass
            if clicked_id is None and map_output.get("last_active_drawing"):
                props = (map_output["last_active_drawing"] or {}).get("properties", {})
                raw_name = props.get("name_of_th", "").upper().strip()
                if raw_name in GEOJSON_NAME_MAP:
                    clicked_id = GEOJSON_NAME_MAP[raw_name]

            if clicked_id and clicked_id != st.session_state.selected_zone:
                st.session_state.selected_zone = clicked_id
                sel_zone = clicked_id
                st.rerun()

        # Horizon quick buttons below the map (synced with session state)
        st.markdown(
            '<div class="sec-label" style="margin-top:0.5rem;">Quick Horizon Switch</div>',
            unsafe_allow_html=True,
        )
        hz_cols = st.columns(len(HORIZON_OPTIONS))
        for i, hz_opt in enumerate(HORIZON_OPTIONS):
            with hz_cols[i]:
                is_cur = (hz_opt == sel_horizon)
                btn_style = (
                    "background:rgba(59,130,246,0.2);border:1px solid rgba(59,130,246,0.4);"
                    "color:#60a5fa;" if is_cur else
                    "background:#111827;border:1px solid rgba(59,130,246,0.1);color:#6b7280;"
                )
                if st.button(
                    hz_opt,
                    key=f"hz_btn_{hz_opt}",
                    use_container_width=True,
                ):
                    st.session_state.horizon = hz_opt
                    st.rerun()

    # ════════════════════════════════════════════════════
    # ZONE DETAIL COLUMN
    # ════════════════════════════════════════════════════
    with col_detail:
        # Zone dropdown selector
        zone_opts = list(ZONE_MAP.keys())
        cur_z_idx = zone_opts.index(sel_zone) if sel_zone in zone_opts else 13
        st.markdown('<div class="sec-label" style="margin:0 0 0.2rem 0;">Selected Zone</div>', unsafe_allow_html=True)
        chosen_z = st.selectbox(
            "Select Zone",
            options=zone_opts,
            format_func=lambda z: f"Zone {z}: {ZONE_MAP[z]}",
            index=cur_z_idx,
            key="zone_dropdown",
            label_visibility="collapsed",
        )
        if chosen_z != sel_zone:
            st.session_state.selected_zone = chosen_z
            st.rerun()

        zm_sel  = zone_meta_map.get(sel_zone, {})
        zlabel  = ZONE_MAP.get(sel_zone, f"Zone {sel_zone}")
        dom_risk = get_dominant_risk_for_zone(data, sel_zone)
        dom_color = RISK_LEVEL_COLOR.get(dom_risk, "#6b7280")
        dom_bg    = RISK_LEVEL_BG.get(dom_risk, "transparent")

        # Zone header
        st.markdown(f"""
<div class="up-card" style="padding:0.8rem 0.9rem;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:#e5e7eb;">{zlabel}</div>
      <div style="font-size:0.65rem;color:#4b5563;margin-top:2px;">Zone {sel_zone}</div>
    </div>
    <div>
      {risk_badge(dom_risk)}
    </div>
  </div>
  <div style="margin-top:0.5rem;font-size:0.68rem;color:#4b5563;">
    Last updated: {format_ist_timestamp(meta.get('timestamp_utc_parsed'))}
  </div>
</div>
""", unsafe_allow_html=True)

        # Current Environmental Conditions (measured AQ vs reanalysis weather kept distinct)
        st.markdown('<div class="sec-label" style="margin:0.2rem 0 0.3rem 0;">Current Conditions</div>', unsafe_allow_html=True)
        cur_cards = ""
        for icon, label, var in (("🍃", "AQI", "AQI"), ("🌧️", "Rain", "rainfall"), ("🌡️", "Temp", "temperature")):
            v_str, u_str, _, note = _var_pred_display(data, sel_zone, "current", var)
            cur_cards += f"""
  <div class="env-card">
    <div class="env-icon">{icon}</div>
    <div class="env-val">{v_str if v_str is not None else "N/A"}</div>
    <div class="env-lbl">{label}{f" {u_str}" if u_str and v_str is not None else ""}</div>
    <div style="font-size:0.55rem;color:#4b5563;margin-top:2px;">{note}</div>
  </div>"""
        st.markdown(f'<div class="env-cards">{cur_cards}</div>', unsafe_allow_html=True)
        src_bits = []
        if zm_sel.get("aqi_station"):
            src_bits.append(f"AQI: {zm_sel['aqi_station']} via OpenAQ, "
                            f"{format_ist_timestamp(_parse_ts(zm_sel.get('aqi_observed_at_utc')))}")
        for sec in zm_sel.get("secondary_stations", []):
            if sec.get("aqi") is not None:
                src_bits.append(f"Also in zone: {sec['station_name']} AQI {sec['aqi']:.0f}")
        if zm_sel.get("weather_valid_at_utc"):
            shared_w = zm_sel.get("weather_shared_with_zones") or []
            src_bits.append(f"Weather: Open-Meteo reanalysis, {format_ist_timestamp(_parse_ts(zm_sel['weather_valid_at_utc']))}"
                            + (f" (same grid cell as zone{'s' if len(shared_w) > 1 else ''} {', '.join(map(str, shared_w))})" if shared_w else ""))
        if src_bits:
            st.markdown('<div style="font-size:0.6rem;color:#4b5563;margin:0.2rem 0 0.3rem 0;">' +
                        "<br>".join(src_bits) + '</div>', unsafe_allow_html=True)

        # Prediction Horizon Dropdown
        st.markdown('<div class="sec-label" style="margin:0.4rem 0 0.2rem 0;">Prediction Horizon</div>', unsafe_allow_html=True)
        cur_hz_idx = HORIZON_OPTIONS.index(sel_horizon) if sel_horizon in HORIZON_OPTIONS else 0
        chosen_hz = st.selectbox(
            "Select Prediction Horizon",
            options=HORIZON_OPTIONS,
            index=cur_hz_idx,
            key="horizon_dropdown",
            label_visibility="collapsed",
        )
        if chosen_hz != sel_horizon:
            st.session_state.horizon = chosen_hz
            st.rerun()

        hz_display = "Current Environmental Conditions" if hz_key == "current" else f"Prediction for Selected Time: {sel_horizon}"
        st.markdown(
            f'<div class="sec-label" style="margin:0.3rem 0 0.2rem 0;">{hz_display}</div>',
            unsafe_allow_html=True,
        )

        aqi_vals  = _var_pred_display(data, sel_zone, hz_key, "AQI")
        rf_vals   = _var_pred_display(data, sel_zone, hz_key, "rainfall")
        temp_vals = _var_pred_display(data, sel_zone, hz_key, "temperature")

        aqi_card  = pred_card_html("🍃", "Predicted AQI",      aqi_vals[0],  aqi_vals[1],  aqi_vals[2],  aqi_vals[3])
        rf_card   = pred_card_html("🌧️", "Predicted Rainfall", rf_vals[0],   rf_vals[1],   rf_vals[2],   rf_vals[3])
        temp_card = pred_card_html("🌡️", "Predicted Temp",     temp_vals[0], temp_vals[1], temp_vals[2], temp_vals[3])

        st.markdown(f'<div class="pred-cards">{aqi_card}{rf_card}{temp_card}</div>', unsafe_allow_html=True)


        # Notices for what is not supported (never hidden behind a zero)
        notices = []
        if hz_key in ("1h", "3h", "6h"):
            notices.append("Temperature: only a next-day daily-mean model exists — switch to <b>24 Hours</b>.")
        if hz_key != "current" and not zm_sel.get("rainfall_forecast_enabled", False):
            notices.append("Rainfall: no rainfall forecast (models are gauge-trained and no validated live "
                           "input exists). Current rainfall is reanalysis — see <b>Current</b>.")
        if hz_key != "current" and zone_aqi_basis(data, sel_zone) == "MODEL_ESTIMATE":
            notices.append("AQI: MRF spatial estimate from neighbouring zones — not a measurement.")
        for n in notices:
            st.markdown(f'<div class="unavail-notice" style="margin-top:0.25rem;">{n}</div>', unsafe_allow_html=True)

        # Key Insights
        st.markdown('<div class="sec-label" style="margin:0.5rem 0 0.3rem 0;">Key Insights</div>', unsafe_allow_html=True)
        insights = generate_zone_insights(data, sel_zone)
        for ins in insights:
            st.markdown(f'<div class="insight-item">&#8227; {ins}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # ALERT CENTER COLUMN
    # ════════════════════════════════════════════════════
    with col_alerts:
        n_act = len(active_alerts)
        st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
  <div style="font-size:0.85rem;font-weight:700;color:#e5e7eb;">Alert Center</div>
  <div style="background:rgba(249,115,22,0.15);border:1px solid rgba(249,115,22,0.35);
       border-radius:20px;padding:1px 9px;font-size:0.65rem;font-weight:700;color:#fb923c;">
    {n_act} Active
  </div>
</div>
""", unsafe_allow_html=True)

        if n_act == 0:
            st.markdown("""
<div class="up-card-sm" style="text-align:center;padding:1.5rem 0.8rem;">
  <div style="font-size:1.4rem;margin-bottom:0.4rem;">&#9989;</div>
  <div style="font-size:0.75rem;color:#4b5563;">No WATCH/WARNING among evaluated forecasts.<br>Zones shown grey were not evaluated.</div>
</div>
""", unsafe_allow_html=True)
        else:
            # Sort: Critical > High > Moderate, then by horizon
            sev_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
            hz_order  = {"1h": 0, "3h": 1, "6h": 2, "24h": 3}

            def alert_sort_key(a):
                s = a.get("severity", "MODERATE").upper()
                h = a.get("horizon", "24h")
                return (sev_order.get(s, 9), hz_order.get(h, 9))

            sorted_alerts = sorted(active_alerts, key=alert_sort_key)

            var_icons = {"AQI": "🍃", "rainfall": "🌧️", "temperature": "🌡️"}
            var_labels = {"AQI": "Air Quality", "rainfall": "Rainfall", "temperature": "Temperature"}
            sev_color_map = {"HIGH": "#f97316", "MODERATE": "#f59e0b", "CRITICAL": "#ef4444"}
            sev_border = {"HIGH": "alert-card-high", "MODERATE": "alert-card-moderate", "CRITICAL": "alert-card-critical"}

            html_parts = ['<div class="alert-scroll">']
            for a in sorted_alerts:
                sev   = a.get("severity", "MODERATE").upper()
                var   = a.get("variable", "AQI")
                zone  = a.get("zone_label", "")
                zid   = a.get("zone_id", "")
                hz    = a.get("horizon", "")
                msg   = a.get("message") or f"{var} alert in {zone}."
                pred  = a.get("predicted_value")
                unit  = a.get("unit", "")
                exc   = a.get("exceedance_prob")

                col_a = sev_color_map.get(sev, "#f59e0b")
                bc    = sev_border.get(sev, "alert-card-moderate")
                icon  = var_icons.get(var, "🍃")
                vlbl  = var_labels.get(var, var)

                pred_str = f"{pred:.1f} {unit}" if pred is not None else ""
                prob_str = f"P={exc:.0%}" if exc is not None else ""

                ts_utc = a.get("timestamp_utc", "")
                try:
                    ts_fmt = format_ist_timestamp(pd.Timestamp(ts_utc, tz="UTC") if ts_utc else None)
                except Exception:
                    ts_fmt = ts_utc

                html_parts.append(f"""
<div class="alert-card {bc}">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem;">
    <div style="font-size:0.75rem;font-weight:700;color:#e5e7eb;">{icon} {vlbl} Alert</div>
    <span class="risk-badge" style="background:{RISK_LEVEL_BG.get(sev.capitalize(),'rgba(249,115,22,0.12)')};
          color:{col_a};border:1px solid {col_a}44;font-size:0.58rem;">{sev}</span>
  </div>
  <div class="zone-chip">Zone {zid}: {zone} &nbsp;&#183;&nbsp; {hz}</div>
  <div style="font-size:0.72rem;color:#9ca3af;line-height:1.4;margin:0.25rem 0;">{msg}</div>
  {f'<div style="font-size:0.65rem;color:#6b7280;">{pred_str} &nbsp; {prob_str}</div>' if pred_str or prob_str else ''}
  <div style="font-size:0.6rem;color:#374151;margin-top:0.3rem;">{ts_fmt}</div>
</div>""")

            html_parts.append("</div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # BOTTOM OVERVIEW STRIP
    # ════════════════════════════════════════════════════
    avg_aqi   = summary_stats.get("avg_aqi")
    high_zones = summary_stats.get("high_risk_zones", 0)
    rf_zones   = summary_stats.get("rainfall_risk_zones", 0)
    est_zones  = summary_stats.get("aqi_estimate_zones", 0)
    unav_zones = summary_stats.get("unavailable_zones", 0)
    mon_aqi    = summary_stats.get("monitored_aqi", 0)
    mins       = get_countdown_minutes(data)

    aqi_color = "#22c55e"
    if avg_aqi:
        if avg_aqi > 200: aqi_color = "#f97316"
        elif avg_aqi > 100: aqi_color = "#f59e0b"

    countdown_disp = f"{mins // 60:d}h {mins % 60:02d}m" if mins is not None else "--"

    st.markdown(f"""
<div class="overview-strip">
  <div class="ov-card">
    <div class="ov-val" style="color:{aqi_color};">{f"{avg_aqi:.0f}" if avg_aqi else "--"}</div>
    <div class="ov-lbl">Avg measured AQI<br><span style="color:#374151;">{mon_aqi} zones with current station data</span></div>
  </div>
  <div class="ov-card">
    <div class="ov-val" style="color:{'#f97316' if high_zones > 0 else '#22c55e'};">{high_zones}/15</div>
    <div class="ov-lbl">Zones with High<br>Risk</div>
  </div>
  <div class="ov-card">
    <div class="ov-val" style="color:{'#60a5fa' if rf_zones > 0 else '#22c55e'};">{rf_zones}/15</div>
    <div class="ov-lbl">Zones with heavy-rain<br>risk {"" if meta.get("rainfall_forecast_enabled") else "(forecast disabled)"}</div>
  </div>
  <div class="ov-card">
    <div class="ov-val" style="color:#9ca3af;">{unav_zones}/15</div>
    <div class="ov-lbl">Zones without an<br>evaluated forecast</div>
  </div>
  <div class="ov-card">
    <div class="ov-val" style="color:#60a5fa;">{n_act}</div>
    <div class="ov-lbl">Active<br>Alerts</div>
  </div>
  <div class="ov-card">
    <div class="ov-val" style="font-size:1.2rem;color:#60a5fa;">{countdown_disp}</div>
    <div class="ov-lbl">Next Update In<br><span style="color:#374151;">~{mins if mins is not None else '--'} min</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE: ALERT CENTER (full-page)
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Alert Center":

    all_alerts = get_alerts(data, active_only=False)
    active     = [a for a in all_alerts if a.get("alert_status") in ("WATCH", "WARNING")]

    st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">Alert Center</div>
    <div class="dash-sub">Live alerts from HMM+MRF pipeline &nbsp;|&nbsp;
    {format_ist_timestamp(meta.get('timestamp_utc_parsed'))}</div>
  </div>
  <div style="font-size:0.7rem;color:#f97316;font-weight:700;">
    {len(active)} Active Alert{'s' if len(active) != 1 else ''}
  </div>
</div>
""", unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        flt_var = st.multiselect("Variable", ["AQI", "rainfall", "temperature"],
                                  default=["AQI", "rainfall", "temperature"], key="ac_var")
    with fc2:
        flt_sev = st.multiselect("Severity", ["HIGH", "MODERATE", "LOW", "UNAVAILABLE", "UNMONITORED"],
                                  default=["HIGH", "MODERATE"], key="ac_sev")
    with fc3:
        flt_hz  = st.multiselect("Horizon", ["1h","3h","6h","24h"],
                                  default=["1h","3h","6h","24h"], key="ac_hz")

    filtered = [
        a for a in all_alerts
        if a.get("variable") in flt_var
        and a.get("severity","").upper() in [s.upper() for s in flt_sev]
        and a.get("horizon") in flt_hz
    ]
    st.markdown(f"**{len(filtered)}** records match filters")

    if not filtered:
        st.info("No alerts match the selected filters.")
    else:
        rows = []
        for a in filtered:
            pred = a.get("predicted_value")
            exc  = a.get("exceedance_prob")
            rows.append({
                "Zone ID":     a.get("zone_id"),
                "Zone":        a.get("zone_label"),
                "Variable":    a.get("variable"),
                "Horizon":     a.get("horizon"),
                "Predicted":   f"{pred:.2f}" if pred is not None else "N/A",
                "Unit":        a.get("unit",""),
                "P(Exceed)":   f"{exc:.1%}" if exc is not None else "N/A",
                "Threshold":   a.get("threshold"),
                "Status":      a.get("alert_status"),
                "Severity":    a.get("severity"),
                "Basis":       a.get("basis") or "",
                "Target time": a.get("target_time_utc") or "",
                "Message":     a.get("message",""),
                "Timestamp":   a.get("timestamp_utc",""),
            })
        df_disp = pd.DataFrame(rows)
        st.dataframe(df_disp, use_container_width=True, height=500)

# ══════════════════════════════════════════════════════════════════════════
# PAGE: DATA & COVERAGE (what is supported, from which source)
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Data & Coverage":
    st.markdown("""
<div class="dash-header"><div>
  <div class="dash-title">Data &amp; Coverage</div>
  <div class="dash-sub">What UrbanPulse can and cannot report right now, and where each number comes from</div>
</div></div>""", unsafe_allow_html=True)

    sup = meta.get("supported_horizons", {})
    zm_all = get_zone_meta_map(data)
    n_meas = sum(1 for z in zm_all.values() if z.get("aqi_state") == "MEASURED")
    n_fc = summary_stats.get("aqi_forecast_zones", 0)
    n_est = summary_stats.get("aqi_estimate_zones", 0)
    rain_active = meta.get("rainfall_active_zones", 0)
    temp_ok = sum(1 for r in data["predictions"].get("temperature", {}).get("24h", []) if r.get("availability") != "UNAVAILABLE")
    st.markdown("#### Supported live conditions")
    st.dataframe(pd.DataFrame([
        {"Variable": "Air quality (CPCB NAQI method)", "Source": "OpenAQ — CPCB/TNPCB CAAQMS stations",
         "Provenance": "Measured", "Current value": f"{n_meas}/15 zones",
         "Forecast horizons": ", ".join(sup.get("AQI", [])) or "none",
         "Forecast zones": f"{n_fc} HMM+MRF, {n_est} MRF estimate"},
        {"Variable": "Rainfall", "Source": "Open-Meteo Historical Weather API",
         "Provenance": "Reanalysis / model analysis", "Current value": "15/15 zones (8 distinct grid cells)",
         "Forecast horizons": ", ".join(sup.get("rainfall", [])) or "none (disabled)",
         "Forecast zones": f"{rain_active}" if meta.get("rainfall_forecast_enabled") else "0 — gauge-trained models, input not validated"},
        {"Variable": "Temperature", "Source": "Open-Meteo Historical Weather API",
         "Provenance": "Reanalysis / model analysis", "Current value": "15/15 zones (8 distinct grid cells)",
         "Forecast horizons": "next-day daily mean" if sup.get("temperature") else "none",
         "Forecast zones": f"{temp_ok}"},
    ]), use_container_width=True, hide_index=True)

    st.markdown("#### Source status")
    rows = []
    for name, sinfo in get_source_status(data).items():
        rows.append({"Source": name, "Last fetch OK": sinfo.get("ok"),
                     "Latest data (IST)": format_ist_timestamp(_parse_ts(sinfo.get("latest_record_utc"))),
                     "Data age (h)": sinfo.get("data_age_hours"), "Stale": sinfo.get("stale"),
                     "Last successful fetch (IST)": format_ist_timestamp(_parse_ts(sinfo.get("last_success_utc"))),
                     "Error": sinfo.get("error") or ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Zone availability")
    zrows = []
    for z in range(1, 16):
        zm = zm_all.get(z, {})
        aq1 = get_predictions_for_zone(data, z, "AQI", "1h") or {}
        r24 = get_predictions_for_zone(data, z, "rainfall", "24h") or {}
        zrows.append({
            "Zone": f"{z} {ZONE_MAP[z]}",
            "AQ station": zm.get("aqi_station") or "—",
            "AQI (latest measured)": (f"{zm['cpcb_aqi_computed']:.0f}" if zm.get("cpcb_aqi_computed") is not None
                                      else str(zm.get("aqi_state") or "—").replace("_", " ").lower()),
            "AQI forecast": {"MODEL_FORECAST": "HMM+MRF", "MODEL_ESTIMATE": "MRF estimate"}.get(aq1.get("availability"),
                             STATUS_EXPLANATION.get(aq1.get("alert_status"), "unavailable")),
            "Rain now (mm/h, reanalysis)": zm.get("rainfall_mm"),
            "Rain last 24 h (mm, reanalysis)": zm.get("precip_24h_mm"),
            "Rain forecast": "HMM (reanalysis-driven)" if r24.get("availability") == "MODEL_FORECAST" else
                             ("disabled" if r24.get("alert_status") == "DISABLED_INPUT_NOT_VALIDATED" else "no model"),
            "Weather grid shared with": ", ".join(map(str, zm.get("weather_shared_with_zones") or [])) or "—",
        })
    st.dataframe(pd.DataFrame(zrows), use_container_width=True, hide_index=True)

    st.markdown("#### Limitations")
    for lim in meta.get("limitations", []):
        st.markdown(f"- {lim}")

# ── Auto-refresh every 60s to pick up new pipeline output ─────────────────────
st.html("""
<script>
setTimeout(function() {
    window.location.reload();
}, 60000);
</script>
""")

