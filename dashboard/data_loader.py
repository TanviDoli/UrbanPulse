"""
UrbanPulse Dashboard — Data Loader
====================================
Loads and parses data/live/current_predictions.json produced by the Phase 2
live pipeline. The dashboard ONLY reads this file — it never calls WeatherAPI,
runs HMM/MRF models, or generates its own predictions.

Provides clean typed accessors for:
  - Pipeline metadata (timestamps, refresh interval, zone counts)
  - Per-zone current observations (AQI, rainfall, temperature)
  - Per-variable per-horizon per-zone predictions
  - Active alert records
  - Summary statistics
"""

import json
import os
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src import freshness as fr

# URBANPULSE_PREDICTIONS_PATH lets the dashboard read a separately written pipeline output
# (e.g. a demo run with --output) without touching data/live/current_predictions.json.
PREDICTIONS_PATH = os.path.abspath(os.environ.get("URBANPULSE_PREDICTIONS_PATH") or
    os.path.join(os.path.dirname(__file__), "..", "data", "live", "current_predictions.json")
)

ZONE_MAP = {
    1: "Thiruvottiyur", 2: "Manali", 3: "Madhavaram", 4: "Tondiarpet",
    5: "Royapuram", 6: "Thiru-Vi-Ka Nagar", 7: "Ambattur", 8: "Anna Nagar",
    9: "Teynampet", 10: "Kodambakkam", 11: "Valasaravakkam", 12: "Alandur",
    13: "Adyar", 14: "Perungudi", 15: "Sholinganallur"
}

ADJACENCY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "zone_adjacency.csv")
)

_ADJ_CACHE: Optional[Dict[int, List[int]]] = None

def get_adjacent_zones(zone_id: int) -> List[int]:
    """Return list of adjacent zone IDs for a given zone from zone_adjacency.csv."""
    global _ADJ_CACHE
    if _ADJ_CACHE is None:
        _ADJ_CACHE = {}
        if os.path.exists(ADJACENCY_PATH):
            try:
                df = pd.read_csv(ADJACENCY_PATH)
                for _, row in df[df["adjacent"] == 1].iterrows():
                    z1, z2 = int(row["zone_id_1"]), int(row["zone_id_2"])
                    _ADJ_CACHE.setdefault(z1, []).append(z2)
                    _ADJ_CACHE.setdefault(z2, []).append(z1)
            except Exception:
                pass
    return _ADJ_CACHE.get(zone_id, [])


# Backend risk state → dashboard display
RISK_STATE_TO_LEVEL = {
    "WARNING": "High",
    "WATCH":   "Moderate",
    "NORMAL":  "Low",
    "UNMONITORED_INSUFFICIENT_DATA":   "Unmonitored",
    "UNAVAILABLE_MISSING_CLIMATOLOGY": "Unavailable",
    "UNAVAILABLE_NO_DATA":             "Unavailable",
    "UNAVAILABLE_STALE_INPUT":         "Unavailable",
    "UNAVAILABLE_NO_MEASUREMENT":      "Unavailable",
    "NOT_SUPPORTED_NO_MODEL":          "Unavailable",
    "NOT_EVALUATED_MODEL_IS_DAILY_MEAN": "Unavailable",
    "NOT_APPLICABLE_HOURLY":           "Unavailable",
    "DISABLED_INPUT_NOT_VALIDATED":    "Unavailable",
    "NOT_SUPPORTED_FAILED_EVALUATION": "Unavailable",
}

# Statuses that come from an actual probabilistic evaluation (everything else = not evaluated)
EVALUATED_STATUSES = ("NORMAL", "WATCH", "WARNING")

STATUS_EXPLANATION = {
    "UNAVAILABLE_NO_DATA": "No input data",
    "UNAVAILABLE_STALE_INPUT": "Station data too old to forecast from",
    "UNAVAILABLE_NO_MEASUREMENT": "No live station measurement",
    "NOT_SUPPORTED_NO_MODEL": "No trained model for this zone",
    "NOT_EVALUATED_MODEL_IS_DAILY_MEAN": "Heatwave not evaluated (model forecasts daily mean, not Tmax)",
    "NOT_APPLICABLE_HOURLY": "Heavy-rain alert uses 24 h accumulation only",
    "UNMONITORED_INSUFFICIENT_DATA": "Unmonitored",
    "DISABLED_INPUT_NOT_VALIDATED": "Rainfall forecast disabled (no validated live input; see Data & Coverage)",
    "NOT_SUPPORTED_FAILED_EVALUATION": "Not available: this horizon did not beat a simple baseline in evaluation",
    "NOT_EVALUATED_HEATWAVE_CRITERION": "Heatwave not evaluated (needs Tmax and a verified climatological normal)",
}

RISK_LEVEL_COLOR = {
    "Low":         "#22c55e",   # green
    "Moderate":    "#f59e0b",   # amber
    "High":        "#f97316",   # orange
    "Critical":    "#ef4444",   # red
    "Unmonitored": "#6b7280",   # gray
    "Unavailable": "#374151",   # dark gray
    "Estimate":    "#64748b",   # slate — MRF spatial estimate, no elevated risk (uncertain)
    "Stale":       "#78716c",   # stone — forecast from stale input / target passed
    "Stale warning": "#a16207", # dark amber — old WATCH/WARNING kept for review, not active
}

RISK_LEVEL_BG = {
    "Low":         "rgba(34,197,94,0.12)",
    "Moderate":    "rgba(245,158,11,0.14)",
    "High":        "rgba(249,115,22,0.15)",
    "Critical":    "rgba(239,68,68,0.15)",
    "Unmonitored": "rgba(107,114,128,0.12)",
    "Unavailable": "rgba(55,65,81,0.12)",
    "Estimate":    "rgba(100,116,139,0.14)",
    "Stale":       "rgba(120,113,108,0.14)",
    "Stale warning": "rgba(161,98,7,0.16)",
}

FORECAST_FRESHNESS_LABEL = {
    fr.CURRENT: "forecast",
    fr.STALE_INPUT: "STALE INPUT",
    fr.EXPIRED: "EXPIRED",
    fr.UNKNOWN: "time unknown",
}


def _now(now=None) -> pd.Timestamp:
    return fr.to_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")


def load_predictions() -> Optional[Dict[str, Any]]:
    """
    Load current_predictions.json. Returns None if file is missing or corrupt.
    Never raises — dashboard must handle None gracefully.
    """
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    try:
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_meta(data: Dict) -> Dict[str, Any]:
    """Return pipeline metadata with parsed timestamps."""
    meta = data.get("meta", {})
    result = dict(meta)

    # Parse timestamps
    for key in ("timestamp_utc", "generated_at_utc", "next_update_utc"):
        raw = meta.get(key)
        if raw:
            try:
                ts = pd.Timestamp(raw)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                result[key + "_parsed"] = ts
            except Exception:
                result[key + "_parsed"] = None
    return result


def get_zone_meta_map(data: Dict) -> Dict[int, Dict]:
    """Return zone_meta as {zone_id: dict}."""
    return {zm["zone_id"]: zm for zm in data.get("zone_meta", [])}


def get_predictions_for_zone(
    data: Dict,
    zone_id: int,
    variable: str,   # "AQI" | "rainfall" | "temperature"
    horizon: str,    # "1h" | "3h" | "6h" | "24h"
) -> Optional[Dict]:
    """
    Return the prediction dict for a specific zone/variable/horizon.
    Returns None if not found.
    """
    try:
        for zp in data["predictions"][variable][horizon]:
            if zp["zone_id"] == zone_id:
                return zp
    except (KeyError, TypeError):
        pass
    return None


def get_all_zones_snapshot(
    data: Dict,
    variable: str = "AQI",
    horizon: str = "1h",
) -> List[Dict]:
    """
    Return a flat list of prediction dicts for all 15 zones at a given variable/horizon,
    enriched with zone_meta fields.
    """
    zone_meta_map = get_zone_meta_map(data)
    try:
        preds = data["predictions"][variable][horizon]
    except KeyError:
        return []

    enriched = []
    for zp in preds:
        zid = zp["zone_id"]
        zm = zone_meta_map.get(zid, {})
        row = {**zp, **{
            "zone_label":       ZONE_MAP.get(zid, f"Zone {zid}"),
            "cpcb_aqi_current": zm.get("cpcb_aqi_computed"),
            "rainfall_current": zm.get("rainfall_mm"),
            "temp_current":     zm.get("temp_c_current"),
            "condition_text":   zm.get("condition_text"),
            "humidity_pct":     zm.get("humidity_pct"),
            "pm2_5":            zm.get("pm2_5_ugm3"),
            "pm10":             zm.get("pm10_ugm3"),
            "no2":              zm.get("no2_ugm3"),
        }}
        enriched.append(row)
    return enriched


def get_risk_level(alert_status: str) -> str:
    """Map backend alert_status to dashboard risk level. Unknown/missing -> Unavailable (never Low)."""
    return RISK_STATE_TO_LEVEL.get(alert_status, "Unavailable")


def _aqi_origin(data: Dict) -> Optional[str]:
    meta = data.get("meta") or {}
    return (meta.get("data_times") or {}).get("aqi_forecast_origin_utc") or meta.get("timestamp_utc")


def _forecast_input_time(data: Dict, zone_id: int, variable: str, availability: Optional[str]):
    if variable == "AQI":
        return fr.aqi_forecast_input_time(zone_id, availability, data.get("aqi_zone_inputs"), _aqi_origin(data))
    if variable == "rainfall":
        return ((data.get("meta") or {}).get("data_times") or {}).get("rainfall_forecast_origin_utc")
    if variable == "temperature":
        return ((data.get("meta") or {}).get("data_times") or {}).get("temperature_forecast_origin_utc")
    return None


def forecast_view(data: Dict, zone_id: int, variable: str, horizon: str, now=None) -> Optional[Dict]:
    """
    Prediction record plus freshness evaluated at VIEW time (src/freshness.py R2) and the
    display risk (R3): keys forecast_freshness, input_observed_at_utc, input_age_hours,
    level, evaluated, stale_warning, is_estimate. Model values are passed through unchanged.
    """
    zp = get_predictions_for_zone(data, zone_id, variable, horizon)
    if zp is None:
        return None
    t_now = _now(now)
    src_time = _forecast_input_time(data, zone_id, variable, zp.get("availability"))
    f = fr.forecast_freshness(zp.get("target_time_utc"), src_time, t_now)
    return {**zp, "forecast_freshness": f, "input_observed_at_utc": src_time,
            "input_age_hours": fr.age_hours(src_time, t_now),
            **fr.display_risk(zp.get("alert_status"), zp.get("availability"), f)}


def aqi_observation(data: Dict, zone_id: int, now=None) -> Dict[str, Any]:
    """
    Latest AQI observation for a zone with rule R1 re-applied at view time.
    state: MEASURED (current) | MEASURED_STALE | NO_LIVE_DATA | NO_STATION.
    A stale reading keeps its value, timestamp and age, but is never labelled current.
    """
    zm = get_zone_meta_map(data).get(zone_id, {})
    t = zm.get("aqi_observed_at_utc")
    value = zm.get("last_measured_aqi")
    if value is None:
        value = zm.get("cpcb_aqi_computed")
    state = zm.get("aqi_state") or "NO_LIVE_DATA"
    if t is not None and value is not None:
        state = "MEASURED" if fr.is_observation_current(t, _now(now)) else "MEASURED_STALE"
    return {"state": state, "value": value if state in ("MEASURED", "MEASURED_STALE") else None,
            "observed_at_utc": t, "age_hours": fr.age_hours(t, _now(now)), "station": zm.get("aqi_station")}


# Map colour priority: current evidence first; stale/estimated/missing data is never green.
_LEVEL_PRIORITY = {"High": 7, "Moderate": 6, "Low": 5, fr.LEVEL_STALE_WARNING: 4,
                   fr.LEVEL_ESTIMATE: 3, fr.LEVEL_STALE: 2, fr.LEVEL_UNAVAILABLE: 1}


def get_dominant_risk_for_zone(data: Dict, zone_id: int, now=None) -> str:
    """
    Zone colour across AQI (all horizons) and rainfall (24 h accumulation), using view-time
    freshness. Low (green) requires a CURRENT forecast from measured input; spatial estimates,
    stale/expired forecasts and missing data are never shown as low risk.
    """
    worst, worst_p = "Unavailable", 0
    checks = [("AQI", h) for h in data.get("predictions", {}).get("AQI", {})] + [("rainfall", "24h")]
    for var, hz in checks:
        v = forecast_view(data, zone_id, var, hz, now)
        p = _LEVEL_PRIORITY.get(v["level"], 0) if v else 0
        if p > worst_p:
            worst, worst_p = v["level"], p
    return worst


def zone_aqi_basis(data: Dict, zone_id: int) -> Optional[str]:
    """'MODEL_FORECAST' (measured input), 'MODEL_ESTIMATE' (MRF only) or None (unavailable)."""
    for hz in data.get("predictions", {}).get("AQI", {}):
        zp = get_predictions_for_zone(data, zone_id, "AQI", hz)
        if zp:
            a = zp.get("availability")
            return None if a in (None, "UNAVAILABLE") else a
    return None


def get_source_status(data: Dict) -> Dict[str, Dict[str, Any]]:
    """Per-source collection status recorded by the pipeline (age, stale flag, errors)."""
    return data.get("meta", {}).get("sources", {}) or {}


def data_is_stale(data: Dict) -> bool:
    return any(s.get("stale") for s in get_source_status(data).values())


def get_zone_risk_map(data: Dict, now=None) -> Dict[int, str]:
    """Return {zone_id: risk_level} for all 15 zones."""
    return {z: get_dominant_risk_for_zone(data, z, now) for z in range(1, 16)}


def evaluation_breakdown(data: Dict, now=None) -> Dict[str, int]:
    """
    Zones by whether a risk could be evaluated now:
      evaluated           Low/Moderate/High from a CURRENT forecast (incl. estimates showing WATCH/WARNING)
      estimate_uncertain  CURRENT MRF spatial estimate with no elevated risk (uncertain, not 'safe')
      stale               only stale-input / expired forecasts (not evaluated)
      unavailable         no forecast at all (not evaluated)
    """
    out = {"evaluated": 0, "estimate_uncertain": 0, "stale": 0, "unavailable": 0, "total": 15}
    for lvl in get_zone_risk_map(data, now).values():
        if lvl in ("High", "Moderate", "Low"):
            out["evaluated"] += 1
        elif lvl == fr.LEVEL_ESTIMATE:
            out["estimate_uncertain"] += 1
        elif lvl in (fr.LEVEL_STALE, fr.LEVEL_STALE_WARNING):
            out["stale"] += 1
        else:
            out["unavailable"] += 1
    return out


def get_summary_stats(data: Dict, now=None) -> Dict[str, Any]:
    """Compute Chennai-wide summary statistics from live predictions (view-time freshness)."""
    zone_risks = get_zone_risk_map(data, now)
    high_or_crit  = sum(1 for l in zone_risks.values() if l in ("High", "Critical"))
    moderate_plus = sum(1 for l in zone_risks.values() if l in ("High", "Critical", "Moderate"))

    # Average AQI across zones whose observation is CURRENT (stale readings are not averaged in)
    aqi_vals, n_stale_obs = [], 0
    for z in range(1, 16):
        obs = aqi_observation(data, z, now)
        if obs["state"] == "MEASURED":
            aqi_vals.append(float(obs["value"]))
        elif obs["state"] == "MEASURED_STALE":
            n_stale_obs += 1
    avg_aqi = round(sum(aqi_vals) / len(aqi_vals), 1) if aqi_vals else None

    # Zones with an evaluated (CURRENT) rainfall WATCH or WARNING at 24h
    rf_risk = 0
    for zid in range(1, 16):
        v = forecast_view(data, zid, "rainfall", "24h", now)
        if v and v["evaluated"] and v.get("alert_status") in ("WATCH", "WARNING"):
            rf_risk += 1

    # Zones with an evaluated AQI forecast / estimate (not unavailable)
    aqi_forecast_zones = sum(1 for z in range(1, 16) if zone_aqi_basis(data, z) == "MODEL_FORECAST")
    aqi_estimate_zones = sum(1 for z in range(1, 16) if zone_aqi_basis(data, z) == "MODEL_ESTIMATE")
    unavailable_zones = sum(1 for l in zone_risks.values() if l == "Unavailable")

    return {
        "avg_aqi":           avg_aqi,
        "high_risk_zones":   high_or_crit,
        "moderate_plus_zones": moderate_plus,
        "rainfall_risk_zones": rf_risk,
        "aqi_forecast_zones": aqi_forecast_zones,
        "aqi_estimate_zones": aqi_estimate_zones,
        "unavailable_zones": unavailable_zones,
        "monitored_aqi":     len(aqi_vals),
        "stale_aqi":         n_stale_obs,
        "evaluation":        evaluation_breakdown(data, now),
        "total_zones":       15,
    }


def get_alerts(data: Dict, active_only: bool = True, now=None) -> List[Dict]:
    """
    Alert records annotated with view-time forecast_freshness. With active_only, only
    WATCH/WARNING from CURRENT forecasts are returned — the file's precomputed active_alerts
    is not used because the file ages after it is written.
    """
    t_now = _now(now)
    records = data.get("alerts") or data.get("active_alerts") or []
    out = []
    for a in records:
        src = _forecast_input_time(data, a.get("zone_id"), a.get("variable"), a.get("availability"))
        out.append({**a, "forecast_freshness": fr.forecast_freshness(a.get("target_time_utc"), src, t_now)})
    if active_only:
        return [a for a in out if a.get("alert_status") in ("WATCH", "WARNING")
                and a["forecast_freshness"] == fr.CURRENT]
    return out


def get_stale_warnings(data: Dict, now=None) -> List[Dict]:
    """WATCH/WARNING from stale-input or expired forecasts: kept for review, never active alerts."""
    return [a for a in get_alerts(data, active_only=False, now=now)
            if a.get("alert_status") in ("WATCH", "WARNING") and a["forecast_freshness"] != fr.CURRENT]


def get_pipeline_status(data: Dict, now=None) -> Dict[str, Any]:
    """ON_SCHEDULE / OVERDUE / NO_SCHEDULE with minutes, per src/freshness.py R4."""
    return fr.pipeline_status(data.get("meta", {}), _now(now))



def format_ist_timestamp(ts_parsed) -> str:
    """Format a parsed timestamp in IST."""
    if ts_parsed is None:
        return "N/A"
    try:
        ist = ts_parsed.tz_convert("Asia/Kolkata")
        return ist.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return str(ts_parsed)


def get_countdown_minutes(data: Dict, now=None) -> Optional[int]:
    """Minutes until the next scheduled update; None when overdue or nothing is scheduled
    (use get_pipeline_status to tell those apart — overdue is never shown as 0 min)."""
    return get_pipeline_status(data, now)["minutes_to_next"]


def format_age(hours: Optional[float]) -> str:
    """'45 min', '5.2 h', '1 d 22 h' — for observation and output ages."""
    if hours is None:
        return "unknown age"
    if hours < 1:
        return f"{max(0, int(round(hours * 60)))} min"
    if hours < 24:
        return f"{hours:.1f} h"
    return f"{int(hours // 24)} d {int(hours % 24)} h"


def generate_zone_insights(data: Dict, zone_id: int, now=None) -> List[str]:
    """
    Generate human-readable Key Insights for a zone from actual prediction data.
    Returns an empty list if no meaningful insight exists.
    """
    insights = []
    zm = get_zone_meta_map(data).get(zone_id, {})

    # AQI trend: current observation vs a CURRENT 24h forecast only (never from stale data)
    obs = aqi_observation(data, zone_id, now)
    cur_aqi = obs["value"] if obs["state"] == "MEASURED" else None
    zp_aqi_1h = forecast_view(data, zone_id, "AQI", "1h", now)
    zp_aqi_24 = forecast_view(data, zone_id, "AQI", "24h", now)
    if obs["state"] == "MEASURED_STALE":
        insights.append(f"Latest AQI reading is {format_age(obs['age_hours'])} old (stale); "
                        f"forecasts made from it are not evaluated.")
    if zp_aqi_24 and zp_aqi_24["forecast_freshness"] != fr.CURRENT:
        zp_aqi_24 = None
    if zp_aqi_1h and not zp_aqi_1h["evaluated"]:
        zp_aqi_1h = None

    if cur_aqi and not math.isnan(float(cur_aqi)):
        if zp_aqi_24 and zp_aqi_24.get("mrf_pred"):
            diff = float(zp_aqi_24["mrf_pred"]) - float(cur_aqi)
            if abs(diff) > 5:
                direction = "increase" if diff > 0 else "decrease"
                insights.append(
                    f"AQI expected to {direction} from {float(cur_aqi):.0f} to "
                    f"{float(zp_aqi_24['mrf_pred']):.0f} over 24 hours."
                )
        if zp_aqi_1h and zp_aqi_1h.get("alert_status") in ("WATCH", "WARNING"):
            insights.append(
                f"Elevated AQI risk in 1 hour (P = {zp_aqi_1h['exceedance_prob']:.0%})."
            )

    # Rainfall
    rf_cur = zm.get("rainfall_mm")
    is_rf_mon = zm.get("is_rainfall_monitored", False)
    for hz in ("1h", "3h", "6h"):
        zp_rf = get_predictions_for_zone(data, zone_id, "rainfall", hz)
        if zp_rf and zp_rf.get("alert_status") in ("WATCH", "WARNING"):
            pred = zp_rf.get("mrf_pred")
            pred_str = f" ({pred:.1f} mm)" if pred else ""
            sev = "high chance of heavy rainfall" if zp_rf["alert_status"] == "WARNING" else "elevated rainfall risk"
            insights.append(f"{sev.capitalize()}{pred_str} in {hz.replace('h', ' hour')}.")
            break  # show only the soonest

    # Rainfall 24h accumulation
    zp_rf24 = get_predictions_for_zone(data, zone_id, "rainfall", "24h")
    if zp_rf24 and zp_rf24.get("accum_24h_mm") and is_rf_mon:
        acc = float(zp_rf24["accum_24h_mm"])
        if acc > 10:
            insights.append(f"Predicted 24-hour rainfall accumulation: {acc:.1f} mm.")

    # Temperature
    zp_temp = get_predictions_for_zone(data, zone_id, "temperature", "24h")
    if zp_temp and zp_temp.get("next_day_tmean_pred_c") is not None and zp_temp.get("last_day_tmean_c") is not None:
        diff = float(zp_temp["next_day_tmean_pred_c"]) - float(zp_temp["last_day_tmean_c"])
        if abs(diff) >= 1.0:
            insights.append(
                f"Daily mean temperature forecast {float(zp_temp['next_day_tmean_pred_c']):.1f}°C for "
                f"{zp_temp.get('target_day_ist')} ({'+' if diff > 0 else ''}{diff:.1f}°C vs previous day; reanalysis-driven).")
    if zone_aqi_basis(data, zone_id) == "MODEL_ESTIMATE":
        insights.append("No usable air-quality monitor in this zone: AQI shown is an MRF spatial estimate from neighbouring zones.")

    # Check neighboring zones for active alerts
    neighbors = get_adjacent_zones(zone_id)
    active_alerts = get_alerts(data, active_only=True, now=now)
    for a in active_alerts:
        nid = a.get("zone_id")
        if nid in neighbors:
            n_name = ZONE_MAP.get(nid, f"Zone {nid}")
            var_name = a.get("variable", "environmental")
            status = a.get("alert_status", "WATCH")
            insights.append(
                f"Nearby zone {n_name} (Zone {nid}) is also under {status} for {var_name}."
            )
            break

    if not insights:
        insights.append("No significant environmental change predicted for this zone.")
    return insights

