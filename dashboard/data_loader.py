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

PREDICTIONS_PATH = os.path.abspath(
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
}

RISK_LEVEL_COLOR = {
    "Low":         "#22c55e",   # green
    "Moderate":    "#f59e0b",   # amber
    "High":        "#f97316",   # orange
    "Critical":    "#ef4444",   # red
    "Unmonitored": "#6b7280",   # gray
    "Unavailable": "#374151",   # dark gray
}

RISK_LEVEL_BG = {
    "Low":         "rgba(34,197,94,0.12)",
    "Moderate":    "rgba(245,158,11,0.14)",
    "High":        "rgba(249,115,22,0.15)",
    "Critical":    "rgba(239,68,68,0.15)",
    "Unmonitored": "rgba(107,114,128,0.12)",
    "Unavailable": "rgba(55,65,81,0.12)",
}


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


def get_dominant_risk_for_zone(data: Dict, zone_id: int) -> str:
    """
    Zone colour = worst EVALUATED alert across AQI (all horizons) and rainfall (24 h accumulation).
    If no variable was evaluated for the zone, the result is "Unavailable" — a zone without data is
    never shown as low risk.
    """
    priority = {"High": 3, "Moderate": 2, "Low": 1}
    worst, worst_p = "Unavailable", 0
    checks = [("AQI", h) for h in data.get("predictions", {}).get("AQI", {})] + [("rainfall", "24h")]
    for var, hz in checks:
        zp = get_predictions_for_zone(data, zone_id, var, hz)
        if zp and zp.get("alert_status") in EVALUATED_STATUSES:
            lvl = get_risk_level(zp["alert_status"])
            if priority[lvl] > worst_p:
                worst, worst_p = lvl, priority[lvl]
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


def get_zone_risk_map(data: Dict) -> Dict[int, str]:
    """Return {zone_id: risk_level} for all 15 zones."""
    return {z: get_dominant_risk_for_zone(data, z) for z in range(1, 16)}


def get_summary_stats(data: Dict) -> Dict[str, Any]:
    """Compute Chennai-wide summary statistics from live predictions."""
    zone_risks = get_zone_risk_map(data)
    high_or_crit  = sum(1 for l in zone_risks.values() if l in ("High", "Critical"))
    moderate_plus = sum(1 for l in zone_risks.values() if l in ("High", "Critical", "Moderate"))

    # Average AQI across monitored zones (current observation)
    aqi_vals = []
    for zm in data.get("zone_meta", []):
        v = zm.get("cpcb_aqi_computed")
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            aqi_vals.append(float(v))
    avg_aqi = round(sum(aqi_vals) / len(aqi_vals), 1) if aqi_vals else None

    # Zones with rainfall risk (WATCH or WARNING at 24h)
    rf_risk = 0
    for zid in range(1, 16):
        zp = get_predictions_for_zone(data, zid, "rainfall", "24h")
        if zp and zp.get("alert_status") in ("WATCH", "WARNING"):
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
        "total_zones":       15,
    }


def get_alerts(data: Dict, active_only: bool = True) -> List[Dict]:
    """Return alert records, optionally filtered to WATCH/WARNING only."""
    if active_only and "active_alerts" in data:
        return data["active_alerts"]
    all_alerts = data.get("alerts", [])
    if active_only:
        return [a for a in all_alerts if a.get("alert_status") in ("WATCH", "WARNING")]
    return all_alerts



def format_ist_timestamp(ts_parsed) -> str:
    """Format a parsed timestamp in IST."""
    if ts_parsed is None:
        return "N/A"
    try:
        ist = ts_parsed.tz_convert("Asia/Kolkata")
        return ist.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return str(ts_parsed)


def get_countdown_minutes(data: Dict) -> Optional[int]:
    """Minutes until next pipeline update, or None."""
    meta = get_meta(data)
    nxt = meta.get("next_update_utc_parsed")
    if nxt is None:
        return None
    now = pd.Timestamp.now(tz="UTC")
    delta = (nxt - now).total_seconds()
    return max(0, int(delta // 60))


def generate_zone_insights(data: Dict, zone_id: int) -> List[str]:
    """
    Generate human-readable Key Insights for a zone from actual prediction data.
    Returns an empty list if no meaningful insight exists.
    """
    insights = []
    zm = get_zone_meta_map(data).get(zone_id, {})

    # AQI trend: current vs 24h
    cur_aqi = zm.get("cpcb_aqi_computed")
    zp_aqi_1h = get_predictions_for_zone(data, zone_id, "AQI", "1h")
    zp_aqi_24 = get_predictions_for_zone(data, zone_id, "AQI", "24h")

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
    active_alerts = get_alerts(data, active_only=True)
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

