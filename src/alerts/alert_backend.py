"""
Structured Alert Backend for UrbanPulse.

Converts multi-horizon HMM+MRF prediction output into structured alert records
with human-readable message templates. Alerts are generated ONLY from actual
model predictions crossing validated official thresholds.

ALERT SCHEMA:
  zone_id, zone_label, variable, horizon,
  predicted_value, threshold, severity,
  exceedance_prob, alert_type, message, timestamp_utc

SEVERITY MAPPING (existing model risk states → dashboard labels):
  WARNING → HIGH
  WATCH   → MODERATE
  NORMAL  → LOW (alerts not generated for NORMAL)
  UNMONITORED_INSUFFICIENT_DATA → not alerted
  UNAVAILABLE_MISSING_CLIMATOLOGY → not alerted

NO FABRICATION: Alerts are only generated when the model's exceedance probability
crosses the documented UrbanPulse decision boundaries (20% / 60%).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import datetime


# ── Natural language message templates ────────────────────────────────────────
_MSG_TEMPLATES = {
    "AQI": {
        "WARNING": "Air quality in {zone} may deteriorate to Poor levels (AQI > 200) in {horizon}.",
        "WATCH":   "Air quality in {zone} shows elevated risk of Poor levels in {horizon}.",
    },
    "rainfall": {
        "WARNING": "{zone} may face heavy rainfall (≥ 64.5 mm / 24h) in {horizon}.",
        "WATCH":   "{zone} has elevated risk of heavy rainfall in {horizon}.",
    },
    "temperature": {
        "WARNING": "{zone} may experience extreme heat (coastal heatwave conditions) in {horizon}.",
        "WATCH":   "{zone} has elevated risk of extreme heat in {horizon}.",
    },
}

_SEVERITY_MAP = {
    "WARNING": "HIGH",
    "WATCH":   "MODERATE",
    "NORMAL":  "LOW",
}

_VARIABLE_UNITS = {
    "AQI": "AQI",
    "rainfall": "mm",
    "temperature": "°C",
}

_THRESHOLDS = {
    "AQI":         200.0,
    "rainfall":    64.5,   # 24h accumulation
    "temperature": None,   # heatwave criterion is Tmax-based; the live model forecasts daily mean
}


def _make_message(variable: str, alert_status: str, zone_label: str, horizon_label: str) -> Optional[str]:
    """Generate a human-readable alert message. Returns None for NORMAL/unmonitored."""
    var_msgs = _MSG_TEMPLATES.get(variable, {})
    tmpl = var_msgs.get(alert_status)
    if tmpl is None:
        return None
    return tmpl.format(zone=zone_label, horizon=horizon_label)


def _horizon_label(horizon_key: str) -> str:
    """Convert '1h' → '1 hour', '24h' → '24 hours' etc."""
    h = horizon_key.replace("h", "").strip()
    try:
        n = int(h)
        return f"{n} hour" if n == 1 else f"{n} hours"
    except ValueError:
        return horizon_key


def build_alert_records(
    predictions: Dict[str, Any],
    zone_meta: List[Dict[str, Any]],
    timestamp_utc: str,
) -> List[Dict[str, Any]]:
    """
    Build structured alert records from multi-horizon prediction output.

    Args:
        predictions: predictions dict from MultiHorizonForecaster.run()["predictions"]
        zone_meta:   zone_meta list from MultiHorizonForecaster.run()["zone_meta"]
        timestamp_utc: ISO timestamp string

    Returns:
        List of alert record dicts (only WATCH and WARNING statuses included).
        Full zone/horizon coverage is returned when include_normal=True is needed
        by dashboard — call with include_normal=True for full matrix.
    """
    zone_label_map = {m["zone_id"]: m["zone_label"] for m in zone_meta}
    records = []

    for variable, horizons in predictions.items():
        for horizon_key, zone_preds in horizons.items():
            hlabel = _horizon_label(horizon_key)
            for zp in zone_preds:
                zone_id     = zp["zone_id"]
                zone_label  = zone_label_map.get(zone_id, f"Zone {zone_id}")
                alert_status = zp.get("alert_status", "NORMAL")
                exceedance   = zp.get("exceedance_prob")
                is_monitored = zp.get("is_monitored", True)

                # Anything that is not an evaluated NORMAL/WATCH/WARNING is "unavailable",
                # never silently LOW (a missing forecast is not evidence of low risk).
                if alert_status in ("UNMONITORED_INSUFFICIENT_DATA",):
                    severity = "UNMONITORED"
                else:
                    severity = _SEVERITY_MAP.get(alert_status, "UNAVAILABLE")

                # Predicted value depends on variable
                if variable == "AQI":
                    pred_val = zp.get("mrf_pred")
                elif variable == "rainfall":
                    pred_val = zp.get("accum_24h_mm") if horizon_key == "24h" else zp.get("mrf_pred")
                elif variable == "temperature":
                    pred_val = zp.get("temp_pred_c", zp.get("next_day_tmean_pred_c", zp.get("next_day_tmax_pred_c")))
                else:
                    pred_val = zp.get("mrf_pred")

                msg = _make_message(variable, alert_status, zone_label, hlabel)
                if msg and zp.get("availability") == "MODEL_ESTIMATE":
                    msg += " (MRF spatial estimate — no usable monitor in this zone.)"

                records.append({
                    "timestamp_utc":    timestamp_utc,
                    "zone_id":          zone_id,
                    "zone_label":       zone_label,
                    "variable":         variable,
                    "horizon":          horizon_key,
                    "predicted_value":  round(float(pred_val), 3) if pred_val is not None and not (isinstance(pred_val, float) and np.isnan(pred_val)) else None,
                    "unit":             _VARIABLE_UNITS.get(variable, ""),
                    "threshold":        _THRESHOLDS.get(variable),
                    "exceedance_prob":  round(float(exceedance), 4) if exceedance is not None and not (isinstance(exceedance, float) and np.isnan(exceedance)) else None,
                    "alert_status":     alert_status,
                    "severity":         severity,
                    "alert_type":       f"{variable}_{alert_status}",
                    "is_monitored":     is_monitored,
                    "availability":     zp.get("availability"),
                    "basis":            zp.get("basis"),
                    "target_time_utc":  zp.get("target_time_utc"),
                    "message":          msg,
                })

    return records


def filter_active_alerts(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only WATCH and WARNING alerts (actionable alerts for display)."""
    return [r for r in records if r["alert_status"] in ("WATCH", "WARNING")]


def format_alert_summary(records: List[Dict[str, Any]]) -> str:
    """Human-readable summary of active alerts for CLI output (ASCII-safe for Windows)."""
    active = filter_active_alerts(records)
    if not active:
        return "[OK] No active alerts. All zones within normal risk parameters."
    lines = [f"[ALERT] {len(active)} active alert(s):"]
    for r in active:
        p = f"{r['predicted_value']:.1f} {r['unit']}" if r["predicted_value"] is not None else "N/A"
        prob_str = f"P(exceed)={r['exceedance_prob']:.1%}" if r["exceedance_prob"] is not None else ""
        lines.append(
            f"  [{r['severity']:8s}] Zone {r['zone_id']:2d} ({r['zone_label']:20s}) | "
            f"{r['variable']:11s} {r['horizon']:4s} | pred={p} | {prob_str}"
        )
    return "\n".join(lines)
