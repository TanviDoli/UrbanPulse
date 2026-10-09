"""
Live Observation Preprocessor for UrbanPulse.

Converts raw WeatherAPI current-observation dicts into the input format
expected by the existing UrbanPulse HMM and CPCB AQI pipeline.

DESIGN:
- AQI: Uses the existing src.preprocessing.cpcb_aqi sub-index methodology.
  WeatherAPI provides individual pollutant concentrations in µg/m³.
  CO is provided in µg/m³ by WeatherAPI; CPCB breakpoints use mg/m³ so conversion
  is applied (÷ 1000) before sub-index calculation.
- Rainfall: WeatherAPI precip_mm is current-hour accumulation. Used directly as
  hourly rainfall input to the HMM.
- Temperature: WeatherAPI temp_c is instantaneous ambient temperature.
  Used ONLY as the current observation. Sub-daily temperature HMM predictions
  are NOT generated (see temperature_note below).

NO FABRICATION RULES:
- If a required field is None/NaN, the zone is marked active_mask=False for that variable.
- Regional means are never substituted.
- WeatherAPI AQI indices are stored separately as metadata and never presented as CPCB AQI.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.cpcb_aqi import calculate_sub_index, compute_cpcb_aqi_row

# Monitored AQI zones (have trained HMM models)
AQI_MONITORED_ZONES = {2, 4, 5, 10, 12, 13, 14}
# Rainfall monitored zones (have trained HMM models)
RAINFALL_MONITORED_ZONES = {4, 5, 10, 14}
# ALL zones have temperature HMM (daily regime)
TEMPERATURE_ALL_ZONES = set(range(1, 16))

# CO unit note: WeatherAPI reports CO in µg/m³; CPCB sub-index breakpoints are in mg/m³
# Conversion factor: 1 mg/m³ = 1000 µg/m³
CO_UGPM3_TO_MGPM3 = 1.0 / 1000.0


def compute_cpcb_aqi_from_weatherapi(obs: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], int]:
    """
    Compute CPCB AQI from WeatherAPI raw pollutant concentrations.

    WeatherAPI provides instantaneous concentrations, not 24h/8h rolling averages.
    This is noted as a methodological limitation: CPCB standard requires rolling windows.
    We use the instantaneous values as best-available current observations, consistent
    with the live CPCB pipeline which also computes running means from real-time feeds.

    CO: converted from µg/m³ → mg/m³ before CPCB breakpoint lookup.
    Returns: (cpcb_aqi, dominant_pollutant, valid_subindex_count)
    """
    sub_dict = {}

    # PM2.5 (µg/m³ — same unit as CPCB)
    pm25 = obs.get("pm2_5_ugm3")
    if pm25 is not None and not np.isnan(pm25):
        si = calculate_sub_index("PM2.5", float(pm25))
        if si is not None and not np.isnan(si):
            sub_dict["PM2.5"] = si

    # PM10 (µg/m³)
    pm10 = obs.get("pm10_ugm3")
    if pm10 is not None and not np.isnan(pm10):
        si = calculate_sub_index("PM10", float(pm10))
        if si is not None and not np.isnan(si):
            sub_dict["PM10"] = si

    # NO2 (µg/m³)
    no2 = obs.get("no2_ugm3")
    if no2 is not None and not np.isnan(no2):
        si = calculate_sub_index("NO2", float(no2))
        if si is not None and not np.isnan(si):
            sub_dict["NO2"] = si

    # SO2 (µg/m³)
    so2 = obs.get("so2_ugm3")
    if so2 is not None and not np.isnan(so2):
        si = calculate_sub_index("SO2", float(so2))
        if si is not None and not np.isnan(si):
            sub_dict["SO2"] = si

    # O3 (µg/m³ — stored as "Ozone" in CPCB pipeline)
    o3 = obs.get("o3_ugm3")
    if o3 is not None and not np.isnan(o3):
        si = calculate_sub_index("Ozone", float(o3))
        if si is not None and not np.isnan(si):
            sub_dict["Ozone"] = si

    # CO: WeatherAPI µg/m³ → mg/m³
    co_ugm3 = obs.get("co_ugm3")
    if co_ugm3 is not None and not np.isnan(co_ugm3):
        co_mgm3 = float(co_ugm3) * CO_UGPM3_TO_MGPM3
        si = calculate_sub_index("CO", co_mgm3)
        if si is not None and not np.isnan(si):
            sub_dict["CO"] = si

    return compute_cpcb_aqi_row(sub_dict)


def preprocess_weatherapi_observations(
    raw_obs: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert WeatherAPI zone observations into UrbanPulse model input vectors.

    Returns:
        {
          'timestamp_utc': pd.Timestamp,
          'aqi': {
              'y_obs':       np.ndarray[15],  # observed CPCB AQI (NaN if unavailable)
              'active_mask': np.ndarray[15],  # True for zones with valid obs + HMM model
          },
          'rainfall': {
              'y_obs':       np.ndarray[15],
              'active_mask': np.ndarray[15],
          },
          'temperature': {
              'y_obs':       np.ndarray[15],  # instantaneous temp_c (NaN if unavailable)
              'active_mask': np.ndarray[15],
          },
          'zone_meta': [{zone_id, zone_label, fetch_ok, aqi_value, dominant_pollutant,
                         cpcb_aqi_note, weatherapi_us_epa_index, ...}] x 15
        }
    """
    n = 15
    aqi_obs    = np.full(n, np.nan)
    aqi_mask   = np.zeros(n, dtype=bool)
    rf_obs     = np.full(n, np.nan)
    rf_mask    = np.zeros(n, dtype=bool)
    temp_obs   = np.full(n, np.nan)
    temp_mask  = np.zeros(n, dtype=bool)

    zone_meta = []
    timestamps = []

    for zone_id in range(1, 16):
        idx = zone_id - 1
        obs = raw_obs.get(zone_id, {})

        if obs.get("timestamp_utc") is not None:
            timestamps.append(obs["timestamp_utc"])

        # ── AQI ──────────────────────────────────────────────────────────────
        cpcb_aqi_val = np.nan
        dominant_p = None
        sub_count = 0
        cpcb_note = ""

        if obs.get("fetch_ok") and zone_id in AQI_MONITORED_ZONES:
            aqi_val, dominant_p, sub_count = compute_cpcb_aqi_from_weatherapi(obs)
            if aqi_val is not None and not np.isnan(aqi_val):
                cpcb_aqi_val = float(aqi_val)
                aqi_obs[idx] = cpcb_aqi_val
                aqi_mask[idx] = True
                cpcb_note = (
                    f"CPCB AQI from WeatherAPI instantaneous concentrations "
                    f"(sub-indices: {sub_count}). Note: instantaneous, not rolling-average."
                )
            else:
                cpcb_note = "Insufficient pollutant data for CPCB AQI computation."
        elif zone_id not in AQI_MONITORED_ZONES:
            cpcb_note = "Zone not in AQI-monitored set; will be inferred by MRF."
        else:
            cpcb_note = f"Fetch failed: {obs.get('error_msg', 'unknown error')}"

        # ── Rainfall ─────────────────────────────────────────────────────────
        rf_val = np.nan
        if obs.get("fetch_ok") and zone_id in RAINFALL_MONITORED_ZONES:
            precip = obs.get("precip_mm")
            if precip is not None and not np.isnan(precip):
                rf_val = max(0.0, float(precip))
                rf_obs[idx] = rf_val
                rf_mask[idx] = True

        # ── Temperature ───────────────────────────────────────────────────────
        # temp_c is instantaneous ambient temp — used as current observation.
        # The daily-regime HMM forecasts tomorrow's Tmax; sub-daily horizons
        # (1h/3h/6h) are NOT supported by the existing temperature HMM.
        temp_val = np.nan
        if obs.get("fetch_ok"):
            t = obs.get("temp_c")
            if t is not None and not np.isnan(t):
                temp_val = float(t)
                temp_obs[idx] = temp_val
                temp_mask[idx] = True  # All zones have a temperature HMM

        zone_meta.append({
            "zone_id":    zone_id,
            "zone_label": obs.get("zone_label", f"Zone {zone_id}"),
            "fetch_ok":   obs.get("fetch_ok", False),
            "error_msg":  obs.get("error_msg"),
            # AQI
            "cpcb_aqi_computed":            cpcb_aqi_val,
            "dominant_pollutant":           dominant_p,
            "cpcb_subindex_count":          sub_count,
            "cpcb_aqi_note":                cpcb_note,
            "weatherapi_us_epa_index":      obs.get("weatherapi_us_epa_index"),
            "weatherapi_gb_defra_index":    obs.get("weatherapi_gb_defra_index"),
            # Rainfall
            "rainfall_mm":      rf_val,
            "is_rainfall_monitored": zone_id in RAINFALL_MONITORED_ZONES,
            # Temperature
            "temp_c_current":   temp_val,
            # Raw pollutants (stored for traceability, not used directly by HMM)
            "co_ugm3":     obs.get("co_ugm3"),
            "no2_ugm3":    obs.get("no2_ugm3"),
            "o3_ugm3":     obs.get("o3_ugm3"),
            "so2_ugm3":    obs.get("so2_ugm3"),
            "pm2_5_ugm3":  obs.get("pm2_5_ugm3"),
            "pm10_ugm3":   obs.get("pm10_ugm3"),
            "humidity_pct":      obs.get("humidity_pct"),
            "condition_text":    obs.get("condition_text"),
        })

    # Representative timestamp: median of all successful fetches
    if timestamps:
        ts = sorted(timestamps)[len(timestamps) // 2]
    else:
        ts = pd.Timestamp.now(tz="UTC").floor("h")

    return {
        "timestamp_utc": ts,
        "aqi":         {"y_obs": aqi_obs,  "active_mask": aqi_mask},
        "rainfall":    {"y_obs": rf_obs,   "active_mask": rf_mask},
        "temperature": {"y_obs": temp_obs, "active_mask": temp_mask},
        "zone_meta":   zone_meta,
    }
