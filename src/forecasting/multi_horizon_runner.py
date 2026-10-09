"""
Multi-Horizon UrbanPulse Forecast Runner.

Runs the EXISTING FROZEN HMM + MRF pipeline at all supported prediction horizons
for all 15 Chennai GCC zones, given current preprocessed observations.

SUPPORTED HORIZONS BY VARIABLE:
  AQI:         1h, 3h, 6h, 24h  (fully supported — hourly HMM)
  Rainfall:    1h, 3h, 6h, 24h  (supported for monitored zones only)
  Temperature: 24h only          (daily-regime HMM; sub-daily is NOT legitimate)

TEMPERATURE SUB-DAILY LIMITATION (documented):
  The existing temperature HMM was trained on DAILY Tmax values (IMD historical record
  ending 2024-06-22). Each "horizon=N" step corresponds to N *days* ahead, not N hours.
  The AT (ambient temperature) sensor data from CAAQMS stations was checked and found
  to contain valid readings in only 1 out of 9 stations (Gandhi Nagar only, ~3,500 rows).
  This is insufficient to train a zone-specific sub-daily temperature HMM.
  Therefore:
    - Horizons 1h/3h/6h for temperature: NOT produced by the HMM.
      The current observation (temp_c from WeatherAPI) is passed through as the
      "current" reading. No HMM-derived sub-daily temperature prediction is fabricated.
    - Horizon 24h for temperature: Uses the existing daily-regime HMM (horizon=1 day).
      Labelled as "next_day_tmax_forecast" to be unambiguous.

NO FABRICATION RULES:
  - Unmonitored zones for AQI: active_mask=False → HMM skipped → MRF infers spatially.
  - Unmonitored zones for rainfall (11 zones): status=UNMONITORED_INSUFFICIENT_DATA.
    For rainfall, MRF beta=0.0 (no spatial coupling), so unmonitored zones receive NaN.
  - Temperature for all zones uses the same frozen daily HMM.

EXISTING ARTIFACTS LOADED (frozen, never retrained):
  models/hmm/hmm_aqi_zone_{z}.joblib         for z in {2,4,5,10,12,13,14}
  models/hmm/hmm_rainfall_zone_{z}.joblib    for z in {4,5,10,14}
  models/hmm/hmm_temperature_zone_{z}.joblib for z in 1..15
  models/hmm_mrf/mrf_aqi.joblib
  models/hmm_mrf/mrf_rainfall.joblib
  models/hmm_mrf/mrf_temperature.joblib
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.hmm_model import TemporalHMM
from src.models.mrf_model import SpatialMRF
from src.alerts.alert_engine import (
    evaluate_aqi_alert,
    evaluate_rainfall_alert_24h,
    evaluate_temperature_heatwave_alert,
    compute_24h_rainfall_accumulation,
)
from src.live.obs_preprocessor import RAINFALL_MONITORED_ZONES, AQI_MONITORED_ZONES

# ── Horizon definitions ────────────────────────────────────────────────────────
AQI_HORIZONS         = [1, 3, 6, 24]   # hours
RAINFALL_HORIZONS    = [1, 3, 6, 24]   # hours
# Temperature: only 24h is legitimate (daily regime HMM)
TEMPERATURE_HORIZONS = [24]            # "24h" = next-day Tmax

_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


def _load_hmm_models() -> Dict[str, Dict[int, TemporalHMM]]:
    """Load frozen zone-specific HMMs from disk."""
    models_dir = os.path.abspath(_MODELS_DIR)
    hmm = {"AQI": {}, "rainfall": {}, "temperature": {}}

    for z in [2, 4, 5, 10, 12, 13, 14]:
        p = os.path.join(models_dir, "hmm", f"hmm_aqi_zone_{z}.joblib")
        if os.path.exists(p):
            hmm["AQI"][z] = joblib.load(p)

    for z in [4, 5, 10, 14]:
        p = os.path.join(models_dir, "hmm", f"hmm_rainfall_zone_{z}.joblib")
        if os.path.exists(p):
            hmm["rainfall"][z] = joblib.load(p)

    for z in range(1, 16):
        p = os.path.join(models_dir, "hmm", f"hmm_temperature_zone_{z}.joblib")
        if os.path.exists(p):
            hmm["temperature"][z] = joblib.load(p)

    return hmm


def _load_mrf_models() -> Dict[str, SpatialMRF]:
    """Load frozen MRF models from disk."""
    models_dir = os.path.abspath(_MODELS_DIR)
    mrf = {}
    for var in ["aqi", "rainfall", "temperature"]:
        p = os.path.join(models_dir, "hmm_mrf", f"mrf_{var}.joblib")
        if os.path.exists(p):
            mrf_key = "AQI" if var == "aqi" else var
            mrf[mrf_key] = joblib.load(p)
    return mrf


class MultiHorizonForecaster:
    """
    Orchestrates multi-horizon predictions for all 15 GCC zones using existing
    frozen HMM + MRF artifacts.

    Loaded once at construction; no retraining ever occurs.
    """

    def __init__(self):
        print("[MultiHorizonForecaster] Loading frozen HMM models...", flush=True)
        self.hmm = _load_hmm_models()
        print(
            f"  AQI models:         {len(self.hmm['AQI'])} zones",
            f"\n  Rainfall models:    {len(self.hmm['rainfall'])} zones",
            f"\n  Temperature models: {len(self.hmm['temperature'])} zones",
        )
        print("[MultiHorizonForecaster] Loading frozen MRF models...", flush=True)
        self.mrf = _load_mrf_models()
        print(f"  MRF variables loaded: {list(self.mrf.keys())}")

    def _run_aqi_horizons(
        self, y_obs: np.ndarray, active_mask: np.ndarray
    ) -> Dict[str, Any]:
        """Run AQI HMM + MRF for all supported horizons."""
        results = {}
        for h in AQI_HORIZONS:
            y_hat    = np.full(15, np.nan)
            variances = np.full(15, np.nan)

            for zone_id in range(1, 16):
                idx = zone_id - 1
                if active_mask[idx] and zone_id in self.hmm["AQI"]:
                    obs_val = float(y_obs[idx])
                    if not np.isnan(obs_val):
                        mod = self.hmm["AQI"][zone_id]
                        mu, var, _ = mod.forecast(np.array([obs_val]), horizon=h)
                        y_hat[idx] = mu
                        variances[idx] = var

            # MRF spatial refinement
            mrf_mod = self.mrf.get("AQI")
            if mrf_mod is not None:
                mu_mrf, var_mrf = mrf_mod.infer(y_hat, variances, active_mask)
            else:
                mu_mrf, var_mrf = y_hat.copy(), variances.copy()

            # Alert evaluation
            zone_preds = []
            for zone_id in range(1, 16):
                idx = zone_id - 1
                mu_z   = float(mu_mrf[idx])
                var_z  = float(var_mrf[idx])
                alert  = evaluate_aqi_alert(mu_z, var_z)
                zone_preds.append({
                    "zone_id": zone_id,
                    "hmm_pred":  round(float(y_hat[idx]), 2) if not np.isnan(y_hat[idx]) else None,
                    "mrf_pred":  round(mu_z, 2) if not np.isnan(mu_z) else None,
                    "mrf_var":   round(var_z, 4) if not np.isnan(var_z) else None,
                    "alert_status": alert["alert_status"],
                    "exceedance_prob": alert["probability_crossing"],
                    "threshold": alert["threshold"],
                    "is_monitored": zone_id in AQI_MONITORED_ZONES,
                })
            results[f"{h}h"] = zone_preds
        return results

    def _run_rainfall_horizons(
        self, y_obs: np.ndarray, active_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run Rainfall HMM + MRF for all supported horizons.
        Unmonitored zones always return UNMONITORED_INSUFFICIENT_DATA.
        MRF beta=0.0 for rainfall → no spatial coupling → unmonitored zones = NaN.
        """
        results = {}
        # Pre-compute gamma_t (filtered state distribution) for monitored zones
        # This is needed for the exact 24h accumulation computation.
        gamma_cache: Dict[int, np.ndarray] = {}
        for zone_id in RAINFALL_MONITORED_ZONES:
            idx = zone_id - 1
            if active_mask[idx] and zone_id in self.hmm["rainfall"]:
                obs_val = float(y_obs[idx])
                if not np.isnan(obs_val):
                    mod = self.hmm["rainfall"][zone_id]
                    gamma_cache[zone_id] = mod.get_filtered_state_distribution(
                        np.array([obs_val])
                    )

        for h in RAINFALL_HORIZONS:
            y_hat     = np.full(15, np.nan)
            variances  = np.full(15, np.nan)

            for zone_id in range(1, 16):
                idx = zone_id - 1
                if active_mask[idx] and zone_id in self.hmm["rainfall"]:
                    obs_val = float(y_obs[idx])
                    if not np.isnan(obs_val):
                        mod = self.hmm["rainfall"][zone_id]
                        mu, var, _ = mod.forecast(np.array([obs_val]), horizon=h)
                        y_hat[idx] = mu
                        variances[idx] = var

            # MRF (beta=0 → effectively identity for monitored, NaN for unmonitored)
            mrf_mod = self.mrf.get("rainfall")
            if mrf_mod is not None:
                mu_mrf, var_mrf = mrf_mod.infer(y_hat, variances, active_mask)
            else:
                mu_mrf, var_mrf = y_hat.copy(), variances.copy()

            zone_preds = []
            for zone_id in range(1, 16):
                idx = zone_id - 1
                is_mon = zone_id in RAINFALL_MONITORED_ZONES
                mu_z  = float(mu_mrf[idx])
                var_z = float(var_mrf[idx])

                # 24h accumulation alert (only at 24h horizon for monitored zones)
                if h == 24 and is_mon and zone_id in gamma_cache:
                    mod = self.hmm["rainfall"][zone_id]
                    mu_acc, var_acc = compute_24h_rainfall_accumulation(
                        gamma_cache[zone_id],
                        mod.transmat_,
                        mod.means_,
                        mod.variances_,
                    )
                    alert = evaluate_rainfall_alert_24h(mu_acc, var_acc, is_monitored=True)
                    accum_pred = round(mu_acc, 2)
                else:
                    alert = evaluate_rainfall_alert_24h(np.nan, np.nan, is_monitored=is_mon)
                    accum_pred = None

                zone_preds.append({
                    "zone_id": zone_id,
                    "hmm_pred":    round(float(y_hat[idx]), 4) if not np.isnan(y_hat[idx]) else None,
                    "mrf_pred":    round(mu_z, 4) if not np.isnan(mu_z) else None,
                    "mrf_var":     round(var_z, 6) if not np.isnan(var_z) else None,
                    "accum_24h_mm": accum_pred,
                    "alert_status": alert["alert_status"],
                    "exceedance_prob": alert["probability_crossing"],
                    "threshold": alert["threshold"],
                    "is_monitored": is_mon,
                })
            results[f"{h}h"] = zone_preds
        return results

    def _run_temperature(
        self, y_obs: np.ndarray, active_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run Temperature HMM for the 24h (next-day Tmax) horizon only.
        Sub-daily horizons are NOT produced — see module docstring.

        Returns results keyed as '24h'.
        """
        results = {}
        y_hat    = np.full(15, np.nan)
        variances = np.full(15, np.nan)

        for zone_id in range(1, 16):
            idx = zone_id - 1
            if active_mask[idx] and zone_id in self.hmm["temperature"]:
                obs_val = float(y_obs[idx])
                if not np.isnan(obs_val):
                    mod = self.hmm["temperature"][zone_id]
                    # horizon=1 in the daily HMM means 1 day ahead
                    mu, var, _ = mod.forecast(np.array([obs_val]), horizon=1)
                    y_hat[idx] = mu
                    variances[idx] = var

        # MRF (beta=0 for temperature → identical to HMM output)
        mrf_mod = self.mrf.get("temperature")
        if mrf_mod is not None:
            mu_mrf, var_mrf = mrf_mod.infer(y_hat, variances, active_mask)
        else:
            mu_mrf, var_mrf = y_hat.copy(), variances.copy()

        zone_preds = []
        for zone_id in range(1, 16):
            idx = zone_id - 1
            mu_z  = float(mu_mrf[idx])
            var_z = float(var_mrf[idx])
            # No verified climatological normal → heatwave alert unavailable
            alert = evaluate_temperature_heatwave_alert(mu_z, var_z, normal_tmax=None)

            zone_preds.append({
                "zone_id": zone_id,
                "current_temp_c":       round(float(y_obs[idx]), 2) if not np.isnan(y_obs[idx]) else None,
                "next_day_tmax_pred_c": round(mu_z, 2) if not np.isnan(mu_z) else None,
                "pred_var":             round(var_z, 4) if not np.isnan(var_z) else None,
                "alert_status": alert["alert_status"],
                "exceedance_prob": alert["probability_crossing"],
                "note": (
                    "Temperature prediction is next-day Tmax from daily HMM. "
                    "Sub-daily (1h/3h/6h) temperature predictions are NOT produced: "
                    "the HMM is a daily-regime model. No fabrication."
                ),
            })
        results["24h"] = zone_preds
        return results

    def run(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all variable/horizon combinations given preprocessed observations.

        Args:
            preprocessed: output of obs_preprocessor.preprocess_weatherapi_observations()

        Returns:
            Structured prediction dict ready for dashboard / alert engine consumption.
        """
        ts = preprocessed["timestamp_utc"]

        aqi_data  = preprocessed["aqi"]
        rf_data   = preprocessed["rainfall"]
        temp_data = preprocessed["temperature"]

        aqi_results  = self._run_aqi_horizons(aqi_data["y_obs"],  aqi_data["active_mask"])
        rf_results   = self._run_rainfall_horizons(rf_data["y_obs"], rf_data["active_mask"])
        temp_results = self._run_temperature(temp_data["y_obs"], temp_data["active_mask"])

        return {
            "timestamp_utc": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "supported_horizons": {
                "AQI":         list(aqi_results.keys()),
                "rainfall":    list(rf_results.keys()),
                "temperature": list(temp_results.keys()),
            },
            "predictions": {
                "AQI":         aqi_results,
                "rainfall":    rf_results,
                "temperature": temp_results,
            },
            "zone_meta":   preprocessed["zone_meta"],
        }
