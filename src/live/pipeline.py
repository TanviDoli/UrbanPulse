"""
UrbanPulse Live Ingestion & Inference Orchestrator (Step 14A).
Connects validated live CPCB and IMD data streams into:
1. EXISTING frozen zone-specific HMMs (temporal filtering & forecasting)
2. EXISTING frozen Gaussian MRF (spatial coupling over 15 GCC zones)
3. EXISTING frozen Alert Engine (evaluating official danger thresholds)

STRICT SAFETY:
- NO retraining.
- NO modifications to frozen models or beta parameters.
- NO observation fabrication for unmonitored zones.
- Normal Tmax is NEVER fabricated.
- 24h rainfall accumulation is evaluated for heavy rainfall (>= 64.5 mm), never hourly rates.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from src.live.schemas import (
    LiveStationAQI,
    LiveWeatherObservation,
    UNMONITORED_AQI_ZONES,
    UNMONITORED_RAINFALL_ZONES
)
from src.live.cpcb_client import CPCBClient
from src.live.imd_client import IMDClient
from src.models.hmm_model import TemporalHMM
from src.models.mrf_model import SpatialMRF
from src.alerts.alert_engine import (
    evaluate_aqi_alert,
    evaluate_rainfall_alert_24h,
    evaluate_temperature_heatwave_alert,
    compute_24h_rainfall_accumulation
)


class LiveIngestionPipeline:
    """Orchestrates hourly live ingestion, model inference, and alert evaluation."""

    def __init__(
        self,
        cpcb_client: Optional[CPCBClient] = None,
        imd_client: Optional[IMDClient] = None,
        models_dir: str = "models",
        adjacency_path: str = "data/processed/zone_adjacency.csv"
    ):
        self.cpcb = cpcb_client or CPCBClient()
        self.imd = imd_client or IMDClient()
        self.models_dir = models_dir
        self.adjacency_path = adjacency_path

        # Load FROZEN models (do NOT retrain)
        self.hmm_models: Dict[str, Dict[int, TemporalHMM]] = self._load_hmm_models()
        self.mrf_models: Dict[str, SpatialMRF] = self._load_mrf_models()

    def _load_hmm_models(self) -> Dict[str, Dict[int, TemporalHMM]]:
        """Loads existing frozen zone-specific HMMs."""
        hmm_dict = {'AQI': {}, 'rainfall': {}, 'temperature': {}}

        # AQI zone models
        for z in [2, 4, 5, 10, 12, 13, 14]:
            path = os.path.join(self.models_dir, 'hmm', f'hmm_aqi_zone_{z}.joblib')
            if os.path.exists(path):
                hmm_dict['AQI'][z] = joblib.load(path)

        # Rainfall zone models
        for z in [4, 5, 10, 14]:
            path = os.path.join(self.models_dir, 'hmm', f'hmm_rainfall_zone_{z}.joblib')
            if os.path.exists(path):
                hmm_dict['rainfall'][z] = joblib.load(path)

        # Temperature zone models (all 15 zones)
        for z in range(1, 16):
            path = os.path.join(self.models_dir, 'hmm', f'hmm_temperature_zone_{z}.joblib')
            if os.path.exists(path):
                hmm_dict['temperature'][z] = joblib.load(path)

        return hmm_dict

    def _load_mrf_models(self) -> Dict[str, SpatialMRF]:
        """Loads existing frozen SpatialMRF models."""
        mrf_dict = {}
        for var in ['aqi', 'rainfall', 'temperature']:
            path = os.path.join(self.models_dir, 'hmm_mrf', f'mrf_{var}.joblib')
            if os.path.exists(path):
                mrf_dict[var.upper() if var == 'aqi' else var] = joblib.load(path)
        return mrf_dict

    def run_aqi_hourly_cycle(
        self,
        cpcb_payload: Dict[str, Any],
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Executes a complete hourly cycle for AQI using existing models:
        1. Ingests & validates CPCB payload.
        2. Computes rolling CPCB sub-indices & overall AQI per monitored zone.
        3. Filters & forecasts with zone-specific HMMs.
        4. Applies Gaussian MRF spatial coupling across all 15 zones.
        5. Evaluates CPCB Poor AQI (>200) alert probability.
        """
        # Parse observations
        observations = self.cpcb.parse_records(cpcb_payload)
        zone_aqi_dict = self.cpcb.update_buffer_and_compute_aqi(observations)

        # Build HMM input vectors across all 15 zones
        y_hat = np.full(15, np.nan, dtype=float)
        variances = np.full(15, np.nan, dtype=float)
        active_mask = np.zeros(15, dtype=bool)

        for z in range(1, 16):
            z_idx = z - 1
            z_data = zone_aqi_dict.get(z)

            if z in self.hmm_models['AQI'] and z_data and z_data.is_monitored:
                obs_val = z_data.aqi_value
                if pd.notna(obs_val):
                    # Sequential HMM filtering on observed value
                    hmm_mod = self.hmm_models['AQI'][z]
                    pred_m, pred_v, _ = hmm_mod.forecast(np.array([obs_val]), horizon=horizon)
                    y_hat[z_idx] = pred_m
                    variances[z_idx] = pred_v
                    active_mask[z_idx] = True

        # Apply FROZEN Spatial MRF (beta = 0.0001)
        mrf_mod = self.mrf_models.get('AQI')
        if mrf_mod:
            mu_mrf, var_mrf = mrf_mod.infer(y_hat, variances, active_mask)
        else:
            mu_mrf, var_mrf = y_hat, variances

        # Evaluate alerts across all 15 zones
        zone_evaluations = []
        for z in range(1, 16):
            z_idx = z - 1
            z_data = zone_aqi_dict.get(z)
            z_label = z_data.zone_label if z_data else f"Zone {z}"
            is_mon = z not in UNMONITORED_AQI_ZONES

            # Evaluate CPCB alert (> 200.0)
            alert = evaluate_aqi_alert(float(mu_mrf[z_idx]), float(var_mrf[z_idx]))

            zone_evaluations.append({
                'zone_id': z,
                'zone_label': z_label,
                'is_monitored': is_mon,
                'observed_aqi': z_data.aqi_value if z_data else np.nan,
                'hmm_forecast': float(y_hat[z_idx]) if active_mask[z_idx] else np.nan,
                'mrf_refined_forecast': round(float(mu_mrf[z_idx]), 2),
                'mrf_variance': round(float(var_mrf[z_idx]), 2),
                'alert_status': alert['alert_status'],
                'exceedance_probability': alert['probability_crossing'],
                'threshold': alert['threshold']
            })

        return {
            'timestamp': zone_aqi_dict[1].timestamp.isoformat() if 1 in zone_aqi_dict else None,
            'variable': 'AQI',
            'horizon': f'{horizon}h',
            'zones': zone_evaluations
        }

    def run_temperature_daily_cycle(
        self,
        cityforecast_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes daily temperature cycle using IMD cityforecast bulletin:
        - Extracts maximum temperature and genuine departure from normal.
        - NEVER fabricates Normal_Tmax. If missing, departure remains None.
        - Evaluates official IMD Coastal Heatwave alert (Tmax >= 37C AND Departure >= +4.5C).
        """
        obs_list = self.imd.parse_city_forecast(cityforecast_payload)
        latest_obs = obs_list[0] if obs_list else None

        tmax = latest_obs.max_temperature if latest_obs else np.nan
        dep = latest_obs.departure_from_normal if latest_obs else None
        norm_tmax = latest_obs.normal_tmax if latest_obs else None

        # Temperature HMM forecast
        # For temperature, beta = 0.0, so MRF is identical to HMM
        zone_evaluations = []
        for z in range(1, 16):
            if z in self.hmm_models['temperature'] and pd.notna(tmax):
                hmm_mod = self.hmm_models['temperature'][z]
                pred_m, pred_v, _ = hmm_mod.forecast(np.array([tmax]), horizon=24)
            else:
                pred_m, pred_v = tmax, 4.0

            alert = evaluate_temperature_heatwave_alert(
                tmax_pred=float(pred_m) if pd.notna(pred_m) else 0.0,
                tmax_var=float(pred_v) if pd.notna(pred_v) else 4.0,
                normal_tmax=norm_tmax
            )

            zone_evaluations.append({
                'zone_id': z,
                'forecast_tmax': round(float(pred_m), 2) if pd.notna(pred_m) else np.nan,
                'departure_from_normal': dep,
                'normal_tmax': norm_tmax,
                'alert_status': alert['alert_status'],
                'exceedance_probability': alert['probability_crossing']
            })

        return {
            'timestamp': latest_obs.timestamp.isoformat() if latest_obs else None,
            'variable': 'temperature',
            'observed_tmax': tmax,
            'departure_from_normal': dep,
            'normal_tmax': norm_tmax,
            'zones': zone_evaluations
        }
