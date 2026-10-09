"""
Live inference (component 3 of 4: collect -> preprocess -> INFER -> present).

Reuses the EXISTING frozen artifacts only (no retraining):
  models/hmm/hmm_aqi_zone_{2,4,5,10,12,13,14}.joblib      hourly CPCB AQI HMMs
  models/hmm/hmm_rainfall_zone_{4,5,10,14}.joblib         hourly rainfall HMMs (trained on gauges)
  models/hmm/hmm_temperature_zone_{1..15}.joblib          DAILY MEAN temperature HMM (IMD Tmean)
  models/hmm_mrf/mrf_{aqi,rainfall,temperature}.joblib    Gaussian MRFs (beta = 1e-4, 0, 0)

What each model is allowed to output (see reports/final_data_audit.md, decision table):
  AQI          1h/3h/6h/24h, from measured OpenAQ station AQI; MRF spatial estimate for zones
               without a usable measurement, only when an adjacent zone has a measured input.
  Rainfall     DISABLED by default (RAINFALL_FORECAST_ENABLED). The rainfall HMMs (zones 4, 5, 10, 14)
               were trained on CAAQMS gauge data; no live gauge feed exists, and the final audit found
               (a) the Perungudi (zone 14) gauge used for training is drifting (~12,600 mm in 2025),
               (b) gauges miss major events seen in reanalysis, and (c) feeding Open-Meteo precipitation
               into the gauge-trained HMMs gives near-zero skill on the existing test period. Rainfall is
               therefore reported as current reanalysis conditions only. The code path is retained and
               tested so it can be re-enabled if a validated input becomes available.
  Temperature  next-day DAILY MEAN only (the HMM was trained on IMD daily mean temperature).
               It is not a Tmax model, so the IMD heatwave criterion (Tmax-based) is not evaluated.
"""

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.alerts.alert_engine import (
    UNAVAILABLE_NO_DATA, compute_24h_rainfall_accumulation, evaluate_aqi_alert,
    evaluate_rainfall_alert_24h,
)
from src.preprocessing.cpcb_aqi import aqi_category

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(ROOT, 'models')

AQI_HORIZONS = [1, 3, 6, 24]
RAINFALL_HORIZONS = [1, 3, 6, 24]
AQI_CONTEXT_HOURS = 72
RAIN_CONTEXT_HOURS = 72
TEMP_CONTEXT_DAYS = 30
# A zone's AQI input is usable only if its last measured hour is within this many hours of the
# forecast origin; the HMM propagates the state through the missing hours (no filling).
AQI_MAX_INPUT_GAP_HOURS = 6

AQI_HMM_ZONES = [2, 4, 5, 10, 12, 13, 14]
RAIN_HMM_ZONES = [4, 5, 10, 14]

RAINFALL_FORECAST_ENABLED = False
RAIN_DISABLED = 'DISABLED_INPUT_NOT_VALIDATED'
STALE_INPUT = 'UNAVAILABLE_STALE_INPUT'
NO_MODEL = 'NOT_SUPPORTED_NO_MODEL'
NO_MEASUREMENT = 'UNAVAILABLE_NO_MEASUREMENT'
HEATWAVE_NOT_EVALUATED = 'NOT_EVALUATED_MODEL_IS_DAILY_MEAN'


def _load(path):
    return joblib.load(path) if os.path.exists(path) else None


class LiveForecaster:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.aqi_hmm = {z: m for z in AQI_HMM_ZONES
                        if (m := _load(os.path.join(models_dir, 'hmm', f'hmm_aqi_zone_{z}.joblib')))}
        self.rain_hmm = {z: m for z in RAIN_HMM_ZONES
                         if (m := _load(os.path.join(models_dir, 'hmm', f'hmm_rainfall_zone_{z}.joblib')))}
        self.temp_hmm = {z: m for z in range(1, 16)
                         if (m := _load(os.path.join(models_dir, 'hmm', f'hmm_temperature_zone_{z}.joblib')))}
        self.mrf = {k: _load(os.path.join(models_dir, 'hmm_mrf', f'mrf_{k}.joblib'))
                    for k in ('aqi', 'rainfall', 'temperature')}
        mrf = self.mrf['aqi']
        self.neighbors: Dict[int, List[int]] = {z: [] for z in range(1, 16)}
        if mrf is not None:
            for u, v, _ in mrf.edge_list:
                self.neighbors[u + 1].append(v + 1)
                self.neighbors[v + 1].append(u + 1)

    def artifacts_summary(self) -> Dict[str, Any]:
        return {
            'aqi_hmm_zones': sorted(self.aqi_hmm), 'rainfall_hmm_zones': sorted(self.rain_hmm),
            'temperature_hmm_zones': sorted(self.temp_hmm),
            'mrf_beta': {k: (float(m.beta) if m is not None else None) for k, m in self.mrf.items()},
        }

    # ── AQI ──────────────────────────────────────────────────────────────────
    def run_aqi(self, stations: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        primary = {s['zone_id']: s for s in stations.values() if s['role'] == 'primary'}
        latest = [s['latest_valid_utc'] for s in primary.values() if s['latest_valid_utc'] is not None]
        if not latest:
            return {'origin_utc': None, 'zones': {}, 'reason': 'No measured AQI available from any station'}
        origin = max(latest)

        zone_input: Dict[int, Dict[str, Any]] = {}
        for z in range(1, 16):
            st = primary.get(z)
            info = {'zone_id': z, 'station': st['station_name'] if st else None,
                    'has_aqi_hmm': z in self.aqi_hmm}
            if st is None or st['latest_valid_utc'] is None:
                info.update(active=False, reason=NO_MEASUREMENT if z in self.aqi_hmm else NO_MODEL)
            else:
                gap_h = (origin - st['latest_valid_utc']) / pd.Timedelta(hours=1)
                info.update(last_measured_utc=st['latest_valid_utc'], input_gap_hours=gap_h)
                if z not in self.aqi_hmm:
                    info.update(active=False, reason=NO_MODEL)
                elif gap_h > AQI_MAX_INPUT_GAP_HOURS:
                    info.update(active=False, reason=STALE_INPUT)
                else:
                    seq = st['aqi'][st['aqi'].index <= origin].iloc[-AQI_CONTEXT_HOURS:]
                    info.update(active=True, reason=None, seq=seq.values.astype(float))
            zone_input[z] = info

        active_mask = np.array([zone_input[z]['active'] for z in range(1, 16)])
        out_h: Dict[str, List[Dict[str, Any]]] = {}
        for h in AQI_HORIZONS:
            y_hat = np.full(15, np.nan)
            var = np.full(15, np.nan)
            for z in range(1, 16):
                if zone_input[z]['active']:
                    mu, v, _ = self.aqi_hmm[z].forecast(zone_input[z]['seq'], horizon=h)
                    y_hat[z - 1], var[z - 1] = mu, v
            if active_mask.any() and self.mrf['aqi'] is not None:
                mu_mrf, var_mrf = self.mrf['aqi'].infer(y_hat, var, active_mask)
            else:
                # With no active zone the MRF posterior is uninformative (mean 0, huge variance);
                # report unavailable instead of a misleading value.
                mu_mrf, var_mrf = np.full(15, np.nan), np.full(15, np.nan)
            rows = []
            for z in range(1, 16):
                zi = zone_input[z]
                has_active_neighbor = any(zone_input[n]['active'] for n in self.neighbors[z])
                if zi['active']:
                    basis = 'HMM+MRF (measured input)'
                elif has_active_neighbor and not np.isnan(mu_mrf[z - 1]):
                    basis = 'MRF spatial estimate (no usable local measurement)'
                else:
                    basis = None
                if basis is None:
                    status = zi['reason'] or UNAVAILABLE_NO_DATA
                    rows.append({'zone_id': z, 'hmm_pred': None, 'mrf_pred': None, 'mrf_var': None,
                                 'alert_status': status, 'exceedance_prob': None, 'threshold': 200.0,
                                 'is_monitored': zi['has_aqi_hmm'], 'availability': 'UNAVAILABLE',
                                 'basis': None, 'reason': zi['reason'],
                                 'target_time_utc': (origin + pd.Timedelta(hours=h)).isoformat()})
                    continue
                alert = evaluate_aqi_alert(float(mu_mrf[z - 1]), float(var_mrf[z - 1]))
                rows.append({
                    'zone_id': z,
                    'hmm_pred': _r(y_hat[z - 1], 2), 'mrf_pred': _r(mu_mrf[z - 1], 2),
                    'mrf_var': _r(var_mrf[z - 1], 2), 'mrf_std': _r(np.sqrt(var_mrf[z - 1]), 1),
                    'category': aqi_category(float(mu_mrf[z - 1])),
                    'alert_status': alert['alert_status'], 'exceedance_prob': _r(alert['probability_crossing'], 4),
                    'threshold': alert['threshold'], 'is_monitored': zi['has_aqi_hmm'],
                    'availability': 'MODEL_FORECAST' if zi['active'] else 'MODEL_ESTIMATE',
                    'basis': basis, 'reason': zi['reason'],
                    'target_time_utc': (origin + pd.Timedelta(hours=h)).isoformat(),
                })
            out_h[f'{h}h'] = rows
        zone_inputs_public = {z: {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v)
                                  for k, v in zi.items() if k != 'seq'} for z, zi in zone_input.items()}
        return {'origin_utc': origin, 'horizons': out_h, 'zone_inputs': zone_inputs_public,
                'n_active': int(active_mask.sum())}

    # ── Rainfall ─────────────────────────────────────────────────────────────
    def run_rainfall(self, precip_by_zone: Dict[int, pd.Series], origin: Optional[pd.Timestamp],
                     enabled: bool = True) -> Dict[str, Any]:
        out_h: Dict[str, List[Dict[str, Any]]] = {f'{h}h': [] for h in RAINFALL_HORIZONS}
        if not enabled:
            for h in RAINFALL_HORIZONS:
                for z in range(1, 16):
                    out_h[f'{h}h'].append({
                        'zone_id': z, 'hmm_pred': None, 'mrf_pred': None, 'mrf_var': None, 'accum_24h_mm': None,
                        'alert_status': RAIN_DISABLED if z in self.rain_hmm else NO_MODEL,
                        'exceedance_prob': None, 'threshold': 64.5, 'is_monitored': z in self.rain_hmm,
                        'availability': 'UNAVAILABLE',
                        'reason': ('Gauge-trained HMM; live input (reanalysis) failed validation'
                                   if z in self.rain_hmm else NO_MODEL),
                        'target_time_utc': None})
            return {'origin_utc': origin, 'horizons': out_h, 'n_active': 0, 'enabled': False}
        gamma: Dict[int, np.ndarray] = {}
        seqs: Dict[int, np.ndarray] = {}
        for z in self.rain_hmm:
            s = precip_by_zone.get(z)
            if s is not None and s.notna().any() and pd.notna(s.iloc[-1]):
                seqs[z] = s.values.astype(float)
                gamma[z] = self.rain_hmm[z].get_filtered_state_distribution(seqs[z])
        active = np.array([z in seqs for z in range(1, 16)])
        for h in RAINFALL_HORIZONS:
            y_hat, var = np.full(15, np.nan), np.full(15, np.nan)
            for z, seq in seqs.items():
                y_hat[z - 1], var[z - 1], _ = self.rain_hmm[z].forecast(seq, horizon=h)
            if active.any() and self.mrf['rainfall'] is not None:
                mu, vv = self.mrf['rainfall'].infer(y_hat, var, active)   # beta = 0 -> identity
            else:
                mu, vv = y_hat, var
            for z in range(1, 16):
                tgt = (origin + pd.Timedelta(hours=h)).isoformat() if origin is not None else None
                if z not in self.rain_hmm:
                    out_h[f'{h}h'].append({'zone_id': z, 'hmm_pred': None, 'mrf_pred': None, 'mrf_var': None,
                                           'accum_24h_mm': None, 'alert_status': NO_MODEL,
                                           'exceedance_prob': None, 'threshold': 64.5, 'is_monitored': False,
                                           'availability': 'UNAVAILABLE', 'reason': NO_MODEL,
                                           'target_time_utc': tgt})
                    continue
                if z not in seqs:
                    out_h[f'{h}h'].append({'zone_id': z, 'hmm_pred': None, 'mrf_pred': None, 'mrf_var': None,
                                           'accum_24h_mm': None, 'alert_status': UNAVAILABLE_NO_DATA,
                                           'exceedance_prob': None, 'threshold': 64.5, 'is_monitored': True,
                                           'availability': 'UNAVAILABLE', 'reason': 'No precipitation input',
                                           'target_time_utc': tgt})
                    continue
                rec = {'zone_id': z, 'hmm_pred': _r(y_hat[z - 1], 3), 'mrf_pred': _r(mu[z - 1], 3),
                       'mrf_var': _r(vv[z - 1], 4), 'accum_24h_mm': None, 'threshold': 64.5,
                       'is_monitored': True, 'availability': 'MODEL_FORECAST',
                       'basis': 'HMM (gauge-trained) driven by Open-Meteo reanalysis precipitation',
                       'reason': None, 'target_time_utc': tgt}
                if h == 24:
                    m = self.rain_hmm[z]
                    mu_acc, var_acc = compute_24h_rainfall_accumulation(gamma[z], m.transmat_, m.means_, m.variances_)
                    a = evaluate_rainfall_alert_24h(mu_acc, var_acc, is_monitored=True)
                    rec.update(accum_24h_mm=_r(mu_acc, 2), accum_24h_std=_r(np.sqrt(var_acc), 2),
                               alert_status=a['alert_status'], exceedance_prob=_r(a['probability_crossing'], 4))
                else:
                    # The official criterion is a 24 h accumulation; hourly values are not alerted.
                    rec.update(alert_status='NOT_APPLICABLE_HOURLY', exceedance_prob=None)
                out_h[f'{h}h'].append(rec)
        return {'origin_utc': origin, 'horizons': out_h, 'n_active': int(active.sum()), 'enabled': True}

    # ── Temperature ──────────────────────────────────────────────────────────
    def run_temperature(self, daily: pd.DataFrame) -> Dict[str, Any]:
        """daily: columns zone_id, date_ist, tmean_c (NaN for incomplete days)."""
        rows, last_dates = [], []
        y_hat, var = np.full(15, np.nan), np.full(15, np.nan)
        info: Dict[int, Dict[str, Any]] = {}
        for z in range(1, 16):
            g = daily[daily['zone_id'] == z].sort_values('date_ist')
            g_valid = g.dropna(subset=['tmean_c'])
            if z not in self.temp_hmm or g_valid.empty:
                info[z] = {'last_day': None}
                continue
            last_day = g_valid['date_ist'].iloc[-1]
            seq = g[g['date_ist'] <= last_day]['tmean_c'].iloc[-TEMP_CONTEXT_DAYS:].values.astype(float)
            y_hat[z - 1], var[z - 1], _ = self.temp_hmm[z].forecast(seq, horizon=1)
            info[z] = {'last_day': last_day, 'last_tmean': float(g_valid['tmean_c'].iloc[-1])}
            last_dates.append(last_day)
        active = ~np.isnan(y_hat)
        if active.any() and self.mrf['temperature'] is not None:
            mu, vv = self.mrf['temperature'].infer(y_hat, var, active)   # beta = 0 -> identity
        else:
            mu, vv = y_hat, var
        for z in range(1, 16):
            ld = info[z]['last_day']
            if ld is None or np.isnan(mu[z - 1]) or not active[z - 1]:
                rows.append({'zone_id': z, 'next_day_tmean_pred_c': None, 'pred_var': None,
                             'alert_status': UNAVAILABLE_NO_DATA, 'exceedance_prob': None,
                             'availability': 'UNAVAILABLE', 'is_monitored': False})
                continue
            rows.append({
                'zone_id': z,
                'last_day_tmean_c': _r(info[z]['last_tmean'], 2),
                'input_last_day_ist': str(ld),
                'target_day_ist': str(pd.Timestamp(ld) + pd.Timedelta(days=1))[:10],
                'next_day_tmean_pred_c': _r(mu[z - 1], 2), 'pred_var': _r(vv[z - 1], 3),
                'pred_std': _r(np.sqrt(vv[z - 1]), 2),
                'alert_status': HEATWAVE_NOT_EVALUATED, 'exceedance_prob': None,
                'availability': 'MODEL_FORECAST', 'is_monitored': False,
                'basis': 'Daily-mean HMM (trained on IMD daily mean) driven by Open-Meteo reanalysis daily means',
                'note': ('Next-day DAILY MEAN temperature. Not a Tmax forecast; the IMD heatwave '
                         'criterion (Tmax and departure from normal) is therefore not evaluated. Low skill: '
                         'next-day MAE ~1.0 °C against Open-Meteo daily means (2025-2026), larger than the '
                         '~0.5 °C typical day-to-day change (see reports/final_data_audit.md).'),
            })
        return {'last_input_day_ist': str(max(last_dates)) if last_dates else None, 'horizons': {'24h': rows}}


def _r(v, nd):
    try:
        f = float(v)
        return None if not np.isfinite(f) else round(f, nd)
    except (TypeError, ValueError):
        return None
