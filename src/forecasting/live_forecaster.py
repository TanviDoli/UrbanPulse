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

RAINFALL_FORECAST_ENABLED = False          # gauge-trained HMMs above (input mismatch); see docstring
# Open-Meteo-trained rainfall HMMs (one per grid cell, src/models/train_and_validate_hmm.py
# train_rainfall_openmeteo). When this artifact exists it replaces the gauge path; each horizon
# is published only if it passed the baseline evaluation stored in the artifact ('gate').
RAIN_OM_ARTIFACT = os.path.join('hmm', 'hmm_rainfall_openmeteo.joblib')
# Two-part rainfall model (src/models/rain_model.py; train_rainfall_twopart). Takes precedence over
# both HMM paths. Per horizon, the probability is published only if it passed its gate and the
# expected amount only if the amount gate also passed. Heavy-rain alerts are never evaluated.
RAIN_TWOPART_ARTIFACT = os.path.join('rainfall', 'rain_twopart_openmeteo.joblib')
RAIN_HEAVY_NOT_EVALUATED = 'NOT_EVALUATED_INSUFFICIENT_HEAVY_RAIN_EVENTS'
# Hourly temperature regressions (src/models/temp_model.py; train_temperature_hourly). When present
# they replace the daily-mean HMM output; only horizons that passed validation AND test are published.
TEMP_HOURLY_ARTIFACT = os.path.join('temperature', 'temp_hourly_openmeteo.joblib')
# Optional package for the frozen long-horizon temperature candidates (+6 h / +24 h, trained on the 2017-2023
# history; src/models/package_history_models.py). DISABLED by default: loaded only when the forecaster is created
# with enable_temp_history=True or URBANPULSE_ENABLE_TEMP_HISTORY_6H_24H=1. It serves only the horizons it contains,
# with its own zone->cell mapping and tables; 1 h / 3 h always come from the production bundle.
TEMP_HISTORY_ARTIFACT = os.path.join('temperature', 'temp_hourly_history_6h_24h.joblib')
TEMP_HISTORY_ENV = 'URBANPULSE_ENABLE_TEMP_HISTORY_6H_24H'


def temp_history_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    v = (os.environ if env is None else env).get(TEMP_HISTORY_ENV, '')
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


# Optional EXPERIMENTAL rainfall package (frozen history-trained R3 +6 h, R4 +24 h occurrence/amount and the C2
# 24 h total; passed the 2024 test, not promoted). Disabled unless URBANPULSE_ENABLE_RAIN_HISTORY_6H_24H=1.
RAIN_HISTORY_ARTIFACT = os.path.join('rainfall', 'rain_history_6h_24h.joblib')
RAIN_HISTORY_ENV = 'URBANPULSE_ENABLE_RAIN_HISTORY_6H_24H'
RAIN_HISTORY_CONTEXT_HOURS = 721            # C2 needs 7-day windows and up to 720 h since the last wet hour


def rain_history_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    v = (os.environ if env is None else env).get(RAIN_HISTORY_ENV, '')
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')
TEMP_HOURLY_HORIZONS = [1, 3, 6, 24]
HEATWAVE_CRITERION_NOT_EVALUATED = 'NOT_EVALUATED_HEATWAVE_CRITERION'
RAIN_OM_ACCUM_KEY = '24h_total'
RAIN_FAILED_EVAL = 'NOT_SUPPORTED_FAILED_EVALUATION'
RAIN_DISABLED = 'DISABLED_INPUT_NOT_VALIDATED'
STALE_INPUT = 'UNAVAILABLE_STALE_INPUT'
NO_MODEL = 'NOT_SUPPORTED_NO_MODEL'
NO_MEASUREMENT = 'UNAVAILABLE_NO_MEASUREMENT'
HEATWAVE_NOT_EVALUATED = 'NOT_EVALUATED_MODEL_IS_DAILY_MEAN'


def _load(path):
    return joblib.load(path) if os.path.exists(path) else None


# ── promoted-candidate inputs (same builders and stored training tables as the experiment) ──
WEATHER_COLUMNS = ['temperature_2m_c', 'relative_humidity_2m_pct', 'wind_speed_10m_kmh',
                   'wind_direction_10m_deg', 'precipitation_mm']
WEATHER_CONTEXT_HOURS = 72


def _ist(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert('Asia/Kolkata')


def temp_candidate_row(frame: pd.DataFrame, h: int, cell: str, enc: Dict[str, Any]) -> pd.DataFrame:
    """One-row feature frame at the origin (= last index of `frame`) for a promoted temperature model."""
    from src.models.encoding import lookup
    from src.models.temp_model import temp_candidate_features
    f = temp_candidate_features(frame, h).iloc[[-1]].copy()
    origin = frame.index[-1]
    tgt, org = _ist(origin + pd.Timedelta(hours=h)), _ist(origin)
    f['clim_target'] = lookup(enc['clim_target'], {'target_month': tgt.month, 'target_hour': tgt.hour}, cell)
    f['anom_now'] = f['t_now'] - lookup(enc['clim_origin'], {'origin_month': org.month, 'origin_hour': org.hour}, cell)
    return f


def rain_candidate_row(precip: pd.Series, frame: pd.DataFrame, h: int, cell: str, cells: List[str],
                       enc: Dict[str, Any]) -> pd.DataFrame:
    """One-row feature frame at the origin for a promoted two-part rain model (precip = the zone's series,
    frame = the representative zone's weather; both end at the origin)."""
    from src.models.encoding import logit_clip, lookup
    from src.models.rain_model import WET_MM, build_features, rain_weather_features
    f = build_features(precip).iloc[[-1]].copy()
    for c in cells:
        f[f'is_{c}'] = float(cell == c)
    w = rain_weather_features(frame, h).iloc[[-1]]
    for col in w.columns:
        f[col] = w[col].to_numpy()
    origin = precip.index[-1]
    tgt, org = _ist(origin + pd.Timedelta(hours=h)), _ist(origin)
    state = int(precip.iloc[-1] >= WET_MM)
    f['logit_pclim'] = logit_clip(lookup(enc['pclim'], {'target_month': tgt.month, 'target_hour': tgt.hour}))
    f['logit_pcond'] = logit_clip(lookup(enc['pcond'], {'state': state, 'origin_month': org.month}))
    return f


def rain_history_row(precip: pd.Series, frame: pd.DataFrame, h: int, cell: str, cells: List[str],
                     enc: Dict[str, Any]) -> pd.DataFrame:
    """Inputs for the history rain models: the promoted-model inputs plus, where stored, the harmonic seasonal
    probability (logistic fitted on 2017-2023 training rows, evaluated at the target time)."""
    from src.models.encoding import logit_clip
    f = rain_candidate_row(precip, frame, h, cell, cells, enc)
    if 'pharm' in enc:
        from src.models.train_and_validate_hmm import harmonic_design
        X = harmonic_design(pd.DatetimeIndex([precip.index[-1] + pd.Timedelta(hours=h)]))[:, 1:]
        f['logit_pharm'] = logit_clip(enc['pharm'].predict_proba(X)[:, 1])
    return f


def c2_feature_row(frame: pd.DataFrame, cell: str, cells: List[str], enc: Dict[str, Any]) -> pd.DataFrame:
    """C2 (24 h total) inputs at the origin (= last row of `frame`, >= 721 h of the cell's weather), built by the
    experiment's own causal feature function plus the stored training-only tables."""
    from src.models.encoding import logit_clip, lookup
    from src.models.rain24_total_experiment import EVENT_MM, cell_features
    f = cell_features(frame).iloc[[-1]].copy()
    for c in cells:
        f[f'is_{c}'] = float(cell == c)
    month = int(f['origin_month'].iloc[0])
    state = int(f['persistence'].iloc[0] >= EVENT_MM) if pd.notna(f['persistence'].iloc[0]) else None
    f['logit_pclim25'] = logit_clip(lookup(enc['pclim25'], {'origin_month': month}, cell))
    f['logit_pcond25'] = (logit_clip(lookup(enc['pcond25'], {'state': state, 'origin_month': month}))
                          if state is not None else np.nan)
    f['log_clim_mean'] = np.log1p(lookup(enc['clim_mean'], {'origin_month': month}, cell))
    return f


class LiveForecaster:
    def __init__(self, models_dir: str = MODELS_DIR, enable_temp_history: Optional[bool] = None,
                 enable_rain_history: Optional[bool] = None):
        self.aqi_hmm = {z: m for z in AQI_HMM_ZONES
                        if (m := _load(os.path.join(models_dir, 'hmm', f'hmm_aqi_zone_{z}.joblib')))}
        self.rain_hmm = {z: m for z in RAIN_HMM_ZONES
                         if (m := _load(os.path.join(models_dir, 'hmm', f'hmm_rainfall_zone_{z}.joblib')))}
        self.temp_hmm = {z: m for z in range(1, 16)
                         if (m := _load(os.path.join(models_dir, 'hmm', f'hmm_temperature_zone_{z}.joblib')))}
        self.rain_om = _load(os.path.join(models_dir, RAIN_OM_ARTIFACT))
        self.rain_twopart = _load(os.path.join(models_dir, RAIN_TWOPART_ARTIFACT))
        self.temp_hourly = _load(os.path.join(models_dir, TEMP_HOURLY_ARTIFACT))
        use_hist = temp_history_enabled() if enable_temp_history is None else bool(enable_temp_history)
        self.temp_history = _load(os.path.join(models_dir, TEMP_HISTORY_ARTIFACT)) if use_hist else None
        use_rh = rain_history_enabled() if enable_rain_history is None else bool(enable_rain_history)
        self.rain_history = _load(os.path.join(models_dir, RAIN_HISTORY_ARTIFACT)) if use_rh else None
        self.mrf = {k: _load(os.path.join(models_dir, 'hmm_mrf', f'mrf_{k}.joblib'))
                    for k in ('aqi', 'rainfall', 'temperature')}
        mrf = self.mrf['aqi']
        self.neighbors: Dict[int, List[int]] = {z: [] for z in range(1, 16)}
        if mrf is not None:
            for u, v, _ in mrf.edge_list:
                self.neighbors[u + 1].append(v + 1)
                self.neighbors[v + 1].append(u + 1)

    @property
    def rainfall_zones(self) -> List[int]:
        """Zones that receive a precipitation input series for rainfall forecasting."""
        bundle = self.rain_twopart or self.rain_om
        return sorted(bundle['zone_to_cell']) if bundle else sorted(self.rain_hmm)

    @property
    def rainfall_enabled(self) -> bool:
        """Open-Meteo-trained models (gated per horizon) when present, else the gauge-path flag."""
        return self.rain_twopart is not None or self.rain_om is not None or RAINFALL_FORECAST_ENABLED

    def artifacts_summary(self) -> Dict[str, Any]:
        return {
            'aqi_hmm_zones': sorted(self.aqi_hmm), 'rainfall_hmm_zones': sorted(self.rain_hmm),
            'temperature_hmm_zones': sorted(self.temp_hmm),
            'mrf_beta': {k: (float(m.beta) if m is not None else None) for k, m in self.mrf.items()},
            'rainfall_openmeteo_gate': ({k: g.get('passed') for k, g in self.rain_om['gate'].items()}
                                        if self.rain_om else None),
            'rainfall_twopart_gate': ({k: {'probability': g.get('probability_passed'), 'amount': g.get('amount_passed')}
                                       for k, g in self.rain_twopart['gate'].items()} if self.rain_twopart else None),
            'temperature_hourly_gate': ({k: g.get('passed') for k, g in self.temp_hourly['gate'].items()}
                                        if self.temp_hourly else None),
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
                     enabled: bool = True, weather_by_zone: Optional[Dict[int, pd.DataFrame]] = None,
                     long_weather_by_zone: Optional[Dict[int, pd.DataFrame]] = None) -> Dict[str, Any]:
        res = self._run_rainfall_core(precip_by_zone, origin, enabled, weather_by_zone)
        if enabled and getattr(self, 'rain_history', None) is not None and origin is not None:
            res = self._apply_rain_history(res, origin, weather_by_zone or {}, long_weather_by_zone or {})
        return res

    def _apply_rain_history(self, res: Dict[str, Any], origin: pd.Timestamp, weather: Dict[int, pd.DataFrame],
                            long_weather: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """EXPERIMENTAL +6 h / +24 h rows from the frozen history-trained package (zone -> its own live grid cell).
        Probability, expected amount and (24 h) the C2 total are published only from complete inputs; heavy-rain
        alerts are never evaluated."""
        from src.models.rain_model import SEASON_OF_MONTH
        hb = self.rain_history
        z2c, rep, cells = hb['zone_to_cell'], hb['rep_zone'], hb['cells']
        c2, c2e = hb['c2']['model'], hb['c2']['encodings']
        tag = 'rain_history_6h_24h (EXPERIMENTAL: frozen 2017-2023 models, passed 2024 test, not promoted)'
        for key, m in hb['models'].items():
            h, g = int(key[:-1]), hb['gate'][key]
            tgt = origin + pd.Timedelta(hours=h)
            season = SEASON_OF_MONTH[tgt.tz_convert('Asia/Kolkata').month]
            rows = {r['zone_id']: r for r in res['horizons'].get(key, [])}
            any_ok = False
            for z in range(1, 16):
                rec = rows.setdefault(z, {'zone_id': z})
                cell = z2c[z]
                rec.update(hmm_pred=None, mrf_pred=None, mrf_var=None, prob_rain=None, median_mm=None,
                           accum_24h_mm=None, exceedance_prob=None, grid_cell=cell, model_package=tag,
                           quantity='hourly_rate_mm_per_h', target_time_utc=tgt.isoformat(),
                           heavy_rain_alert=RAIN_HEAVY_NOT_EVALUATED, evaluation=dict(g))
                frame = weather.get(rep[cell])
                x = None
                if g['probability_passed'] and frame is not None and len(frame) >= 25:
                    x = rain_history_row(frame['precipitation_mm'], frame, h, cell, cells, hb['encodings'][key])
                    if x[m.feature_names].isna().any(axis=None):
                        x = None
                if x is not None:
                    pr = m.predict(x[m.feature_names], np.array([season]))
                    rec.update(prob_rain=_r(pr['p_wet'][0], 3), mrf_pred=_r(pr['mean'][0], 3), median_mm=_r(pr['median'][0], 2))
                if key == '24h' and hb['gate'].get('24h_total', {}).get('passed'):
                    lw = long_weather.get(rep[cell])
                    if lw is not None and len(lw) >= RAIN_HISTORY_CONTEXT_HOURS:
                        xc = c2_feature_row(lw, cell, cells, c2e)
                        if not xc[c2.feats].isna().any(axis=None):
                            pc = c2.predict(xc)
                            rec.update(accum_24h_mm=_r(pc['mean'][0], 2), accum_24h_median=_r(pc['median'][0], 2),
                                       accum_24h_q10=_r(pc['Q'][0, 4:6].mean(), 2), accum_24h_q90=_r(pc['Q'][0, 44:46].mean(), 2),
                                       prob_total_ge_2_5mm=_r(pc['p25'][0], 3), accum_quantity='total_next_24h_mm',
                                       evaluation={**rec['evaluation'], 'total_24h': hb['gate']['24h_total']})
                ok = rec.get('prob_rain') is not None or rec.get('accum_24h_mm') is not None
                any_ok = any_ok or ok
                rec.update(availability='MODEL_FORECAST' if ok else 'UNAVAILABLE',
                           alert_status='NOT_APPLICABLE_HOURLY' if ok else UNAVAILABLE_NO_DATA,
                           reason=None if ok else 'Incomplete input for the experimental model',
                           basis=('EXPERIMENTAL history-trained model (Open-Meteo 2017-2023), driven by Open-Meteo data'
                                  if ok else None))
            res['horizons'][key] = [rows[z] for z in range(1, 16)]
            sup = res.setdefault('supported_horizons', [])
            if any_ok and key not in sup:
                sup.append(key)
        res['supported_horizons'] = [k for k in ('1h', '3h', '6h', '24h') if k in res.get('supported_horizons', [])]
        res['enabled'] = bool(res['supported_horizons'])
        return res

    def _run_rainfall_core(self, precip_by_zone: Dict[int, pd.Series], origin: Optional[pd.Timestamp],
                           enabled: bool = True, weather_by_zone: Optional[Dict[int, pd.DataFrame]] = None) -> Dict[str, Any]:
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
        if self.rain_twopart is not None:
            return self._run_rainfall_twopart(precip_by_zone, origin, weather_by_zone)
        if self.rain_om is not None:
            return self._run_rainfall_openmeteo(precip_by_zone, origin)
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

    def _rain_inputs(self, key: str, z: int, base: Dict[int, pd.DataFrame], precip_by_zone: Dict[int, pd.Series],
                     weather_by_zone: Optional[Dict[int, pd.DataFrame]]) -> Optional[pd.DataFrame]:
        """Feature row for zone z and horizon `key`; promoted models (with stored encodings) get the
        candidate inputs from the representative zone's weather. None if any required input is missing."""
        b = self.rain_twopart
        model, enc = b['models'][key], (b.get('encodings') or {}).get(key)
        if enc is None:
            x = base.get(z)
        else:
            rep = b['rep_zone'][b['zone_to_cell'][z]]
            frame = (weather_by_zone or {}).get(rep)
            s = precip_by_zone.get(z)
            if frame is None or s is None or len(frame) < 25 or len(s) < 25:
                return None
            x = rain_candidate_row(s, frame, int(key[:-1]), b['zone_to_cell'][z], b['cells'], enc)
        if x is None or x[model.feature_names].isna().any(axis=None):
            return None
        return x

    def _run_rainfall_twopart(self, precip_by_zone: Dict[int, pd.Series],
                              origin: Optional[pd.Timestamp],
                              weather_by_zone: Optional[Dict[int, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Two-part model per horizon: prob_rain = P(rain >= 0.1 mm in the target hour) and
        mrf_pred = expected hourly amount (mm/h). Published only for horizons whose probability
        (and amount) passed validation AND test; others are NOT_SUPPORTED_FAILED_EVALUATION.
        The 24 h total failed its gate and heavy-rain alerts are not evaluated, so neither is published.
        """
        from src.models.rain_model import SEASON_OF_MONTH, build_features
        b = self.rain_twopart
        gate, z2c, cells = b['gate'], b['zone_to_cell'], b['cells']
        feats: Dict[int, pd.DataFrame] = {}
        for z, s in precip_by_zone.items():
            if z in z2c and s is not None and len(s) >= 24 and not s.iloc[-24:].isna().any():
                f = build_features(s).iloc[[-1]].copy()            # features at the origin only
                for c in cells:
                    f[f'is_{c}'] = float(z2c[z] == c)
                feats[z] = f
        basis = ('Two-part model (rain probability + seasonal amount) trained on Open-Meteo reanalysis, '
                 'driven by Open-Meteo precipitation')
        out_h: Dict[str, List[Dict[str, Any]]] = {}
        supported: List[str] = []
        for h in RAINFALL_HORIZONS:
            key = f'{h}h'
            g = gate.get(key, {})
            p_ok, a_ok = bool(g.get('probability_passed')), bool(g.get('amount_passed'))
            if p_ok and feats:
                supported.append(key)
            tgt_ts = origin + pd.Timedelta(hours=h) if origin is not None else None
            season = SEASON_OF_MONTH[tgt_ts.tz_convert('Asia/Kolkata').month] if tgt_ts is not None else None
            model = b['models'][key]
            rows = []
            for z in range(1, 16):
                x = self._rain_inputs(key, z, feats, precip_by_zone, weather_by_zone) if z in z2c and p_ok else None
                rec = {'zone_id': z, 'hmm_pred': None, 'mrf_pred': None, 'mrf_var': None, 'accum_24h_mm': None,
                       'prob_rain': None, 'median_mm': None, 'exceedance_prob': None, 'threshold': None,
                       'is_monitored': x is not None, 'grid_cell': z2c.get(z), 'quantity': 'hourly_rate_mm_per_h',
                       'target_time_utc': tgt_ts.isoformat() if tgt_ts is not None else None,
                       'heavy_rain_alert': RAIN_HEAVY_NOT_EVALUATED,
                       'evaluation': {'probability_passed': p_ok, 'amount_passed': a_ok,
                                      'test_brier_skill': g.get('test_bss'), 'test_crps_skill': g.get('test_crpss_clim')}}
                if z not in z2c:
                    rec.update(alert_status=NO_MODEL, availability='UNAVAILABLE', reason=NO_MODEL, basis=None)
                elif not p_ok:                       # a failed horizon is unavailable for every zone
                    rec.update(alert_status=RAIN_FAILED_EVAL, availability='UNAVAILABLE', basis=None,
                               reason='Horizon did not beat the climatology/persistence baselines on validation and test')
                elif x is None:
                    rec.update(alert_status=UNAVAILABLE_NO_DATA, availability='UNAVAILABLE',
                               reason='Incomplete input (precipitation or weather, last 24 h)', basis=None)
                else:
                    pr = model.predict(x[model.feature_names], np.array([season]))
                    rec.update(availability='MODEL_FORECAST', basis=basis, reason=None,
                               alert_status='NOT_APPLICABLE_HOURLY', prob_rain=_r(pr['p_wet'][0], 3))
                    if a_ok:
                        rec.update(mrf_pred=_r(pr['mean'][0], 3), median_mm=_r(pr['median'][0], 2))
                rows.append(rec)
            out_h[key] = rows
            if key in supported and not any(r['availability'] == 'MODEL_FORECAST' for r in rows):
                supported.remove(key)                # passed evaluation, but no zone had complete inputs
        return {'origin_utc': origin, 'horizons': out_h, 'n_active': len(feats) if supported else 0,
                'enabled': bool(supported), 'source': 'open_meteo_twopart', 'supported_horizons': supported}

    def _run_rainfall_openmeteo(self, precip_by_zone: Dict[int, pd.Series],
                                origin: Optional[pd.Timestamp]) -> Dict[str, Any]:
        """
        Open-Meteo-trained per-cell HMMs driven by Open-Meteo precipitation (same source as training).
        Hourly rows ('1h'..'24h'): expected rain RATE in the target hour (mm/h).
        '24h' row additionally: accum_24h_mm = expected TOTAL over the next 24 h (alert quantity).
        A quantity whose horizon failed the stored baseline evaluation is not published.
        """
        b = self.rain_om
        gate, z2c = b['gate'], b['zone_to_cell']
        acc_ok = bool(gate.get(RAIN_OM_ACCUM_KEY, {}).get('passed'))
        seqs = {z: s.values.astype(float) for z, s in precip_by_zone.items()
                if z in z2c and s is not None and s.notna().any() and pd.notna(s.iloc[-1])}
        basis = 'HMM trained on Open-Meteo reanalysis (per grid cell), driven by Open-Meteo precipitation'
        out_h: Dict[str, List[Dict[str, Any]]] = {}
        supported: List[str] = []
        for h in RAINFALL_HORIZONS:
            key = f'{h}h'
            h_ok = bool(gate.get(key, {}).get('passed'))
            if (h_ok or (h == 24 and acc_ok)) and seqs:
                supported.append(key)
            tgt = (origin + pd.Timedelta(hours=h)).isoformat() if origin is not None else None
            rows = []
            for z in range(1, 16):
                rec = {'zone_id': z, 'hmm_pred': None, 'mrf_pred': None, 'mrf_var': None, 'accum_24h_mm': None,
                       'exceedance_prob': None, 'threshold': 64.5, 'is_monitored': z in seqs,
                       'grid_cell': z2c.get(z), 'quantity': 'hourly_rate_mm_per_h', 'target_time_utc': tgt,
                       'evaluation': {'hourly_passed': h_ok,
                                      'hourly_test_skill': gate.get(key, {}).get('test_skill'),
                                      **({'total_24h_passed': acc_ok,
                                          'total_24h_test_skill': gate.get(RAIN_OM_ACCUM_KEY, {}).get('test_skill')}
                                         if h == 24 else {})}}
                if z not in z2c:
                    rec.update(alert_status=NO_MODEL, availability='UNAVAILABLE', reason=NO_MODEL, basis=None)
                elif z not in seqs:
                    rec.update(alert_status=UNAVAILABLE_NO_DATA, availability='UNAVAILABLE',
                               reason='No precipitation input', basis=None)
                elif not (h_ok or (h == 24 and acc_ok)):
                    rec.update(alert_status=RAIN_FAILED_EVAL, availability='UNAVAILABLE', basis=None,
                               reason='Horizon did not beat the persistence/climatology baseline in evaluation')
                else:
                    m = b['models'][z2c[z]]
                    rec.update(availability='MODEL_FORECAST', basis=basis, reason=None,
                               alert_status='NOT_APPLICABLE_HOURLY')
                    if h_ok:
                        mu, var, _ = m.forecast(seqs[z], horizon=h)
                        rec.update(hmm_pred=_r(mu, 3), mrf_pred=_r(mu, 3), mrf_var=_r(var, 4))
                    if h == 24 and acc_ok:
                        g = m.get_filtered_state_distribution(seqs[z])
                        mu_acc, var_acc = compute_24h_rainfall_accumulation(g, m.transmat_, m.means_, m.variances_)
                        a = evaluate_rainfall_alert_24h(mu_acc, var_acc, is_monitored=True)
                        rec.update(accum_24h_mm=_r(mu_acc, 2), accum_24h_std=_r(np.sqrt(var_acc), 2),
                                   accum_quantity='total_next_24h_mm', alert_status=a['alert_status'],
                                   exceedance_prob=_r(a['probability_crossing'], 4))
                rows.append(rec)
            out_h[key] = rows
        return {'origin_utc': origin, 'horizons': out_h, 'n_active': len(seqs) if supported else 0,
                'enabled': bool(supported), 'source': 'open_meteo_trained', 'supported_horizons': supported}

    # ── Temperature ──────────────────────────────────────────────────────────
    def run_temperature_hourly(self, temp_by_zone: Dict[int, pd.Series],
                               origin: Optional[pd.Timestamp],
                               weather_by_zone: Optional[Dict[int, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Hourly 2 m temperature at t+h from per-horizon regressions (Open-Meteo-trained, per grid cell
        input). temp_pred_c is published only for horizons that passed validation AND test; others are
        NOT_SUPPORTED_FAILED_EVALUATION. The IMD heatwave criterion (Tmax + verified normal) is not evaluated.
        A promoted model (with stored encodings) uses the candidate inputs from the representative zone's
        weather frame (`weather_by_zone`); the others use the temperature series as before.
        """
        from src.models.temp_model import temp_features
        b = self.temp_hourly
        out_h: Dict[str, List[Dict[str, Any]]] = {}
        supported: List[str] = []
        hist = getattr(self, 'temp_history', None)
        for h in TEMP_HOURLY_HORIZONS:
            key = f'{h}h'
            # optional history package (only when enabled and only for the horizons it contains)
            use_hist = hist is not None and key in hist['models']
            src = hist if use_hist else b
            gate, z2c, encs = src['gate'], src['zone_to_cell'], (src.get('encodings') or {})
            ok = bool(gate.get(key, {}).get('passed'))
            tgt = (origin + pd.Timedelta(hours=h)).isoformat() if origin is not None else None
            model = src['models'].get(key)
            rows, any_value = [], False
            for z in range(1, 16):
                rec = {'zone_id': z, 'temp_pred_c': None, 'pred_std': None, 'exceedance_prob': None,
                       'is_monitored': False, 'grid_cell': z2c.get(z), 'target_time_utc': tgt,
                       'quantity': 'hourly_temperature_c',
                       'evaluation': {'passed': ok, 'test_skill': gate.get(key, {}).get('test_skill'),
                                      'validate_skill': gate.get(key, {}).get('validate_skill')}}
                if use_hist:
                    rec['model_package'] = 'temp_hourly_history_6h_24h (2017-2023, frozen 2024 evaluation)'
                    rec['evaluation']['test_period'] = gate.get(key, {}).get('test_period')
                x = None
                if z in z2c and ok:
                    if key in encs:
                        frame = (weather_by_zone or {}).get(src['rep_zone'][z2c[z]])
                        if frame is not None and len(frame) >= 49:
                            x = temp_candidate_row(frame, h, z2c[z], encs[key])
                    else:
                        s = temp_by_zone.get(z)
                        x = temp_features(s, h).iloc[[-1]] if s is not None and len(s) >= 25 else None
                if z not in z2c:
                    rec.update(alert_status=NO_MODEL, availability='UNAVAILABLE', reason=NO_MODEL)
                elif not ok:
                    rec.update(alert_status=RAIN_FAILED_EVAL, availability='UNAVAILABLE',
                               reason='Horizon did not beat persistence/same-hour-yesterday/climatology on validation and test')
                elif x is None or x[model.features].isna().any(axis=None):
                    rec.update(alert_status=UNAVAILABLE_NO_DATA, availability='UNAVAILABLE',
                               reason='Incomplete temperature/weather input')
                else:
                    rec.update(temp_pred_c=_r(model.predict(x[model.features])[0], 2), pred_std=_r(model.resid_std_, 2),
                               alert_status=HEATWAVE_CRITERION_NOT_EVALUATED, availability='MODEL_FORECAST',
                               basis='Hourly regression trained on Open-Meteo reanalysis, driven by Open-Meteo temperature')
                    any_value = True
                rows.append(rec)
            if any_value:
                supported.append(key)
            out_h[key] = rows
        return {'origin_utc': origin, 'last_input_day_ist': None, 'horizons': out_h,
                'supported_horizons': supported, 'source': 'open_meteo_hourly_regression'}

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
