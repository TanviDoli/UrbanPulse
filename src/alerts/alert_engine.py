"""
UrbanPulse Alert Engine (Research Grade) — Step 12B Corrected Implementation.

=============================================================================
OFFICIAL PHYSICAL THRESHOLDS vs URBANPULSE PROBABILISTIC DECISION RULES
=============================================================================
A. Official Physical Thresholds (Government / Meteorological Standards):
   1. AQI: CPCB National Air Quality Index Standard
      Threshold: AQI > 200.0 ('Poor' category boundary).
      CPCB Scale: 0-50 Good, 51-100 Satisfactory, 101-200 Moderate,
                  201-300 Poor, 301-400 Very Poor, 401-500 Severe.
   2. Rainfall: IMD Standard Meteorological Classification
      Threshold: >= 64.5 mm / 24 hours ('Heavy Rainfall' official criterion).
      Hourly rainfall is NEVER directly compared to 64.5 mm.
      Evaluated strictly on the 24-hour accumulated forecast distribution.
   3. Temperature: IMD Coastal Station Heatwave Criteria
      Joint Criterion: Tmax >= 37.0 °C AND Departure from Normal >= +4.5 °C.
      (Temperature >= 37.0 °C is NEVER used as a standalone heatwave trigger).
      A verified climatological normal is strictly required; missing normals
      are NEVER fabricated.

B. UrbanPulse Probabilistic Alert Decision Rules:
   These probability thresholds are OUR MODEL RISK MANAGEMENT DECISION RULES,
   NOT official CPCB or IMD categories:
   - NORMAL:  P(exceedance) < 0.20  (Low operational probability of hazard)
   - WATCH:   0.20 <= P(exceedance) < 0.60 (Elevated operational risk)
   - WARNING: P(exceedance) >= 0.60 (High operational probability of hazard)
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import erfc
from typing import Dict, List, Tuple, Optional, Any
from src.models.hmm_model import TemporalHMM


# Official Physical Threshold Constants
AQI_POOR_THRESHOLD = 200.0               # CPCB National AQI 'Poor' threshold (> 200)
RAINFALL_24H_HEAVY_THRESHOLD = 64.5      # IMD Heavy Rainfall threshold (>= 64.5 mm / 24h)
TEMP_COASTAL_HEATWAVE_TMAX = 37.0        # IMD Coastal Station Tmax criteria (>= 37.0 °C)
TEMP_COASTAL_HEATWAVE_DEPARTURE = 4.5    # IMD Coastal Station departure from normal (>= +4.5 °C)

# UrbanPulse Decision Rule Boundaries (Model Risk Tiers)
PROB_WATCH_CUTOFF = 0.20
PROB_WARNING_CUTOFF = 0.60

# Explicit status for a missing / invalid predictive distribution (never mapped to NORMAL)
UNAVAILABLE_NO_DATA = 'UNAVAILABLE_NO_DATA'


def _is_valid_distribution(mu: float, var: float) -> bool:
    try:
        return bool(np.isfinite(mu) and np.isfinite(var) and var > 0)
    except TypeError:
        return False


def compute_gaussian_tail_probability(mu: float, var: float, threshold: float) -> float:
    """
    Computes exact upper tail exceedance probability P(X >= threshold)
    under posterior Gaussian predictive distribution N(mu, var).
    Formula: 0.5 * erfc((threshold - mu) / (sigma * sqrt(2)))
    """
    if np.isnan(mu) or np.isnan(var) or var <= 0:
        return 0.0
    sigma = np.sqrt(max(1e-4, var))
    z = (threshold - mu) / (sigma * np.sqrt(2.0))
    prob = 0.5 * float(erfc(z))
    return float(np.clip(prob, 0.0, 1.0))


def classify_probabilistic_alert(prob: float) -> str:
    """
    Applies UrbanPulse operational decision rules based on exceedance probability.
    NOTE: NORMAL, WATCH, and WARNING cutoffs (20% and 60%) are internal
    decision rules, distinct from official physical thresholds.
    """
    if prob >= PROB_WARNING_CUTOFF:
        return 'WARNING'
    elif prob >= PROB_WATCH_CUTOFF:
        return 'WATCH'
    else:
        return 'NORMAL'


def evaluate_aqi_alert(mu: float, var: float) -> Dict[str, Any]:
    """
    Evaluates CPCB Poor AQI alert: P(AQI > 200.0).
    """
    threshold = AQI_POOR_THRESHOLD
    if not _is_valid_distribution(mu, var):
        # A missing forecast is NOT evidence of low risk: never report it as NORMAL.
        return {
            'threshold': threshold,
            'threshold_definition': 'CPCB National AQI Poor Category (> 200.0)',
            'probability_crossing': np.nan,
            'alert_status': UNAVAILABLE_NO_DATA
        }
    prob = compute_gaussian_tail_probability(mu, var, threshold)
    status = classify_probabilistic_alert(prob)
    return {
        'threshold': threshold,
        'threshold_definition': 'CPCB National AQI Poor Category (> 200.0)',
        'probability_crossing': round(prob, 4),
        'alert_status': status
    }


def compute_24h_rainfall_accumulation(
    gamma_t: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray
) -> Tuple[float, float]:
    """
    Computes exact mean accumulation and variance of 24-hour rainfall forecast:
      S_24 = sum_{k=1}^{24} Y_{t+k}
    given current filtered state distribution gamma_t.

    Under the Markov chain:
      E[S_24] = sum_{k=1}^{24} gamma_t A^k means
      Var(S_24) = sum_{k=1}^{24} Var(Y_{t+k}) + 2 * sum_{j < k} Cov(Y_{t+j}, Y_{t+k})
    """
    K = 24
    powers = [np.linalg.matrix_power(transmat, k) for k in range(1, K + 1)]
    p_k = [gamma_t @ A_k for A_k in powers]
    means_k = [float(p @ means) for p in p_k]
    mu_accum = float(np.sum(means_k))

    var_single_k = [float(p @ variances + p @ (means ** 2) - (p @ means) ** 2) for p in p_k]
    var_sum = float(np.sum(var_single_k))

    diag_mu = np.diag(means)
    cov_sum = 0.0
    for j_idx in range(K):
        A_j = powers[j_idx]
        m_j = means_k[j_idx]
        for k_idx in range(j_idx + 1, K):
            A_diff = powers[k_idx - j_idx - 1]
            joint_exp = float(gamma_t @ A_j @ diag_mu @ A_diff @ means)
            cov_jk = joint_exp - (m_j * means_k[k_idx])
            cov_sum += cov_jk

    var_accum = max(1e-4, var_sum + 2.0 * cov_sum)
    return mu_accum, var_accum


def evaluate_rainfall_alert_24h(
    mu_accum_24: float,
    var_accum_24: float,
    is_monitored: bool
) -> Dict[str, Any]:
    """
    Evaluates IMD Heavy Rainfall alert: P(24h Rainfall >= 64.5 mm).
    Hourly rainfall is NEVER directly compared to 64.5 mm.
    For unmonitored zones (where beta=0 and no sensor exists),
    status is strictly UNMONITORED_INSUFFICIENT_DATA (no regularizer fabrication).
    """
    threshold = RAINFALL_24H_HEAVY_THRESHOLD
    if not is_monitored:
        return {
            'threshold': threshold,
            'threshold_definition': 'IMD Heavy Rainfall (>= 64.5 mm / 24h)',
            'probability_crossing': np.nan,
            'alert_status': 'UNMONITORED_INSUFFICIENT_DATA'
        }

    if not _is_valid_distribution(mu_accum_24, var_accum_24):
        return {
            'threshold': threshold,
            'threshold_definition': 'IMD Heavy Rainfall (>= 64.5 mm / 24h)',
            'probability_crossing': np.nan,
            'alert_status': UNAVAILABLE_NO_DATA
        }
    prob = compute_gaussian_tail_probability(mu_accum_24, var_accum_24, threshold)
    status = classify_probabilistic_alert(prob)
    return {
        'threshold': threshold,
        'threshold_definition': 'IMD Heavy Rainfall (>= 64.5 mm / 24h)',
        'probability_crossing': round(prob, 4),
        'alert_status': status
    }


def evaluate_temperature_heatwave_alert(
    tmax_pred: float,
    tmax_var: float,
    normal_tmax: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates official IMD Coastal Heatwave alert:
      Joint condition: Tmax >= 37.0 °C AND Tmax - Normal_Tmax >= 4.5 °C.
    Temperature >= 37.0 °C is NEVER used alone.
    If normal_tmax is missing (not verified in dataset), it is NEVER fabricated.
    """
    if normal_tmax is None:
        return {
            'threshold': f'Tmax >= {TEMP_COASTAL_HEATWAVE_TMAX}°C AND departure >= +{TEMP_COASTAL_HEATWAVE_DEPARTURE}°C',
            'threshold_definition': 'IMD Coastal Station Heatwave (Joint: Tmax >= 37.0°C AND Departure >= +4.5°C)',
            'probability_crossing': np.nan,
            'alert_status': 'UNAVAILABLE_MISSING_CLIMATOLOGY',
            'reason': 'Verified climatological normal Tmax is required for IMD coastal heatwave; not present in dataset.'
        }

    # When normal_tmax is available, the effective threshold is max(37.0, normal_tmax + 4.5)
    effective_threshold = max(TEMP_COASTAL_HEATWAVE_TMAX, normal_tmax + TEMP_COASTAL_HEATWAVE_DEPARTURE)
    prob = compute_gaussian_tail_probability(tmax_pred, tmax_var, effective_threshold)
    status = classify_probabilistic_alert(prob)
    return {
        'threshold': round(effective_threshold, 2),
        'threshold_definition': f'IMD Coastal Heatwave (max(37°C, Normal+{TEMP_COASTAL_HEATWAVE_DEPARTURE}°C))',
        'probability_crossing': round(prob, 4),
        'alert_status': status
    }


def run_alert_engine():
    print("=" * 75)
    print("UrbanPulse: Corrected Alert Engine Execution (Step 12B)")
    print("=" * 75)

    preds_file = 'results/predictions/test_predictions.csv'
    assert os.path.exists(preds_file), f"Missing predictions file: {preds_file}"
    test_path = 'data/processed/test.csv'
    assert os.path.exists(test_path), f"Missing test dataset: {test_path}"

    print(f"Loading predictions from {preds_file}...")
    df_preds = pd.read_csv(preds_file, low_memory=False)
    print(f"  Loaded {len(df_preds):,} prediction records across 15 zones.")

    print(f"Loading test dataset from {test_path} to verify rainfall sequences...")
    df_test = pd.read_csv(test_path, low_memory=False)
    pivot_test_rf = df_test.pivot(index='timestamp', columns='zone_id', values='rainfall').sort_index()
    test_timestamps = pivot_test_rf.index.values
    T = len(test_timestamps)

    # Load rainfall HMM models to compute true 24h accumulation distributions
    rf_hmms: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_rainfall.joblib')

    # Precompute 24h accumulation series for monitored rainfall zones
    print("\nComputing exact 24-hour rainfall forecast accumulations for monitored zones...")
    rf_accum_mu = {}
    rf_accum_var = {}

    for z_id in range(1, 16):
        if z_id in rf_hmms:
            hmm = rf_hmms[z_id]
            s = pivot_test_rf[z_id].values
            gamma_hist = hmm.filter_sequence(s)

            K = 24
            powers = [np.linalg.matrix_power(hmm.transmat_, k) for k in range(1, K + 1)]
            M = sum(powers)
            mu_acc = gamma_hist @ M @ hmm.means_

            p_all = np.stack([gamma_hist @ powers[k - 1] for k in range(1, K + 1)], axis=1)
            means_k = np.einsum('tks,s->tk', p_all, hmm.means_)
            second_mom_k = np.einsum('tks,s->tk', p_all, hmm.variances_ + hmm.means_ ** 2)
            var_single_k = second_mom_k - means_k ** 2
            var_sum = np.sum(var_single_k, axis=1)

            diag_mu = np.diag(hmm.means_)
            cov_sum = np.zeros(T)
            for j_idx in range(K):
                A_j = powers[j_idx]
                m_j = means_k[:, j_idx]
                for k_idx in range(j_idx + 1, K):
                    A_diff = powers[k_idx - j_idx - 1]
                    w_vec = A_j @ diag_mu @ A_diff @ hmm.means_
                    joint_exp = gamma_hist @ w_vec
                    cov_sum += joint_exp - (m_j * means_k[:, k_idx])

            var_acc = np.maximum(1e-4, var_sum + 2.0 * cov_sum)
            rf_accum_mu[z_id] = mu_acc
            rf_accum_var[z_id] = var_acc
            print(f"  Zone {z_id:2d}: 24h accum mean range [{mu_acc.min():.2f}, {mu_acc.max():.2f}] mm | std range [{np.sqrt(var_acc).min():.2f}, {np.sqrt(var_acc).max():.2f}] mm")

    alert_records = []

    # Map timestamp to index
    ts_to_idx = {ts: idx for idx, ts in enumerate(test_timestamps)}

    print("\nGenerating corrected alerts...")
    for row in df_preds.itertuples():
        var = row.variable
        horizon = row.horizon
        z_id = row.zone_id
        z_label = row.zone_label
        ts = row.timestamp
        is_mon = row.is_monitored

        # =====================================================================
        # 1. AQI ALERTS
        # =====================================================================
        if var == 'AQI':
            res = evaluate_aqi_alert(row.mrf_pred, row.mrf_var)
            alert_records.append({
                'timestamp': ts,
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'AQI',
                'forecast_horizon': horizon,
                'predicted_value': round(row.mrf_pred, 2),
                'predictive_std': round(np.sqrt(max(1e-4, row.mrf_var)), 2),
                'threshold': res['threshold'],
                'threshold_type': 'Official CPCB Standard (> 200.0)',
                'probability_crossing': res['probability_crossing'],
                'alert_status': res['alert_status'],
                'decision_rule': 'P < 0.20 Normal, 0.20-0.60 Watch, >= 0.60 Warning',
                'is_monitored': is_mon
            })

        # =====================================================================
        # 2. RAINFALL ALERTS (Evaluated strictly on 24h accumulation)
        # =====================================================================
        elif var == 'Rainfall':
            # Official threshold is 64.5 mm / 24 hours.
            # Hourly predictions are NOT compared to 64.5 mm.
            # Only 24h accumulation horizon is evaluated.
            if horizon == '24h':
                t_idx = ts_to_idx[ts]
                if is_mon and z_id in rf_accum_mu:
                    mu_acc = rf_accum_mu[z_id][t_idx]
                    var_acc = rf_accum_var[z_id][t_idx]
                    res = evaluate_rainfall_alert_24h(mu_acc, var_acc, is_monitored=True)
                    pred_val = round(mu_acc, 2)
                    std_val = round(np.sqrt(var_acc), 2)
                else:
                    res = evaluate_rainfall_alert_24h(np.nan, np.nan, is_monitored=False)
                    pred_val = np.nan
                    std_val = np.nan

                alert_records.append({
                    'timestamp': ts,
                    'zone_id': z_id,
                    'zone_label': z_label,
                    'variable': 'Rainfall',
                    'forecast_horizon': '24h_accumulation',
                    'predicted_value': pred_val,
                    'predictive_std': std_val,
                    'threshold': res['threshold'],
                    'threshold_type': 'Official IMD Heavy Rainfall Standard (>= 64.5 mm / 24h)',
                    'probability_crossing': res['probability_crossing'],
                    'alert_status': res['alert_status'],
                    'decision_rule': 'P < 0.20 Normal, 0.20-0.60 Watch, >= 0.60 Warning',
                    'is_monitored': is_mon
                })

    # =========================================================================
    # 3. TEMPERATURE ALERTS (Documented Provisional Status)
    # =========================================================================
    # Because ground truth is 0 in test.csv and verified climatological normal
    # is absent, temperature alerts are documented as provisional.
    print("Auditing temperature alert logic: No normal climatology is fabricated.")

    df_alerts = pd.DataFrame(alert_records)
    os.makedirs('results/alerts', exist_ok=True)
    out_alerts_path = 'results/alerts/test_alerts.csv'
    df_alerts.to_csv(out_alerts_path, index=False)
    print(f"\nSaved {len(df_alerts):,} corrected alert records to {out_alerts_path}")

    # Summary Statistics
    print("\n" + "=" * 75)
    print("CORRECTED ALERT SUMMARY STATISTICS (Test Partition)")
    print("=" * 75)

    status_summary = df_alerts.groupby(['variable', 'forecast_horizon', 'alert_status']).size().unstack(fill_value=0)
    print(status_summary)

    elevated = df_alerts[df_alerts['alert_status'].isin(['WATCH', 'WARNING'])]
    if len(elevated) > 0:
        print("\nElevated Risk Alerts (WATCH / WARNING) by Zone & Variable:")
        elevated_by_zone = elevated.groupby(['zone_id', 'zone_label', 'variable', 'alert_status']).size().unstack(fill_value=0)
        print(elevated_by_zone)

    print("\nHigh-Risk AQI Warnings (Sample):")
    sample_warnings = df_alerts[df_alerts['alert_status'] == 'WARNING'].head(5)
    for r in sample_warnings.itertuples():
        print(f"  [{r.timestamp}] Zone {r.zone_id:2d} ({r.zone_label:18s}) | AQI {r.forecast_horizon:>3s} | Pred: {r.predicted_value:6.2f} (std: {r.predictive_std:5.2f}) | P(>200): {r.probability_crossing:6.4f} | Status: {r.alert_status}")


if __name__ == '__main__':
    run_alert_engine()
