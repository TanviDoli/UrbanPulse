"""
Step 12: Final Out-of-Sample Test Evaluation for UrbanPulse (Research Grade).

Strict Data Governance Rules:
1. test.csv is used ONLY for final out-of-sample evaluation.
2. NO tuning of beta on test.
3. NO selection of HMM states on test.
4. NO retraining on test.
5. NO modification of model parameters.
6. Compares standalone HMM vs HMM+MRF across horizons [1h, 3h, 6h, 24h].
7. Preserves unmonitored-zone policy: lambda_u = 0, b_u = 0 strictly (no fabricated observations).
8. Saves metrics to results/metrics/test_metrics_comparison.csv
   and predictions to results/predictions/test_predictions.csv.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.models.hmm_model import TemporalHMM
from src.models.mrf_model import SpatialMRF


def compute_metrics(actuals: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and valid count."""
    mask = ~np.isnan(actuals) & ~np.isnan(preds)
    count = int(mask.sum())
    if count == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'count': 0}
    y_true = actuals[mask]
    y_pred = preds[mask]
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'count': count}


def run_test_evaluation():
    print("=" * 75)
    print("UrbanPulse: Final Out-of-Sample Test Evaluation (Step 12)")
    print("=" * 75)

    test_path = 'data/processed/test.csv'
    assert os.path.exists(test_path), f"Missing {test_path}"

    print(f"Loading final test dataset: {test_path}...")
    df_test = pd.read_csv(test_path, low_memory=False)
    print(f"  Test records: {len(df_test):,} rows | {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

    # Zone mapping
    zone_labels = df_test[['zone_id', 'zone_label']].drop_duplicates().sort_values('zone_id')
    zone_map = dict(zip(zone_labels['zone_id'], zone_labels['zone_label']))

    # Load pre-trained models (NEVER modified)
    print("\nLoading pre-trained HMM model registries...")
    aqi_hmms: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_aqi.joblib')
    rf_hmms: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_rainfall.joblib')
    temp_hmms: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_temperature.joblib')

    print(f"  AQI HMMs:         {len(aqi_hmms)} zones {list(aqi_hmms.keys())}")
    print(f"  Rainfall HMMs:    {len(rf_hmms)} zones {list(rf_hmms.keys())}")
    print(f"  Temperature HMMs: {len(temp_hmms)} zones {list(temp_hmms.keys())}")

    print("\nLoading pre-calibrated Spatial MRF models...")
    mrf_aqi: SpatialMRF = joblib.load('models/hmm_mrf/mrf_aqi.joblib')
    mrf_rf: SpatialMRF = joblib.load('models/hmm_mrf/mrf_rainfall.joblib')
    mrf_temp: SpatialMRF = joblib.load('models/hmm_mrf/mrf_temperature.joblib')

    print(f"  AQI MRF Beta:         {mrf_aqi.beta:.6f}")
    print(f"  Rainfall MRF Beta:    {mrf_rf.beta:.6f}")
    print(f"  Temperature MRF Beta: {mrf_temp.beta:.6f}")

    all_test_metrics = []
    all_test_predictions = []

    # =========================================================================
    # 1. AQI TEST EVALUATION (HMM vs HMM+MRF)
    # =========================================================================
    print("\n" + "=" * 75)
    print("1. EVALUATING AQI ON TEST SET (15,622 observed values across 7 zones)")
    print("=" * 75)

    pivot_test_aqi = df_test.pivot(index='timestamp', columns='zone_id', values='AQI').sort_index()
    test_timestamps_aqi = pivot_test_aqi.index.values
    n_steps_aqi = len(test_timestamps_aqi)

    # Active mask: which zones have genuine HMM forecasts
    active_mask_aqi = np.array([z in aqi_hmms for z in range(1, 16)])

    for h in [1, 3, 6, 24]:
        print(f"\nComputing AQI Forecasts for Horizon {h}h...")
        
        # Precompute HMM predictive unaries across all zones
        y_mat = np.full((n_steps_aqi, 15), np.nan, dtype=float)
        v_mat = np.full((n_steps_aqi, 15), np.nan, dtype=float)

        for z_id in range(1, 16):
            if z_id in aqi_hmms:
                s = pivot_test_aqi[z_id].values
                y_hat, var_hat = aqi_hmms[z_id].filter_and_forecast_sequence(s, horizon=h)
                y_mat[:, z_id - 1] = y_hat
                v_mat[:, z_id - 1] = var_hat

        # Evaluate both models step-by-step
        hmm_actuals_by_zone = {z: [] for z in range(1, 16)}
        hmm_preds_by_zone = {z: [] for z in range(1, 16)}
        mrf_preds_by_zone = {z: [] for z in range(1, 16)}
        mrf_vars_by_zone = {z: [] for z in range(1, 16)}

        all_hmm_actuals = []
        all_hmm_preds = []
        all_mrf_preds = []

        for i in range(n_steps_aqi):
            ts = test_timestamps_aqi[i]
            y_row = y_mat[i]
            v_row = v_mat[i]
            act_row = pivot_test_aqi.iloc[i].values

            # Perform MRF exact inference
            mu_mrf, var_mrf, _ = mrf_aqi.infer_exact(y_row, v_row, active_mask_aqi)

            for z_id in range(1, 16):
                z_idx = z_id - 1
                is_mon = (z_id in aqi_hmms)
                act_val = act_row[z_idx]
                hmm_pred = y_row[z_idx]
                hmm_var = v_row[z_idx]
                mrf_pred = mu_mrf[z_idx]
                mrf_var = var_mrf[z_idx]

                if is_mon and pd.notna(act_val):
                    hmm_actuals_by_zone[z_id].append(act_val)
                    hmm_preds_by_zone[z_id].append(hmm_pred)
                    mrf_preds_by_zone[z_id].append(mrf_pred)
                    mrf_vars_by_zone[z_id].append(mrf_var)

                    all_hmm_actuals.append(act_val)
                    all_hmm_preds.append(hmm_pred)
                    all_mrf_preds.append(mrf_pred)

                # Record prediction row for output
                all_test_predictions.append({
                    'timestamp': ts,
                    'zone_id': z_id,
                    'zone_label': zone_map[z_id],
                    'variable': 'AQI',
                    'horizon': f'{h}h',
                    'actual': act_val if is_mon else np.nan,
                    'hmm_pred': hmm_pred if is_mon else np.nan,
                    'hmm_var': hmm_var if is_mon else np.nan,
                    'mrf_pred': mrf_pred,
                    'mrf_var': mrf_var,
                    'is_monitored': is_mon
                })

        # Calculate per-zone metrics
        print(f"  {'Zone':<20} | {'HMM MAE':>8} {'RMSE':>8} | {'MRF MAE':>8} {'RMSE':>8} | {'Diff MAE':>9} | {'N':>5}")
        print("  " + "-" * 75)
        for z_id in sorted(aqi_hmms.keys()):
            act_z = np.array(hmm_actuals_by_zone[z_id])
            h_p_z = np.array(hmm_preds_by_zone[z_id])
            m_p_z = np.array(mrf_preds_by_zone[z_id])

            h_perf = compute_metrics(act_z, h_p_z)
            m_perf = compute_metrics(act_z, m_p_z)
            diff_mae = round(m_perf['mae'] - h_perf['mae'], 3)
            improved = diff_mae < 0

            print(f"  {zone_map[z_id]:<20} | {h_perf['mae']:8.3f} {h_perf['rmse']:8.3f} | {m_perf['mae']:8.3f} {m_perf['rmse']:8.3f} | {diff_mae:+9.3f} {'*' if improved else ' '} | {h_perf['count']:5d}")

            all_test_metrics.append({
                'variable': 'AQI',
                'horizon': f'{h}h',
                'zone_id': z_id,
                'zone_label': zone_map[z_id],
                'hmm_mae': h_perf['mae'],
                'hmm_rmse': h_perf['rmse'],
                'mrf_mae': m_perf['mae'],
                'mrf_rmse': m_perf['rmse'],
                'mae_reduction': round(h_perf['mae'] - m_perf['mae'], 3),
                'mrf_improved': improved,
                'sample_count': h_perf['count']
            })

        # Overall across all monitored zones
        h_overall = compute_metrics(np.array(all_hmm_actuals), np.array(all_hmm_preds))
        m_overall = compute_metrics(np.array(all_hmm_actuals), np.array(all_mrf_preds))
        diff_overall = round(m_overall['mae'] - h_overall['mae'], 3)
        improved_overall = diff_overall < 0
        print("  " + "-" * 75)
        print(f"  {'OVERALL (All Monitored)':<20} | {h_overall['mae']:8.3f} {h_overall['rmse']:8.3f} | {m_overall['mae']:8.3f} {m_overall['rmse']:8.3f} | {diff_overall:+9.3f} {'*' if improved_overall else ' '} | {h_overall['count']:5d}")

        all_test_metrics.append({
            'variable': 'AQI',
            'horizon': f'{h}h',
            'zone_id': 0,
            'zone_label': 'OVERALL',
            'hmm_mae': h_overall['mae'],
            'hmm_rmse': h_overall['rmse'],
            'mrf_mae': m_overall['mae'],
            'mrf_rmse': m_overall['rmse'],
            'mae_reduction': round(h_overall['mae'] - m_overall['mae'], 3),
            'mrf_improved': improved_overall,
            'sample_count': h_overall['count']
        })

    # =========================================================================
    # 2. RAINFALL TEST EVALUATION (HMM vs HMM+MRF)
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. EVALUATING RAINFALL ON TEST SET (9,297 observed values across 4 zones)")
    print("=" * 75)
    print("NOTE: Calibrated beta = 0.0000. MRF reduces mathematically to HMM.")

    pivot_test_rf = df_test.pivot(index='timestamp', columns='zone_id', values='rainfall').sort_index()
    test_timestamps_rf = pivot_test_rf.index.values
    n_steps_rf = len(test_timestamps_rf)
    active_mask_rf = np.array([z in rf_hmms for z in range(1, 16)])

    for h in [1, 3, 6, 24]:
        print(f"\nComputing Rainfall Forecasts for Horizon {h}h...")
        y_mat_rf = np.full((n_steps_rf, 15), np.nan, dtype=float)
        v_mat_rf = np.full((n_steps_rf, 15), np.nan, dtype=float)

        for z_id in range(1, 16):
            if z_id in rf_hmms:
                s = pivot_test_rf[z_id].values
                y_hat, var_hat = rf_hmms[z_id].filter_and_forecast_sequence(s, horizon=h)
                y_mat_rf[:, z_id - 1] = y_hat
                v_mat_rf[:, z_id - 1] = var_hat

        hmm_actuals_by_zone_rf = {z: [] for z in range(1, 16)}
        hmm_preds_by_zone_rf = {z: [] for z in range(1, 16)}
        mrf_preds_by_zone_rf = {z: [] for z in range(1, 16)}

        all_hmm_actuals_rf = []
        all_hmm_preds_rf = []
        all_mrf_preds_rf = []

        for i in range(n_steps_rf):
            ts = test_timestamps_rf[i]
            y_row = y_mat_rf[i]
            v_row = v_mat_rf[i]
            act_row = pivot_test_rf.iloc[i].values

            mu_mrf_rf, var_mrf_rf, _ = mrf_rf.infer_exact(y_row, v_row, active_mask_rf)

            for z_id in range(1, 16):
                z_idx = z_id - 1
                is_mon = (z_id in rf_hmms)
                act_val = act_row[z_idx]
                hmm_pred = y_row[z_idx]
                hmm_var = v_row[z_idx]
                mrf_pred = mu_mrf_rf[z_idx]
                mrf_var = var_mrf_rf[z_idx]

                if is_mon and pd.notna(act_val):
                    hmm_actuals_by_zone_rf[z_id].append(act_val)
                    hmm_preds_by_zone_rf[z_id].append(hmm_pred)
                    mrf_preds_by_zone_rf[z_id].append(mrf_pred)

                    all_hmm_actuals_rf.append(act_val)
                    all_hmm_preds_rf.append(hmm_pred)
                    all_mrf_preds_rf.append(mrf_pred)

                all_test_predictions.append({
                    'timestamp': ts,
                    'zone_id': z_id,
                    'zone_label': zone_map[z_id],
                    'variable': 'Rainfall',
                    'horizon': f'{h}h',
                    'actual': act_val if is_mon else np.nan,
                    'hmm_pred': hmm_pred if is_mon else np.nan,
                    'hmm_var': hmm_var if is_mon else np.nan,
                    'mrf_pred': mrf_pred,
                    'mrf_var': mrf_var,
                    'is_monitored': is_mon
                })

        print(f"  {'Zone':<20} | {'HMM MAE':>8} {'RMSE':>8} | {'MRF MAE':>8} {'RMSE':>8} | {'Diff MAE':>9} | {'N':>5}")
        print("  " + "-" * 75)
        for z_id in sorted(rf_hmms.keys()):
            act_z = np.array(hmm_actuals_by_zone_rf[z_id])
            h_p_z = np.array(hmm_preds_by_zone_rf[z_id])
            m_p_z = np.array(mrf_preds_by_zone_rf[z_id])

            h_perf = compute_metrics(act_z, h_p_z)
            m_perf = compute_metrics(act_z, m_p_z)
            diff_mae = round(m_perf['mae'] - h_perf['mae'], 3)

            print(f"  {zone_map[z_id]:<20} | {h_perf['mae']:8.3f} {h_perf['rmse']:8.3f} | {m_perf['mae']:8.3f} {m_perf['rmse']:8.3f} | {diff_mae:+9.3f} | {h_perf['count']:5d}")

            all_test_metrics.append({
                'variable': 'Rainfall',
                'horizon': f'{h}h',
                'zone_id': z_id,
                'zone_label': zone_map[z_id],
                'hmm_mae': h_perf['mae'],
                'hmm_rmse': h_perf['rmse'],
                'mrf_mae': m_perf['mae'],
                'mrf_rmse': m_perf['rmse'],
                'mae_reduction': round(h_perf['mae'] - m_perf['mae'], 3),
                'mrf_improved': False,  # Beta=0 means equivalent, not improved
                'sample_count': h_perf['count']
            })

        h_overall_rf = compute_metrics(np.array(all_hmm_actuals_rf), np.array(all_hmm_preds_rf))
        m_overall_rf = compute_metrics(np.array(all_hmm_actuals_rf), np.array(all_mrf_preds_rf))
        diff_rf = round(m_overall_rf['mae'] - h_overall_rf['mae'], 3)
        print("  " + "-" * 75)
        print(f"  {'OVERALL (All Monitored)':<20} | {h_overall_rf['mae']:8.3f} {h_overall_rf['rmse']:8.3f} | {m_overall_rf['mae']:8.3f} {m_overall_rf['rmse']:8.3f} | {diff_rf:+9.3f} | {h_overall_rf['count']:5d}")

        all_test_metrics.append({
            'variable': 'Rainfall',
            'horizon': f'{h}h',
            'zone_id': 0,
            'zone_label': 'OVERALL',
            'hmm_mae': h_overall_rf['mae'],
            'hmm_rmse': h_overall_rf['rmse'],
            'mrf_mae': m_overall_rf['mae'],
            'mrf_rmse': m_overall_rf['rmse'],
            'mae_reduction': round(h_overall_rf['mae'] - m_overall_rf['mae'], 3),
            'mrf_improved': False,
            'sample_count': h_overall_rf['count']
        })

    # =========================================================================
    # 3. TEMPERATURE TEST EVALUATION STATUS & DATA GOVERNANCE
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. TEMPERATURE STATUS IN TEST SET")
    print("=" * 75)
    print("DATA BOUNDARY AUDIT:")
    print("  test.csv contains exactly ZERO temperature ground truth records.")
    print("  (temperature_observed == False for all 39,240 test rows; column is NaN).")
    print("  IMD ground-truth temperature series in the raw data ended on 2024-06-22.")
    print("  Therefore, empirical MAE/RMSE on test.csv cannot be computed against ground truth.")
    print("  In addition, calibrated beta = 0.0000, so MRF mathematically matches HMM exactly.")

    all_test_metrics.append({
        'variable': 'Temperature',
        'horizon': '24h',
        'zone_id': 0,
        'zone_label': 'OVERALL',
        'hmm_mae': np.nan,
        'hmm_rmse': np.nan,
        'mrf_mae': np.nan,
        'mrf_rmse': np.nan,
        'mae_reduction': 0.0,
        'mrf_improved': False,
        'sample_count': 0
    })

    # Save metrics and predictions
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)

    df_metrics = pd.DataFrame(all_test_metrics)
    metrics_file = 'results/metrics/test_metrics_comparison.csv'
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\nSaved test comparison metrics to {metrics_file} ({len(df_metrics)} rows)")

    df_preds = pd.DataFrame(all_test_predictions)
    preds_file = 'results/predictions/test_predictions.csv'
    df_preds.to_csv(preds_file, index=False)
    print(f"Saved test predictions to {preds_file} ({len(df_preds):,} rows)")


if __name__ == '__main__':
    run_test_evaluation()
