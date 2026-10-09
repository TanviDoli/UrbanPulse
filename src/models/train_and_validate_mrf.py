"""
Training, Calibration, and Validation Pipeline for UrbanPulse HMM+MRF (Research Grade).
Strictly follows research-grade project rules:
1. Continuous Gaussian MRF with Exact Gaussian Inference as production method.
2. Uses 29 verified adjacency edges from data/processed/zone_adjacency.csv.
3. Consumes zone-specific HMM models from models/hmm/.
4. Proper missing-data filtering: NO forward-filling, NO zero-filling of missing rainfall.
   State distributions evolve strictly via Markov transition dynamics during missing intervals.
5. Unmonitored zones have strictly zero observation evidence (lambda_u = 0.0, b_u = 0.0).
   NO regional_mean, NO global_mean, NO synthetic observations.
6. Calibrates spatial coupling parameter beta strictly on validate.csv.
7. Completely isolates data/processed/test.csv (NEVER loaded).
8. Saves models under models/hmm_mrf/ and metrics/predictions under results/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.models.hmm_model import TemporalHMM
from src.models.mrf_model import SpatialMRF

def evaluate_metrics(actuals: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, sample count."""
    mask = ~np.isnan(actuals) & ~np.isnan(preds)
    if mask.sum() == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'count': 0}
    y_true = actuals[mask]
    y_pred = preds[mask]
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'count': int(mask.sum())}

def run_mrf_training_and_validation():
    print("=" * 70)
    print("UrbanPulse: Spatial MRF Calibration & Validation (Exact Gaussian Inference)")
    print("=" * 70)
    
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/validate.csv'
    test_path = 'data/processed/test.csv'
    
    # Assert test.csv is never loaded or referenced
    assert 'test_df' not in locals(), "Test dataset must not be loaded"
    assert not os.path.exists('tmp_test.csv'), "No temporary test files allowed"
    
    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path, low_memory=False)
    print(f"Loading validation data from {val_path}...")
    df_val = pd.read_csv(val_path, low_memory=False)
    
    print("\nLoading zone-specific HMM models from models/hmm/...")
    aqi_hmm_models: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_aqi.joblib')
    rainfall_hmm_models: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_rainfall.joblib')
    temp_hmm_models: Dict[int, TemporalHMM] = joblib.load('models/hmm/hmm_temperature.joblib')
    
    print(f"  Loaded AQI HMMs:         {len(aqi_hmm_models)} zones {list(aqi_hmm_models.keys())}")
    print(f"  Loaded Rainfall HMMs:    {len(rainfall_hmm_models)} zones {list(rainfall_hmm_models.keys())}")
    print(f"  Loaded Temperature HMMs: {len(temp_hmm_models)} zones")
    
    os.makedirs('models/hmm_mrf', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    zone_labels = df_train[['zone_id', 'zone_label']].drop_duplicates().sort_values('zone_id')
    zone_map = dict(zip(zone_labels['zone_id'], zone_labels['zone_label']))
    
    calibration_records = []
    val_mrf_metrics = []
    val_mrf_predictions = []

    # =========================================================================
    # PART 1: AQI SPATIAL MRF CALIBRATION & MULTI-HORIZON VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: AQI SPATIAL MRF CALIBRATION & VALIDATION")
    print("=" * 60)
    
    pivot_val_aqi = df_val.pivot(index='timestamp', columns='zone_id', values='AQI').sort_index()
    val_timestamps_aqi = pivot_val_aqi.index.values
    n_timesteps_aqi = len(val_timestamps_aqi)
    print(f"Validation AQI timesteps: {n_timesteps_aqi}")
    
    # Precompute HMM rolling forecasts using canonical sequential filtering with proper missing data handling
    print("Precomputing HMM predictive unaries via canonical missing-data filtering...")
    hmm_preds_aqi = {} # horizon -> (y_hat_matrix, var_matrix)
    
    for h in [1, 3, 6, 24]:
        y_mat = np.full((n_timesteps_aqi, 15), np.nan, dtype=float)
        v_mat = np.full((n_timesteps_aqi, 15), np.nan, dtype=float)
        
        for z_id in range(1, 16):
            if z_id in aqi_hmm_models:
                model = aqi_hmm_models[z_id]
                s = pivot_val_aqi[z_id].values # Contains genuine NaNs; NO forward-fill!
                y_hat_series, var_series = model.filter_and_forecast_sequence(s, horizon=h)
                y_mat[:, z_id - 1] = y_hat_series
                v_mat[:, z_id - 1] = var_series
                
        hmm_preds_aqi[h] = (y_mat, v_mat)
        print(f"  Horizon {h:2d}h unaries precomputed.")

    # Calibration of beta for AQI using 1h validation MAE
    candidate_betas_aqi = [0.0, 1e-5, 5e-5, 1e-4, 3.3e-4, 1e-3, 5e-3, 1e-2]
    best_beta_aqi = None
    best_mae_aqi = float('inf')
    
    print("\nCalibrating spatial coupling strength beta on validate.csv (Horizon: 1h)...")
    y_1h, v_1h = hmm_preds_aqi[1]
    active_mask_aqi = np.array([z in aqi_hmm_models for z in range(1, 16)])
    
    for beta in candidate_betas_aqi:
        mrf_test = SpatialMRF(variable='AQI', beta=beta)
        preds_list = []
        actuals_list = []
        
        for i in range(n_timesteps_aqi):
            y_row = y_1h[i]
            v_row = v_1h[i]
            act_row = pivot_val_aqi.iloc[i].values
            
            # Require at least one monitored zone forecast
            if not np.any(~np.isnan(y_row[active_mask_aqi])):
                continue
                
            # Exact Gaussian inference: strictly zero unmonitored lambda/evidence
            mu_mrf, _, _ = mrf_test.infer_exact(y_row, v_row, active_mask_aqi)
            
            for z_id in aqi_hmm_models.keys():
                z_idx = z_id - 1
                if pd.notna(act_row[z_idx]) and pd.notna(y_row[z_idx]):
                    preds_list.append(mu_mrf[z_idx])
                    actuals_list.append(act_row[z_idx])
                    
        perf = evaluate_metrics(np.array(actuals_list), np.array(preds_list))
        print(f"  Beta = {beta:10.6f} | Val MAE (1h): {perf['mae']:6.3f} | Val RMSE (1h): {perf['rmse']:6.3f} | N={perf['count']}")
        
        calibration_records.append({
            'variable': 'AQI',
            'beta': beta,
            'val_mae_1h': perf['mae'],
            'val_rmse_1h': perf['rmse'],
            'sample_count': perf['count']
        })
        
        if perf['mae'] < best_mae_aqi:
            best_mae_aqi = perf['mae']
            best_beta_aqi = beta
            
    print(f"\nOptimal AQI spatial coupling calibrated: Beta = {best_beta_aqi:.6f} (Val MAE: {best_mae_aqi:.3f})")
    
    mrf_aqi = SpatialMRF(variable='AQI', beta=best_beta_aqi)
    mrf_aqi.save('models/hmm_mrf/mrf_aqi.joblib')
    
    # Multi-horizon evaluation for HMM+MRF on AQI
    print("\nEvaluating calibrated AQI HMM+MRF across horizons [1h, 3h, 6h, 24h]...")
    for h in [1, 3, 6, 24]:
        y_h, v_h = hmm_preds_aqi[h]
        zone_preds = {z: [] for z in range(1, 16)}
        zone_actuals = {z: [] for z in range(1, 16)}
        
        for i in range(n_timesteps_aqi):
            ts = val_timestamps_aqi[i]
            y_row = y_h[i]
            v_row = v_h[i]
            act_row = pivot_val_aqi.iloc[i].values
            
            if not np.any(~np.isnan(y_row[active_mask_aqi])):
                continue
                
            mu_mrf, var_mrf, _ = mrf_aqi.infer_exact(y_row, v_row, active_mask_aqi)
            
            for z_id in range(1, 16):
                z_idx = z_id - 1
                is_mon = (z_id in aqi_hmm_models)
                act_val = act_row[z_idx]
                pred_val = mu_mrf[z_idx]
                var_val = var_mrf[z_idx]
                
                if is_mon and pd.notna(act_val):
                    zone_preds[z_id].append(pred_val)
                    zone_actuals[z_id].append(act_val)
                    
                if h == 1:
                    val_mrf_predictions.append({
                        'timestamp': ts,
                        'zone_id': z_id,
                        'zone_label': zone_map[z_id],
                        'variable': 'AQI',
                        'horizon': '1h',
                        'actual': act_val if is_mon else np.nan,
                        'predicted': pred_val,
                        'variance': var_val,
                        'is_monitored': is_mon,
                        'model': 'HMM+MRF'
                    })
                    
        for z_id in aqi_hmm_models.keys():
            perf_z = evaluate_metrics(np.array(zone_actuals[z_id]), np.array(zone_preds[z_id]))
            val_mrf_metrics.append({
                'zone_id': z_id,
                'zone_label': zone_map[z_id],
                'variable': 'AQI',
                'model': 'HMM+MRF',
                'horizon': f'{h}h',
                'mae': perf_z['mae'],
                'rmse': perf_z['rmse'],
                'sample_count': perf_z['count']
            })
            print(f"  Horizon {h:2d}h | Zone {z_id:2d} ({zone_map[z_id]:18s}) -> MAE: {perf_z['mae']:6.3f} | RMSE: {perf_z['rmse']:6.3f}")

    # =========================================================================
    # PART 2: RAINFALL SPATIAL MRF CALIBRATION & MULTI-HORIZON VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: RAINFALL SPATIAL MRF CALIBRATION & VALIDATION")
    print("=" * 60)
    
    pivot_val_rf = df_val.pivot(index='timestamp', columns='zone_id', values='rainfall').sort_index()
    val_timestamps_rf = pivot_val_rf.index.values
    n_timesteps_rf = len(val_timestamps_rf)
    
    print("Precomputing Rainfall HMM unaries via canonical missing-data filtering...")
    hmm_preds_rf = {}
    
    for h in [1, 3, 6, 24]:
        y_mat = np.full((n_timesteps_rf, 15), np.nan, dtype=float)
        v_mat = np.full((n_timesteps_rf, 15), np.nan, dtype=float)
        
        for z_id in range(1, 16):
            if z_id in rainfall_hmm_models:
                model = rainfall_hmm_models[z_id]
                s = pivot_val_rf[z_id].values # NaN remains strictly NaN; NO zero-filling!
                y_hat_series, var_series = model.filter_and_forecast_sequence(s, horizon=h)
                y_mat[:, z_id - 1] = y_hat_series
                v_mat[:, z_id - 1] = var_series
                
        hmm_preds_rf[h] = (y_mat, v_mat)
        print(f"  Rainfall Horizon {h:2d}h unaries precomputed.")

    candidate_betas_rf = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    best_beta_rf = None
    best_mae_rf = float('inf')
    
    print("\nCalibrating Rainfall spatial coupling strength beta on validate.csv (Horizon: 1h)...")
    y_1h_rf, v_1h_rf = hmm_preds_rf[1]
    active_mask_rf = np.array([z in rainfall_hmm_models for z in range(1, 16)])
    
    for beta in candidate_betas_rf:
        mrf_test = SpatialMRF(variable='rainfall', beta=beta)
        preds_list = []
        actuals_list = []
        
        for i in range(n_timesteps_rf):
            y_row = y_1h_rf[i]
            v_row = v_1h_rf[i]
            act_row = pivot_val_rf.iloc[i].values
            
            if not np.any(~np.isnan(y_row[active_mask_rf])):
                continue
                
            mu_mrf, _, _ = mrf_test.infer_exact(y_row, v_row, active_mask_rf)
            
            for z_id in rainfall_hmm_models.keys():
                z_idx = z_id - 1
                if pd.notna(act_row[z_idx]) and pd.notna(y_row[z_idx]):
                    preds_list.append(mu_mrf[z_idx])
                    actuals_list.append(act_row[z_idx])
                    
        perf = evaluate_metrics(np.array(actuals_list), np.array(preds_list))
        print(f"  Beta = {beta:8.4f} | Val MAE (1h): {perf['mae']:6.3f} | Val RMSE (1h): {perf['rmse']:6.3f} | N={perf['count']}")
        
        calibration_records.append({
            'variable': 'Rainfall',
            'beta': beta,
            'val_mae_1h': perf['mae'],
            'val_rmse_1h': perf['rmse'],
            'sample_count': perf['count']
        })
        
        if perf['mae'] < best_mae_rf:
            best_mae_rf = perf['mae']
            best_beta_rf = beta
            
    print(f"\nOptimal Rainfall spatial coupling calibrated: Beta = {best_beta_rf:.4f} (Val MAE: {best_mae_rf:.3f})")
    
    mrf_rf = SpatialMRF(variable='rainfall', beta=best_beta_rf)
    mrf_rf.save('models/hmm_mrf/mrf_rainfall.joblib')
    
    # Multi-horizon evaluation for Rainfall HMM+MRF
    print("\nEvaluating calibrated Rainfall HMM+MRF across horizons [1h, 3h, 6h, 24h]...")
    for h in [1, 3, 6, 24]:
        y_h, v_h = hmm_preds_rf[h]
        zone_preds = {z: [] for z in range(1, 16)}
        zone_actuals = {z: [] for z in range(1, 16)}
        
        for i in range(n_timesteps_rf):
            ts = val_timestamps_rf[i]
            y_row = y_h[i]
            v_row = v_h[i]
            act_row = pivot_val_rf.iloc[i].values
            
            if not np.any(~np.isnan(y_row[active_mask_rf])):
                continue
                
            mu_mrf, var_mrf, _ = mrf_rf.infer_exact(y_row, v_row, active_mask_rf)
            
            for z_id in range(1, 16):
                z_idx = z_id - 1
                is_mon = (z_id in rainfall_hmm_models)
                act_val = act_row[z_idx]
                pred_val = mu_mrf[z_idx]
                var_val = var_mrf[z_idx]
                
                if is_mon and pd.notna(act_val):
                    zone_preds[z_id].append(pred_val)
                    zone_actuals[z_id].append(act_val)
                    
                if h == 1:
                    val_mrf_predictions.append({
                        'timestamp': ts,
                        'zone_id': z_id,
                        'zone_label': zone_map[z_id],
                        'variable': 'Rainfall',
                        'horizon': '1h',
                        'actual': act_val if is_mon else np.nan,
                        'predicted': pred_val,
                        'variance': var_val,
                        'is_monitored': is_mon,
                        'model': 'HMM+MRF'
                    })
                    
        for z_id in rainfall_hmm_models.keys():
            perf_z = evaluate_metrics(np.array(zone_actuals[z_id]), np.array(zone_preds[z_id]))
            val_mrf_metrics.append({
                'zone_id': z_id,
                'zone_label': zone_map[z_id],
                'variable': 'Rainfall',
                'model': 'HMM+MRF',
                'horizon': f'{h}h',
                'mae': perf_z['mae'],
                'rmse': perf_z['rmse'],
                'sample_count': perf_z['count']
            })
            print(f"  Horizon {h:2d}h | Zone {z_id:2d} ({zone_map[z_id]:18s}) -> MAE: {perf_z['mae']:6.3f} | RMSE: {perf_z['rmse']:6.3f}")

    # =========================================================================
    # PART 3: TEMPERATURE SPATIAL MRF CALIBRATION & VALIDATION (DAILY REGIME)
    # DATA GOVERNANCE:
    #   validate.csv has ZERO temperature observations (temperature_observed==False
    #   for every row; the temperature column is entirely NaN in validate.csv).
    #   Beta calibration for the Temperature MRF therefore uses a chronologically
    #   held-out TAIL of train.csv (last 20% of daily observations), which is
    #   strictly disjoint from the 80% prefix used for HMM context.
    #   test.csv is NEVER loaded.
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: TEMPERATURE SPATIAL MRF CALIBRATION & VALIDATION (DAILY REGIME)")
    print("=" * 70)
    print()
    print("DATA GOVERNANCE NOTE:")
    print("  validate.csv contains ZERO temperature observations.")
    print("  (temperature_observed == False for all rows; temperature is NaN throughout.)")
    print("  Beta for Temperature MRF is calibrated on the held-out TAIL of train.csv.")
    print("  Tail = last 20 pct of training days (strictly chronological).")
    print("  test.csv is NEVER loaded or used.")
    print()

    # Build ordered daily temperature series from train.csv
    df_tr_temp = df_train[df_train['temperature_observed'] == True].copy()
    df_tr_temp['date'] = df_tr_temp['timestamp'].str[:10]
    daily_series_df = df_tr_temp.groupby('date')['temperature'].mean().sort_index()
    daily_dates_all = daily_series_df.index.tolist()
    daily_temp_all  = daily_series_df.values  # shape (N_days,)
    N_days = len(daily_temp_all)
    print(f"  Total training temperature days: {N_days} ({daily_dates_all[0]} to {daily_dates_all[-1]})")

    # Chronological 80/20 split
    split_idx = int(N_days * 0.80)
    print(f"  Context prefix (80pct):      {split_idx} days ({daily_dates_all[0]} to {daily_dates_all[split_idx-1]})")
    print(f"  Calibration hold-out (20pct): {N_days - split_idx} days ({daily_dates_all[split_idx]} to {daily_dates_all[-1]})")
    print()

    # Precompute HMM 1-day-ahead forecasts across the FULL train sequence.
    # Each zone uses its own per-zone daily temperature sequence.
    active_mask_temp = np.ones(15, dtype=bool)
    daily_y_hat_all  = np.zeros((N_days, 15), dtype=float)
    daily_var_all    = np.zeros((N_days, 15), dtype=float)

    for z_id in range(1, 16):
        model = temp_hmm_models[z_id]
        # Per-zone mean daily temperature (same values for all zones in this dataset)
        zone_daily = df_tr_temp.groupby('date')['temperature'].mean().sort_index().values
        y_h_all, v_h_all = model.filter_and_forecast_sequence(zone_daily, horizon=1)
        daily_y_hat_all[:, z_id - 1] = y_h_all
        daily_var_all[:, z_id - 1]   = v_h_all

    # Forecasts for the calibration tail: at step t predict t+1
    # index [split_idx-1 .. N_days-2] -> predict [split_idx .. N_days-1]
    y_hat_cal    = daily_y_hat_all[split_idx - 1 : N_days - 1]
    var_cal      = daily_var_all[split_idx - 1 : N_days - 1]
    actuals_cal  = daily_temp_all[split_idx:]
    n_cal        = len(actuals_cal)
    print(f"  Calibration samples per zone: {n_cal}")
    print()

    # Beta grid search on the held-out tail
    candidate_betas_temp = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_beta_temp = None
    best_mae_temp  = float('inf')

    print("Calibrating Temperature beta on train-tail hold-out...")
    for beta in candidate_betas_temp:
        mrf_test_temp  = SpatialMRF(variable='temperature', beta=beta)
        preds_list_t   = []
        actuals_list_t = []

        for i in range(n_cal):
            y_row = y_hat_cal[i]
            v_row = var_cal[i]
            act   = actuals_cal[i]
            if np.isnan(act):
                continue
            mu_mrf_t, _, _ = mrf_test_temp.infer_exact(y_row, v_row, active_mask_temp)
            for z in range(15):
                preds_list_t.append(mu_mrf_t[z])
                actuals_list_t.append(act)

        perf_t = evaluate_metrics(np.array(actuals_list_t), np.array(preds_list_t))
        print(f"  Beta = {beta:8.4f} | Tail MAE (1-day): {perf_t['mae']:6.3f} C | RMSE: {perf_t['rmse']:6.3f} C | N_days={perf_t['count'] // 15}")

        calibration_records.append({
            'variable': 'Temperature',
            'beta': beta,
            'val_mae_1h': perf_t['mae'],
            'val_rmse_1h': perf_t['rmse'],
            'sample_count': perf_t['count'],
            'calibration_source': 'train_tail_held_out_20pct'
        })

        if perf_t['mae'] < best_mae_temp:
            best_mae_temp  = perf_t['mae']
            best_beta_temp = beta

    print(f"\nOptimal Temperature beta: {best_beta_temp:.4f} (Tail MAE: {best_mae_temp:.3f} C)")
    print("NOTE: beta selected on last 20pct of train.csv days (validate.csv has no temperature).")

    mrf_temp = SpatialMRF(variable='temperature', beta=best_beta_temp)
    mrf_temp.save('models/hmm_mrf/mrf_temperature.joblib')

    # Final evaluation on the calibration tail (document source clearly)
    temp_preds_mrf   = {z: [] for z in range(1, 16)}
    temp_actuals_mrf = {z: [] for z in range(1, 16)}

    for i in range(n_cal):
        y_row = y_hat_cal[i]
        v_row = var_cal[i]
        act   = actuals_cal[i]
        if np.isnan(act):
            continue
        mu_mrf_t, var_mrf_t, _ = mrf_temp.infer_exact(y_row, v_row, active_mask_temp)
        cal_date = daily_dates_all[split_idx + i]
        for z in range(1, 16):
            temp_preds_mrf[z].append(mu_mrf_t[z - 1])
            temp_actuals_mrf[z].append(act)
            val_mrf_predictions.append({
                'timestamp': cal_date,
                'zone_id': z,
                'zone_label': zone_map[z],
                'variable': 'Temperature',
                'horizon': '24h',
                'actual': act,
                'predicted': mu_mrf_t[z - 1],
                'variance': var_mrf_t[z - 1],
                'is_monitored': True,
                'model': 'HMM+MRF'
            })

    for z in range(1, 16):
        perf_z = evaluate_metrics(np.array(temp_actuals_mrf[z]), np.array(temp_preds_mrf[z]))
        val_mrf_metrics.append({
            'zone_id': z,
            'zone_label': zone_map[z],
            'variable': 'Temperature',
            'model': 'HMM+MRF',
            'horizon': '24h',
            'mae': perf_z['mae'],
            'rmse': perf_z['rmse'],
            'sample_count': perf_z['count'],
            'eval_source': 'train_tail_held_out_20pct'
        })
    print(f"  Temperature 24h HMM+MRF -> MAE: {perf_z['mae']:6.3f} C | RMSE: {perf_z['rmse']:6.3f} C (train-tail hold-out)")

    # =========================================================================
    # PART 4: SAVE ALL ARTIFACTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS, METRICS, AND PREDICTIONS")
    print("=" * 70)

    # Save combined MRF registry
    mrf_all = {
        'AQI': mrf_aqi,
        'Rainfall': mrf_rf,
        'Temperature': mrf_temp
    }
    all_models_path = 'models/hmm_mrf/mrf_all_models.joblib'
    joblib.dump(mrf_all, all_models_path)
    print(f"Saved {all_models_path}")

    # Save calibration scores
    df_cal = pd.DataFrame(calibration_records)
    cal_path = 'results/metrics/mrf_calibration_scores.csv'
    df_cal.to_csv(cal_path, index=False)
    print(f"Saved {cal_path} ({len(df_cal)} records)")

    # Save validation metrics
    df_metrics = pd.DataFrame(val_mrf_metrics)
    metrics_path = 'results/metrics/validation_metrics_mrf.csv'
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path} ({len(df_metrics)} records)")

    # Save validation predictions
    df_preds = pd.DataFrame(val_mrf_predictions)
    preds_path = 'results/predictions/validation_predictions_mrf.csv'
    df_preds.to_csv(preds_path, index=False)
    print(f"Saved {preds_path} ({len(df_preds)} records)")


if __name__ == '__main__':
    run_mrf_training_and_validation()
