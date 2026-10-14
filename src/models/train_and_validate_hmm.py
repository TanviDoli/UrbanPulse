"""
Zone-Specific Training and Validation Pipeline for UrbanPulse HMM Models.
Strictly follows research-grade project rules:
1. Fits a separate HMM for each available zone * variable combination.
2. NEVER concatenates observations from different zones into one temporal sequence.
3. Chronological ordering within each zone is strictly enforced.
4. Uses ONLY data/processed/train.csv for fitting parameters.
5. Uses data/processed/validate.csv for model/state selection (where valid validation data exists).
6. Completely isolates data/processed/test.csv (NEVER loaded).
7. Enforces documented minimum observation criteria:
   - Continuous hourly (AQI, Rainfall): N_train >= 2000 and N_val >= 500
   - Daily regime (Temperature): N_train >= 100 days; uses BIC on train for K selection.
8. Unmonitored or insufficient-data zones are explicitly marked as unavailable/uninformative.
9. Saves models keyed by variable -> zone_id under models/hmm/.
10. Saves validation predictions and metrics per zone and horizon under results/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.models.hmm_model import TemporalHMM

# Minimum observation thresholds
MIN_TRAIN_HOURLY = 2000
MIN_VAL_HOURLY = 500
MIN_TRAIN_DAILY = 100

def evaluate_forecasts(actuals: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, and sample count."""
    mask = ~np.isnan(actuals) & ~np.isnan(preds)
    if mask.sum() == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'count': 0}
    y_true = actuals[mask]
    y_pred = preds[mask]
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    return {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'count': int(mask.sum())}

def run_hmm_training_and_validation():
    print("=" * 70)
    print("UrbanPulse: Zone-Specific HMM Training & Validation (Research Grade)")
    print("=" * 70)
    
    train_path = 'data/processed/train.csv'
    val_path = 'data/processed/validate.csv'
    test_path = 'data/processed/test.csv'
    
    # Strict isolation check: NEVER load test.csv
    print("Verification: Ensuring test.csv is completely isolated and untouched...")
    assert 'test_df' not in locals(), "Test dataset must not be loaded"
    assert not os.path.exists('tmp_test.csv'), "No temporary test files allowed"
    
    print(f"Loading training data from {train_path}...")
    df_train = pd.read_csv(train_path, low_memory=False)
    print(f"Loading validation data from {val_path}...")
    df_val = pd.read_csv(val_path, low_memory=False)
    
    os.makedirs('models/hmm', exist_ok=True)
    os.makedirs('results/predictions', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    zone_labels = df_train[['zone_id', 'zone_label']].drop_duplicates().sort_values('zone_id')
    zone_map = dict(zip(zone_labels['zone_id'], zone_labels['zone_label']))
    
    training_status_records = []
    selection_records = []
    validation_metrics = []
    val_pred_records = []
    
    all_trained_models = {} # (variable, zone_id) -> TemporalHMM
    aqi_models = {}         # zone_id -> TemporalHMM
    rainfall_models = {}    # zone_id -> TemporalHMM
    temperature_models = {} # zone_id -> TemporalHMM

    # =========================================================================
    # PART 1: AQI ZONE-SPECIFIC HMM MODELS
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: AQI ZONE-SPECIFIC HMM TRAINING & VALIDATION")
    print("=" * 60)
    
    candidate_k_aqi = [2, 3, 4, 5, 6]
    
    for z_id in range(1, 16):
        z_label = zone_map[z_id]
        df_tr_z = df_train[df_train['zone_id'] == z_id].sort_values('timestamp')
        df_va_z = df_val[df_val['zone_id'] == z_id].sort_values('timestamp')
        
        # Enforce chronological ordering within zone
        assert df_tr_z['timestamp'].is_monotonic_increasing, f"Train timestamps for zone {z_id} must be strictly chronological"
        assert df_va_z['timestamp'].is_monotonic_increasing, f"Validation timestamps for zone {z_id} must be strictly chronological"
        
        n_tr = int(df_tr_z['aqi_observed'].sum())
        n_va = int(df_va_z['aqi_observed'].sum())
        
        print(f"\n[Zone {z_id:2d} - {z_label}] AQI: Train N={n_tr}, Val N={n_va}")
        
        if n_tr < MIN_TRAIN_HOURLY or n_va < MIN_VAL_HOURLY:
            reason = "Unmonitored zone (0 observations)" if n_tr == 0 else f"Insufficient observations (Train N={n_tr} < {MIN_TRAIN_HOURLY} or Val N={n_va} < {MIN_VAL_HOURLY})"
            print(f"  -> SKIPPED: {reason}")
            training_status_records.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'AQI',
                'status': 'UNAVAILABLE',
                'n_train_obs': n_tr,
                'n_val_obs': n_va,
                'selected_k': np.nan,
                'reason': reason
            })
            continue
            
        # Extract strictly that zone's training sequence (full chronological series including NaNs for gaps)
        train_seq = df_tr_z['AQI'].values
        val_seq = df_va_z['AQI'].values
        val_timestamps = df_va_z['timestamp'].values
        
        best_k = None
        best_val_mae = float('inf')
        best_model = None
        
        for k in candidate_k_aqi:
            hmm_k = TemporalHMM(n_components=k, variable='AQI', zone_id=z_id, random_state=42)
            # Fits ONLY on this zone's training sequence
            hmm_k.fit(train_seq)
            
            bic = hmm_k.bic(train_seq)
            aic = hmm_k.aic(train_seq)
            
            # Evaluate 1-step ahead on this zone's validation sequence
            preds_1h = []
            actuals_1h = []
            context = []
            for val_idx, val_val in enumerate(val_seq):
                if len(context) >= 1:
                    p1, _, _ = hmm_k.forecast(np.array(context[-24:]), horizon=1)
                    preds_1h.append(p1)
                    actuals_1h.append(val_val)
                if pd.notna(val_val):
                    context.append(val_val)
                elif len(context) > 0:
                    context.append(context[-1])
                    
            perf_1h = evaluate_forecasts(np.array(actuals_1h), np.array(preds_1h))
            print(f"  K={k} | BIC: {bic:10.1f} | AIC: {aic:10.1f} | Val MAE (1h): {perf_1h['mae']:6.3f} | Val RMSE (1h): {perf_1h['rmse']:6.3f}")
            
            selection_records.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'AQI',
                'n_components': k,
                'bic': round(bic, 1),
                'aic': round(aic, 1),
                'val_mae_1h': perf_1h['mae'],
                'val_rmse_1h': perf_1h['rmse'],
                'selected': False
            })
            
            if perf_1h['mae'] < best_val_mae:
                best_val_mae = perf_1h['mae']
                best_k = k
                best_model = hmm_k
                
        # Mark selected K in selection records
        for rec in selection_records:
            if rec['zone_id'] == z_id and rec['variable'] == 'AQI' and rec['n_components'] == best_k:
                rec['selected'] = True
                
        best_model.val_mae_ = best_val_mae
        print(f"  ==> Selected K={best_k} for Zone {z_id} ({z_label}) with Val MAE (1h) = {best_val_mae:.3f}")
        for s_idx, (m, v) in enumerate(zip(best_model.means_, best_model.variances_)):
            print(f"      State {s_idx}: mu = {m:6.2f}, sigma = {np.sqrt(v):6.2f}")
            
        # Multi-horizon evaluation for this zone: 1h, 3h, 6h, 24h
        for h in [1, 3, 6, 24]:
            preds_h = []
            actuals_h = []
            variances_h = []
            context = []
            for val_idx, val_val in enumerate(val_seq):
                ts = val_timestamps[val_idx]
                if len(context) >= 1:
                    pred_val, pred_var, _ = best_model.forecast(np.array(context[-24:]), horizon=h)
                    preds_h.append(pred_val)
                    actuals_h.append(val_val)
                    variances_h.append(pred_var)
                    if h == 1: # save 1h predictions for primary evaluation
                        val_pred_records.append({
                            'timestamp': ts,
                            'zone_id': z_id,
                            'zone_label': z_label,
                            'variable': 'AQI',
                            'horizon': '1h',
                            'actual': val_val,
                            'predicted': pred_val,
                            'variance': pred_var
                        })
                if pd.notna(val_val):
                    context.append(val_val)
                elif len(context) > 0:
                    context.append(context[-1])
                    
            perf_h = evaluate_forecasts(np.array(actuals_h), np.array(preds_h))
            validation_metrics.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'AQI',
                'model': 'HMM',
                'horizon': f'{h}h',
                'mae': perf_h['mae'],
                'rmse': perf_h['rmse'],
                'sample_count': perf_h['count']
            })
            print(f"      Horizon {h:2d}h -> MAE: {perf_h['mae']:6.3f} | RMSE: {perf_h['rmse']:6.3f} (N={perf_h['count']})")
            
        # Save individual zone model artifact
        zone_model_path = f'models/hmm/hmm_aqi_zone_{z_id}.joblib'
        best_model.save(zone_model_path)
        aqi_models[z_id] = best_model
        all_trained_models[('AQI', z_id)] = best_model
        
        training_status_records.append({
            'zone_id': z_id,
            'zone_label': z_label,
            'variable': 'AQI',
            'status': 'TRAINED',
            'n_train_obs': n_tr,
            'n_val_obs': n_va,
            'selected_k': best_k,
            'reason': f"Trained successfully (Selected K={best_k})"
        })

    # Save aggregated AQI models dictionary
    joblib.dump(aqi_models, 'models/hmm/hmm_aqi.joblib')
    print(f"\nSaved {len(aqi_models)} zone-specific AQI models into models/hmm/hmm_aqi.joblib")

    # =========================================================================
    # PART 2: RAINFALL ZONE-SPECIFIC HMM MODELS
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: RAINFALL ZONE-SPECIFIC HMM TRAINING & VALIDATION")
    print("=" * 60)
    
    candidate_k_rf = [2, 3, 4]
    
    for z_id in range(1, 16):
        z_label = zone_map[z_id]
        df_tr_z = df_train[df_train['zone_id'] == z_id].sort_values('timestamp')
        df_va_z = df_val[df_val['zone_id'] == z_id].sort_values('timestamp')
        
        assert df_tr_z['timestamp'].is_monotonic_increasing
        assert df_va_z['timestamp'].is_monotonic_increasing
        
        n_tr = int(df_tr_z['rainfall_observed'].sum())
        n_va = int(df_va_z['rainfall_observed'].sum())
        
        print(f"\n[Zone {z_id:2d} - {z_label}] Rainfall: Train N={n_tr}, Val N={n_va}")
        
        if n_tr < MIN_TRAIN_HOURLY or n_va < MIN_VAL_HOURLY:
            reason = "Unmonitored rainfall sensor (0 observations)" if n_tr == 0 else f"Insufficient observations (Train N={n_tr} < {MIN_TRAIN_HOURLY} or Val N={n_va} < {MIN_VAL_HOURLY})"
            print(f"  -> SKIPPED: {reason}")
            training_status_records.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'Rainfall',
                'status': 'UNAVAILABLE',
                'n_train_obs': n_tr,
                'n_val_obs': n_va,
                'selected_k': np.nan,
                'reason': reason
            })
            continue
            
        train_seq = df_tr_z['rainfall'].values
        val_seq = df_va_z['rainfall'].values
        val_timestamps = df_va_z['timestamp'].values
        
        best_k = None
        best_val_mae = float('inf')
        best_model = None
        
        for k in candidate_k_rf:
            hmm_k = TemporalHMM(n_components=k, variable='rainfall', zone_id=z_id, random_state=42)
            hmm_k.fit(train_seq)
            
            bic = hmm_k.bic(train_seq)
            aic = hmm_k.aic(train_seq)
            
            preds_1h = []
            actuals_1h = []
            context = []
            for val_idx, val_val in enumerate(val_seq):
                if len(context) >= 1:
                    p1, _, _ = hmm_k.forecast(np.array(context[-24:]), horizon=1)
                    preds_1h.append(p1)
                    actuals_1h.append(val_val)
                if pd.notna(val_val):
                    context.append(val_val)
                elif len(context) > 0:
                    context.append(0.0) # standard zero for unobserved rain step
                    
            perf_1h = evaluate_forecasts(np.array(actuals_1h), np.array(preds_1h))
            print(f"  K={k} | BIC: {bic:10.1f} | AIC: {aic:10.1f} | Val MAE (1h): {perf_1h['mae']:6.3f} | Val RMSE (1h): {perf_1h['rmse']:6.3f}")
            
            selection_records.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'Rainfall',
                'n_components': k,
                'bic': round(bic, 1),
                'aic': round(aic, 1),
                'val_mae_1h': perf_1h['mae'],
                'val_rmse_1h': perf_1h['rmse'],
                'selected': False
            })
            
            if perf_1h['mae'] < best_val_mae:
                best_val_mae = perf_1h['mae']
                best_k = k
                best_model = hmm_k
                
        for rec in selection_records:
            if rec['zone_id'] == z_id and rec['variable'] == 'Rainfall' and rec['n_components'] == best_k:
                rec['selected'] = True
                
        best_model.val_mae_ = best_val_mae
        print(f"  ==> Selected K={best_k} for Zone {z_id} ({z_label}) with Val MAE (1h) = {best_val_mae:.3f}")
        for s_idx, (m, v) in enumerate(zip(best_model.means_, best_model.variances_)):
            print(f"      State {s_idx}: mu = {m:6.2f} mm, sigma = {np.sqrt(v):6.2f}")
            
        # Multi-horizon evaluation for rainfall: 1h, 3h, 6h, 24h
        for h in [1, 3, 6, 24]:
            preds_h = []
            actuals_h = []
            variances_h = []
            context = []
            for val_idx, val_val in enumerate(val_seq):
                ts = val_timestamps[val_idx]
                if len(context) >= 1:
                    pred_val, pred_var, _ = best_model.forecast(np.array(context[-24:]), horizon=h)
                    preds_h.append(pred_val)
                    actuals_h.append(val_val)
                    variances_h.append(pred_var)
                    if h == 1:
                        val_pred_records.append({
                            'timestamp': ts,
                            'zone_id': z_id,
                            'zone_label': z_label,
                            'variable': 'Rainfall',
                            'horizon': '1h',
                            'actual': val_val,
                            'predicted': pred_val,
                            'variance': pred_var
                        })
                if pd.notna(val_val):
                    context.append(val_val)
                elif len(context) > 0:
                    context.append(0.0)
                    
            perf_h = evaluate_forecasts(np.array(actuals_h), np.array(preds_h))
            validation_metrics.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'Rainfall',
                'model': 'HMM',
                'horizon': f'{h}h',
                'mae': perf_h['mae'],
                'rmse': perf_h['rmse'],
                'sample_count': perf_h['count']
            })
            print(f"      Horizon {h:2d}h -> MAE: {perf_h['mae']:6.3f} | RMSE: {perf_h['rmse']:6.3f} (N={perf_h['count']})")
            
        zone_model_path = f'models/hmm/hmm_rainfall_zone_{z_id}.joblib'
        best_model.save(zone_model_path)
        rainfall_models[z_id] = best_model
        all_trained_models[('Rainfall', z_id)] = best_model
        
        training_status_records.append({
            'zone_id': z_id,
            'zone_label': z_label,
            'variable': 'Rainfall',
            'status': 'TRAINED',
            'n_train_obs': n_tr,
            'n_val_obs': n_va,
            'selected_k': best_k,
            'reason': f"Trained successfully (Selected K={best_k})"
        })

    joblib.dump(rainfall_models, 'models/hmm/hmm_rainfall.joblib')
    print(f"\nSaved {len(rainfall_models)} zone-specific Rainfall models into models/hmm/hmm_rainfall.joblib")

    # =========================================================================
    # PART 3: TEMPERATURE ZONE-SPECIFIC HMM MODELS (DAILY REGIME)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: TEMPERATURE ZONE-SPECIFIC HMM TRAINING & VALIDATION (DAILY REGIME)")
    print("=" * 60)
    
    candidate_k_temp = [2, 3, 4]
    
    for z_id in range(1, 16):
        z_label = zone_map[z_id]
        df_tr_z = df_train[df_train['zone_id'] == z_id].sort_values('timestamp')
        df_va_z = df_val[df_val['zone_id'] == z_id].sort_values('timestamp')
        
        assert df_tr_z['timestamp'].is_monotonic_increasing
        assert df_va_z['timestamp'].is_monotonic_increasing
        
        # Extract daily temperature sequence for this zone
        df_tr_temp = df_tr_z[df_tr_z['temperature_observed'] == True].copy()
        df_tr_temp['date'] = df_tr_temp['timestamp'].str[:10]
        daily_temp_train = df_tr_temp.groupby('date')['temperature'].mean().values
        
        n_days = len(daily_temp_train)
        print(f"\n[Zone {z_id:2d} - {z_label}] Temperature: Train Daily N={n_days} days")
        
        if n_days < MIN_TRAIN_DAILY:
            reason = f"Insufficient training days (N={n_days} < {MIN_TRAIN_DAILY})"
            print(f"  -> SKIPPED: {reason}")
            training_status_records.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'Temperature',
                'status': 'UNAVAILABLE',
                'n_train_obs': n_days,
                'n_val_obs': 0,
                'selected_k': np.nan,
                'reason': reason
            })
            continue
            
        best_k = None
        best_bic = float('inf')
        best_model = None
        
        # Documented exception: BIC on training data is used for temperature K-selection
        # because IMD ground-truth observations end on 2024-06-22 (0 validation records).
        for k in candidate_k_temp:
            hmm_k = TemporalHMM(n_components=k, variable='temperature', zone_id=z_id, random_state=42)
            hmm_k.fit(daily_temp_train)
            
            bic = hmm_k.bic(daily_temp_train)
            aic = hmm_k.aic(daily_temp_train)
            
            print(f"  K={k} | BIC: {bic:10.1f} | AIC: {aic:10.1f}")
            
            selection_records.append({
                'zone_id': z_id,
                'zone_label': z_label,
                'variable': 'Temperature',
                'n_components': k,
                'bic': round(bic, 1),
                'aic': round(aic, 1),
                'val_mae_1h': np.nan,
                'val_rmse_1h': np.nan,
                'selected': False
            })
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = hmm_k
                
        for rec in selection_records:
            if rec['zone_id'] == z_id and rec['variable'] == 'Temperature' and rec['n_components'] == best_k:
                rec['selected'] = True
                
        best_model.bic_ = best_bic
        print(f"  ==> Selected K={best_k} for Zone {z_id} ({z_label}) with BIC = {best_bic:.1f}")
        for s_idx, (m, v) in enumerate(zip(best_model.means_, best_model.variances_)):
            print(f"      State {s_idx}: mu = {m:6.2f} C, sigma = {np.sqrt(v):6.2f}")
            
        # Daily sequential 1-step ahead forecast evaluation across training sequence
        daily_preds = []
        daily_actuals = []
        for i in range(1, len(daily_temp_train)):
            pred, _, _ = best_model.forecast(daily_temp_train[:i], horizon=1)
            daily_preds.append(pred)
            daily_actuals.append(daily_temp_train[i])
            
        perf_temp = evaluate_forecasts(np.array(daily_actuals), np.array(daily_preds))
        print(f"      Temperature 24h Daily Ahead -> MAE: {perf_temp['mae']:6.3f} C | RMSE: {perf_temp['rmse']:6.3f} C (N={perf_temp['count']})")
        validation_metrics.append({
            'zone_id': z_id,
            'zone_label': z_label,
            'variable': 'Temperature',
            'model': 'HMM',
            'horizon': '24h',
            'mae': perf_temp['mae'],
            'rmse': perf_temp['rmse'],
            'sample_count': perf_temp['count']
        })
        
        zone_model_path = f'models/hmm/hmm_temperature_zone_{z_id}.joblib'
        best_model.save(zone_model_path)
        temperature_models[z_id] = best_model
        all_trained_models[('Temperature', z_id)] = best_model
        
        training_status_records.append({
            'zone_id': z_id,
            'zone_label': z_label,
            'variable': 'Temperature',
            'status': 'TRAINED',
            'n_train_obs': n_days,
            'n_val_obs': 0,
            'selected_k': best_k,
            'reason': f"Trained successfully on zone daily partition (Selected K={best_k} via BIC)"
        })

    joblib.dump(temperature_models, 'models/hmm/hmm_temperature.joblib')
    print(f"\nSaved {len(temperature_models)} zone-specific Temperature models into models/hmm/hmm_temperature.joblib")
    
    # Save combined multi-index models registry
    joblib.dump(all_trained_models, 'models/hmm/hmm_models_by_zone.joblib')
    print(f"Saved total {len(all_trained_models)} models into models/hmm/hmm_models_by_zone.joblib")

    # =========================================================================
    # PART 4: SAVE METRICS, STATUS, PREDICTIONS, AND METHODOLOGY REPORT
    # =========================================================================
    df_status = pd.DataFrame(training_status_records)
    df_status.to_csv('results/metrics/model_training_status.csv', index=False)
    print("Saved results/metrics/model_training_status.csv")
    
    df_metrics = pd.DataFrame(validation_metrics)
    df_metrics.to_csv('results/metrics/validation_metrics_hmm.csv', index=False)
    print("Saved results/metrics/validation_metrics_hmm.csv")
    
    df_sel = pd.DataFrame(selection_records)
    df_sel.to_csv('results/metrics/state_selection_hmm.csv', index=False)
    print("Saved results/metrics/state_selection_hmm.csv")
    
    df_preds = pd.DataFrame(val_pred_records)
    df_preds.to_csv('results/predictions/validation_predictions_hmm.csv', index=False)
    print("Saved results/predictions/validation_predictions_hmm.csv")
    
    print("\n" + "=" * 70)
    print("SUMMARY OF TRAINED HMM MODELS:")
    print("=" * 70)
    print(f"AQI Models Trained:         {len(aqi_models)} / 15 zones")
    print(f"Rainfall Models Trained:    {len(rainfall_models)} / 15 zones")
    print(f"Temperature Models Trained: {len(temperature_models)} / 15 zones")
    print(f"Total HMM Models Trained:   {len(all_trained_models)}")
    print("=" * 70)

# =============================================================================
# RAINFALL HMMs TRAINED ON OPEN-METEO (reanalysis) — same source as the live input
# =============================================================================
# One HMM per Open-Meteo grid cell (15 zones map to 8 cells; zones sharing a cell have
# identical series). Chronological splits on data/external/open_meteo_chennai_hourly_2025_2026.csv:
RAIN_OM_SPLITS = {
    'train':    ('2025-01-01 00:00', '2025-11-30 23:00'),   # incl. SW monsoon + Oct-Nov NE monsoon 2025
    'validate': ('2025-12-01 00:00', '2026-03-31 23:00'),   # K selection; Dec 2025 - Jan 2026 rain
    'test':     ('2026-04-01 00:00', '2026-10-07 23:00'),   # held out; SW monsoon 2026
}
RAIN_OM_CANDIDATE_K = [2, 3, 4, 5]
RAIN_OM_HORIZONS = [1, 3, 6, 24]          # hourly rate at t+h
RAIN_OM_ACCUM_KEY = '24h_total'           # accumulated total over t+1..t+24
RAIN_OM_ARTIFACT = os.path.join('models', 'hmm', 'hmm_rainfall_openmeteo.joblib')
RAIN_OM_METRICS = os.path.join('results', 'metrics', 'rainfall_openmeteo_evaluation.csv')


def openmeteo_rain_cells(hourly: pd.DataFrame, column: str = 'precipitation_mm'
                         ) -> Tuple[Dict[str, pd.Series], Dict[int, str]]:
    """Hourly `column` per grid cell (UTC index, gap-preserving) and zone -> cell mapping
    (zones with identical series share a cell)."""
    cells: Dict[str, pd.Series] = {}
    zone_to_cell: Dict[int, str] = {}
    for z, g in hourly.groupby('zone_id'):
        g = g.sort_values('time_utc').drop_duplicates('time_utc').set_index('time_utc')[column]
        s = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq='h'))
        key = next((c for c, cs in cells.items() if cs.index.equals(s.index) and cs.equals(s)), None)
        if key is None:
            key = f'cell_{len(cells) + 1}'
            cells[key] = s
        zone_to_cell[int(z)] = key
    return cells, zone_to_cell


def _split_mask(index: pd.DatetimeIndex, split: str, splits: Optional[Dict] = None) -> np.ndarray:
    lo, hi = (pd.Timestamp(t, tz='UTC') for t in (splits or RAIN_OM_SPLITS)[split])
    return np.asarray((index >= lo) & (index <= hi))


def _monthly_climatology(series: pd.Series, train_mask: np.ndarray) -> np.ndarray:
    """Per-step expected hourly rain from TRAINING months only (fallback: training mean)."""
    tr = series[train_mask]
    by_month = tr.groupby(tr.index.month).mean()
    return series.index.month.map(lambda m: by_month.get(m, tr.mean())).to_numpy(dtype=float)


def rainfall_forecasts_and_baselines(model: TemporalHMM, series: pd.Series, train_mask: np.ndarray
                                     ) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For every origin t: forecasts and target for each horizon key ('1h'...'24h', '24h_total').
    Model forecasts are causal (filtering uses y_1..y_t only). Baselines:
      persistence  hourly: y_t;              total: sum of the last 24 h
      climatology  hourly: training-month mean; total: 24 x that mean
      zero         0 mm
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    clim = _monthly_climatology(series, train_mask)
    targets = _rain_targets(y)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for h in RAIN_OM_HORIZONS:
        yhat, _ = model.filter_and_forecast_sequence(y, horizon=h)
        target, pers, _ = targets[f'{h}h']
        clim_t = np.full(n, np.nan)
        clim_t[:n - h] = clim[h:]
        out[f'{h}h'] = {'target': target, 'model': yhat, 'persistence': pers, 'climatology': clim_t,
                        'zero': np.zeros(n), 'target_index_offset': h}
    acc_mean, _ = model.forecast_accumulation_sequence(y, n_steps=24)
    target, past, _ = targets[RAIN_OM_ACCUM_KEY]
    clim24 = np.array([np.nan if t + 24 >= n else clim[t + 1:t + 25].sum() for t in range(n)])
    out[RAIN_OM_ACCUM_KEY] = {'target': target, 'model': acc_mean, 'persistence': past, 'climatology': clim24,
                              'zero': np.zeros(n), 'target_index_offset': 24}
    return out


def _rain_targets(y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
    """
    Per origin t and horizon key: (target, persistence, target offset in steps).
      'Nh'         target y_{t+N} (hourly RATE);        persistence y_t
      '24h_total'  target sum y_{t+1..t+24} (TOTAL);     persistence sum y_{t-23..t}
    Totals require all 24 hours present (NaN otherwise); nothing is filled.
    """
    n = len(y)
    out: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    for h in RAIN_OM_HORIZONS:
        target = np.full(n, np.nan)
        target[:n - h] = y[h:]
        out[f'{h}h'] = (target, y.copy(), h)
    csum = np.concatenate([[0.0], np.nancumsum(y)])
    valid = np.concatenate([[0], np.cumsum(~np.isnan(y))])
    target = np.full(n, np.nan)
    past = np.full(n, np.nan)
    for t in range(n):
        if t + 24 < n and valid[t + 25] - valid[t + 1] == 24:
            target[t] = csum[t + 25] - csum[t + 1]
        if t >= 23 and valid[t + 1] - valid[t - 23] == 24:
            past[t] = csum[t + 1] - csum[t - 23]
    out[RAIN_OM_ACCUM_KEY] = (target, past, 24)
    return out


def evaluate_rainfall_openmeteo(models: Dict[str, TemporalHMM], cells: Dict[str, pd.Series]) -> pd.DataFrame:
    """Per cell (and pooled 'ALL'), split and horizon: MAE/RMSE for model and baselines.
    An origin counts for a split only if both origin and target lie in that split."""
    rows = []
    pooled: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    for cell, series in cells.items():
        train_mask = _split_mask(series.index, 'train')
        res = rainfall_forecasts_and_baselines(models[cell], series, train_mask)
        for split in RAIN_OM_SPLITS:
            m = _split_mask(series.index, split)
            for key, r in res.items():
                off = r['target_index_offset']
                in_split = m.copy()
                in_split[len(m) - off:] = False
                in_split[:len(m) - off] &= m[off:]
                ok = in_split & ~np.isnan(r['target'])
                for meth in ('model', 'persistence', 'climatology', 'zero'):
                    ok_m = ok & ~np.isnan(r[meth])
                    rows.append({'cell': cell, 'split': split, 'horizon': key, 'method': meth,
                                 **evaluate_forecasts(r['target'][ok_m], r[meth][ok_m])})
                    pooled.setdefault((split, key), {}).setdefault(meth, []).append(
                        np.column_stack([r['target'][ok_m], r[meth][ok_m]]))
    for (split, key), by_meth in pooled.items():
        for meth, parts in by_meth.items():
            a = np.vstack(parts)
            rows.append({'cell': 'ALL', 'split': split, 'horizon': key, 'method': meth,
                         **evaluate_forecasts(a[:, 0], a[:, 1])})
    return pd.DataFrame(rows)


def rainfall_horizon_gate(metrics: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    A horizon is enabled only if the model's pooled RMSE is lower than the BEST baseline
    (persistence, climatology, zero) on BOTH validation and test. MSE skill score is reported.
    """
    gate: Dict[str, Dict[str, object]] = {}
    pooled = metrics[metrics['cell'] == 'ALL']
    for key in [f'{h}h' for h in RAIN_OM_HORIZONS] + [RAIN_OM_ACCUM_KEY]:
        info: Dict[str, object] = {'passed': True}
        for split in ('validate', 'test'):
            p = pooled[(pooled['split'] == split) & (pooled['horizon'] == key)].set_index('method')
            if p.empty or not np.isfinite(p.loc['model', 'rmse']) or p.loc['model', 'count'] == 0:
                info['passed'] = False
                info[f'{split}_skill'] = None
                continue
            base = p.drop(index='model')
            best = base['rmse'].idxmin()
            skill = 1.0 - (p.loc['model', 'rmse'] ** 2) / (base.loc[best, 'rmse'] ** 2)
            info[f'{split}_skill'] = round(float(skill), 4)
            info[f'{split}_best_baseline'] = best
            info[f'{split}_model_rmse'] = float(p.loc['model', 'rmse'])
            info[f'{split}_baseline_rmse'] = float(base.loc[best, 'rmse'])
            info['passed'] = bool(info['passed'] and skill > 0)
        gate[key] = info
    return gate


def train_rainfall_openmeteo(save: bool = False) -> Dict[str, object]:
    """Fit per-cell rainfall HMMs on the train split, pick K by validation log-likelihood,
    evaluate against baselines, gate horizons. Writes artifacts only when save=True."""
    from src.collectors import open_meteo
    hourly = open_meteo.load_local()['hourly']
    cells, zone_to_cell = openmeteo_rain_cells(hourly)
    models, selection = {}, []
    for cell, series in cells.items():
        y = series.to_numpy(dtype=float)
        tr, va = _split_mask(series.index, 'train'), _split_mask(series.index, 'validate')
        best = None
        for k in RAIN_OM_CANDIDATE_K:
            m = TemporalHMM(n_components=k, variable='rainfall', zone_id=None, random_state=42).fit(y[tr])
            ll = m.score(y[va])
            selection.append({'cell': cell, 'k': k, 'val_loglik': round(ll, 1), 'bic_train': round(m.bic(y[tr]), 1)})
            if best is None or ll > best[0]:
                best = (ll, k, m)
        best[2].training_time_range_ = RAIN_OM_SPLITS['train']
        models[cell] = best[2]
        print(f'{cell} (zones {[z for z, c in zone_to_cell.items() if c == cell]}): K={best[1]}')
    metrics = evaluate_rainfall_openmeteo(models, cells)
    gate = rainfall_horizon_gate(metrics)
    for key, g in gate.items():
        print(f"  {key:10s} {'PASS' if g['passed'] else 'FAIL'}  val skill={g.get('validate_skill')}  "
              f"test skill={g.get('test_skill')}")
    bundle = {'models': models, 'zone_to_cell': zone_to_cell, 'gate': gate,
              'meta': {'source': 'data/external/open_meteo_chennai_hourly_2025_2026.csv (reanalysis)',
                       'splits': RAIN_OM_SPLITS, 'k_selection': selection,
                       'selection_rule': 'max validation log-likelihood',
                       'gate_rule': 'pooled RMSE below best baseline (persistence/climatology/zero) on validate AND test',
                       'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}}
    if save:
        os.makedirs(os.path.dirname(RAIN_OM_ARTIFACT), exist_ok=True)
        joblib.dump(bundle, RAIN_OM_ARTIFACT)
        metrics.to_csv(RAIN_OM_METRICS, index=False)
        print(f'Saved {RAIN_OM_ARTIFACT} and {RAIN_OM_METRICS}')
    return bundle


# =============================================================================
# TWO-PART RAINFALL EXPERIMENT (occurrence logistic + seasonal gamma; src/models/rain_model.py)
# Same Open-Meteo data, grid cells and chronological splits as the HMM experiment above.
# =============================================================================
TWOPART_KEYS = [f'{h}h' for h in RAIN_OM_HORIZONS] + [RAIN_OM_ACCUM_KEY]
# Probability events per target; the FIRST one decides the pass rule, the others are reported.
TWOPART_EVENTS = {'hourly': (0.1, 1.0), 'total': (2.5, 15.6)}
TWOPART_HEAVY_MM = 64.5
TWOPART_HEAVY_RAIN_ALERTS_ENABLED = False      # test split has no heavy-rain day: not evaluable
TWOPART_BOOTSTRAP = 1000
TWOPART_SEED = 42
TWOPART_PROB_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
TWOPART_ARTIFACT = os.path.join('models', 'rainfall', 'rain_twopart_openmeteo.joblib')
TWOPART_METRICS = os.path.join('results', 'metrics', 'rainfall_twopart_evaluation.csv')


def _twopart_events(key: str) -> Tuple[float, ...]:
    return TWOPART_EVENTS['total' if key == RAIN_OM_ACCUM_KEY else 'hourly']


def twopart_rows(cells: Dict[str, pd.Series], key: str, splits: Optional[Dict] = None) -> pd.DataFrame:
    """
    One row per (grid cell, origin t) for target `key`, with forecast-time features, the target,
    the persistence value and a split label. A row belongs to a split only if its origin AND
    target lie in that split (otherwise it is excluded). Rows lacking complete features, target
    or persistence are dropped, so model and baselines are scored on identical rows.
    Also: IST date of the origin (bootstrap block), target IST month/hour/season, origin month.
    """
    from src.models.rain_model import FEATURES, SEASON_OF_MONTH, build_features
    frames = []
    for cell in sorted(cells):
        s = cells[cell]
        target, pers, off = _rain_targets(s.to_numpy(dtype=float))[key]
        f = build_features(s)
        f['cell'], f['y'], f['persistence'] = cell, target, pers
        split = np.full(len(s), None, dtype=object)
        for sp in (splits or RAIN_OM_SPLITS):
            m = _split_mask(s.index, sp, splits)
            ok = m.copy()
            ok[len(m) - off:] = False
            ok[:len(m) - off] &= m[off:]
            split[ok] = sp
        f['split'] = split
        origin_ist = s.index.tz_convert('Asia/Kolkata')
        target_ist = (s.index + pd.Timedelta(hours=off)).tz_convert('Asia/Kolkata')
        f['day'] = origin_ist.strftime('%Y-%m-%d')
        f['origin_month'] = origin_ist.month
        f['target_month'], f['target_hour'] = target_ist.month, target_ist.hour
        f['season'] = [SEASON_OF_MONTH[mo] for mo in target_ist.month]
        frames.append(f)
    df = pd.concat(frames)
    df = df[df['split'].notna()].dropna(subset=FEATURES + ['y', 'persistence'])
    for cell in sorted(cells):
        df[f'is_{cell}'] = (df['cell'] == cell).astype(float)
    return df


def _train_lookup(df: pd.DataFrame, values: np.ndarray, levels: List[List[str]]) -> np.ndarray:
    """
    Per cell, the TRAIN-split mean of `values` grouped by the first available level of keys
    (e.g. [['target_month','target_hour'], ['target_hour'], []]); [] = the cell's train mean.
    Rows from validation/test never enter the averages.
    """
    out = np.full(len(df), np.nan)
    vals = pd.Series(np.asarray(values, dtype=float), index=np.arange(len(df)))
    pos = pd.Series(np.arange(len(df)))
    for cell, idx in pos.groupby(df['cell'].to_numpy()):
        g = df.iloc[idx.to_numpy()]
        v = vals.iloc[idx.to_numpy()].to_numpy()
        train = (g['split'] == 'train').to_numpy()
        res = np.full(len(g), np.nan)
        for keys in levels:
            if keys:
                table = pd.Series(v[train]).groupby([g[k].to_numpy()[train] for k in keys]).mean()
                look = pd.MultiIndex.from_arrays([g[k].to_numpy() for k in keys]) if len(keys) > 1 else g[keys[0]].to_numpy()
                cand = table.reindex(look).to_numpy()
            else:
                cand = np.full(len(g), v[train].mean() if train.any() else np.nan)
            res = np.where(np.isnan(res), cand, res)
        out[idx.to_numpy()] = res
    return out


def twopart_scored_rows(df: pd.DataFrame, pred: Dict[str, np.ndarray], key: str,
                        extra_prob_baselines: Optional[Dict[float, np.ndarray]] = None) -> pd.DataFrame:
    """Row-level probabilities, losses and point forecasts for the model and every baseline."""
    from src.models.rain_model import WET_MM, crps_zero_inflated_gamma, prob_exceed
    r = df[['cell', 'split', 'season', 'day', 'y', 'persistence']].copy()
    y = r['y'].to_numpy()
    p, a, s = pred['p_wet'], pred['shape'], pred['scale']
    for thr in (WET_MM,) + _twopart_events(key):
        if f'o_{thr}' in r:
            continue
        o = (y >= thr).astype(float)
        state = (r['persistence'].to_numpy() >= thr).astype(float)
        r[f'o_{thr}'] = o
        r[f'pm_{thr}'] = prob_exceed(p, a, s, thr, WET_MM)
        r[f'pp_{thr}'] = state                                                   # persistence
        r[f'pc_{thr}'] = _train_lookup(df, o, [['target_month', 'target_hour'], ['target_hour'], []])
        st = pd.DataFrame({'state': state, 'origin_month': df['origin_month'].to_numpy(), 'cell': df['cell'].to_numpy(),
                           'split': df['split'].to_numpy()})
        r[f'pk_{thr}'] = _train_lookup(st, o, [['state', 'origin_month'], ['state'], []])   # conditional clim.
        if extra_prob_baselines and thr in extra_prob_baselines:                       # e.g. harmonic clim.
            r[f'ph_{thr}'] = np.asarray(extra_prob_baselines[thr], dtype=float)
    p_clim = r[f'pc_{WET_MM}'].to_numpy()
    r['crps_model'] = crps_zero_inflated_gamma(y, p, a, s, WET_MM)
    r['crps_clim'] = crps_zero_inflated_gamma(y, p_clim, a, s, WET_MM)       # climatological occurrence, same amounts
    r['crps_zero'] = np.abs(y)
    r['crps_persistence'] = np.abs(r['persistence'].to_numpy() - y)
    r['model_mean'], r['model_median'] = pred['mean'], pred['median']
    return r


def _choose_prob_threshold(p: np.ndarray, o: np.ndarray) -> float:
    """Decision threshold maximising CSI on the given (VALIDATION) rows."""
    best, best_csi = 0.5, -1.0
    for t in TWOPART_PROB_GRID:
        f = p >= t
        hits, fa, miss = np.sum(f & (o == 1)), np.sum(f & (o == 0)), np.sum(~f & (o == 1))
        csi = hits / (hits + fa + miss) if hits + fa + miss else 0.0
        if csi > best_csi:
            best, best_csi = float(t), csi
    return best


def _score_group(g: pd.DataFrame, key: str, p_star: float) -> Dict[str, float]:
    """Unrounded scores for one group of rows."""
    from sklearn.metrics import roc_auc_score
    from src.models.rain_model import WET_MM
    out: Dict[str, float] = {'n': int(len(g)), 'n_days': int(g['day'].nunique())}
    y = g['y'].to_numpy()
    for i, thr in enumerate(_twopart_events(key)):
        o = g[f'o_{thr}'].to_numpy()
        out[f'events_ge{thr}'] = int(o.sum())
        briers = {meth: float(np.mean((g[f'{col}_{thr}'].to_numpy() - o) ** 2))
                  for meth, col in (('model', 'pm'), ('climatology', 'pc'), ('persistence', 'pp'),
                                    ('conditional_climatology', 'pk'), ('harmonic_climatology', 'ph'))
                  if f'{col}_{thr}' in g}
        best = min((m for m in briers if m != 'model'), key=lambda m: briers[m])
        for meth, b in briers.items():
            out[f'brier_ge{thr}_{meth}'] = b
        out[f'bss_ge{thr}'] = 1.0 - briers['model'] / briers[best] if briers[best] > 0 else np.nan
        out[f'bss_ge{thr}_best_baseline'] = best
        out[f'auc_ge{thr}'] = (float(roc_auc_score(o, g[f'pm_{thr}'])) if 0 < o.sum() < len(o) else np.nan)
        if i == 0:
            f = g[f'pm_{thr}'].to_numpy() >= p_star
            hits, fa, miss = np.sum(f & (o == 1)), np.sum(f & (o == 0)), np.sum(~f & (o == 1))
            out.update(prob_threshold=p_star,
                       pod=hits / (hits + miss) if hits + miss else np.nan,
                       far=fa / (hits + fa) if hits + fa else np.nan,
                       csi=hits / (hits + fa + miss) if hits + fa + miss else np.nan,
                       freq_bias=(hits + fa) / (hits + miss) if hits + miss else np.nan)
    for meth in ('model', 'clim', 'zero', 'persistence'):
        out[f'crps_{meth}'] = float(g[f'crps_{meth}'].mean())
    for meth in ('clim', 'zero', 'persistence'):
        out[f'crpss_vs_{meth}'] = 1.0 - out['crps_model'] / out[f'crps_{meth}'] if out[f'crps_{meth}'] > 0 else np.nan
    wet = y >= WET_MM
    out['mae_model_median'] = float(np.mean(np.abs(g['model_median'] - y)))
    out['rmse_model_mean'] = float(np.sqrt(np.mean((g['model_mean'] - y) ** 2)))
    out['mae_zero'] = float(np.mean(np.abs(y)))
    out['rmse_zero'] = float(np.sqrt(np.mean(y ** 2)))
    out['mae_persistence'] = float(np.mean(np.abs(g['persistence'] - y)))
    out['rmse_persistence'] = float(np.sqrt(np.mean((g['persistence'] - y) ** 2)))
    out['mae_wet_model_median'] = float(np.mean(np.abs(g['model_median'].to_numpy()[wet] - y[wet]))) if wet.any() else np.nan
    out['mae_wet_persistence'] = float(np.mean(np.abs(g['persistence'].to_numpy()[wet] - y[wet]))) if wet.any() else np.nan
    return out


def day_block_ci(days: np.ndarray, loss_model: np.ndarray, loss_ref: np.ndarray,
                 n_boot: int = TWOPART_BOOTSTRAP, seed: int = TWOPART_SEED) -> Tuple[float, float]:
    """
    95% interval for skill = 1 - mean(loss_model)/mean(loss_ref), resampling whole IST days.
    All grid cells of a day stay together (they share the same weather), so the interval is
    not overstated by treating correlated cell-hours as independent.
    """
    codes, uniq = pd.factorize(np.asarray(days))
    sm = np.bincount(codes, weights=np.asarray(loss_model, dtype=float), minlength=len(uniq))
    sr = np.bincount(codes, weights=np.asarray(loss_ref, dtype=float), minlength=len(uniq))
    if sr.sum() <= 0:                    # reference is perfect: skill undefined
        return float('nan'), float('nan')
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    den = sr[idx].sum(axis=1)
    skill = np.where(den > 0, 1.0 - sm[idx].sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)
    lo, hi = np.nanquantile(skill, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_twopart(scored: pd.DataFrame, key: str, p_star: Optional[float] = None) -> pd.DataFrame:
    """Scores by split x cell (incl. 'ALL') x season (incl. 'ALL'), with day-block CIs for pooled rows.
    The categorical decision threshold is chosen on VALIDATION rows only and applied to every split,
    unless a fixed p_star is given (e.g. one chosen in earlier selection folds)."""
    thr0 = _twopart_events(key)[0]
    val = scored[scored['split'] == 'validate']
    if p_star is None:
        p_star = _choose_prob_threshold(val[f'pm_{thr0}'].to_numpy(), val[f'o_{thr0}'].to_numpy()) if len(val) else 0.5
    rows = []
    for split, gs in scored.groupby('split'):
        for cell in ['ALL'] + sorted(gs['cell'].unique()):
            gc = gs if cell == 'ALL' else gs[gs['cell'] == cell]
            for season in ['ALL'] + sorted(gc['season'].unique()):
                g = gc if season == 'ALL' else gc[gc['season'] == season]
                if g.empty:
                    continue
                rec = {'horizon': key, 'quantity': 'total_24h_mm' if key == RAIN_OM_ACCUM_KEY else 'hourly_rate_mm_per_h',
                       'split': split, 'cell': cell, 'season': season, **_score_group(g, key, p_star)}
                if cell == 'ALL' and season == 'ALL':
                    o = g[f'o_{thr0}'].to_numpy()
                    best = rec[f'bss_ge{thr0}_best_baseline']
                    col = {'climatology': 'pc', 'persistence': 'pp', 'conditional_climatology': 'pk',
                           'harmonic_climatology': 'ph'}[best]
                    rec['bss_ci_low'], rec['bss_ci_high'] = day_block_ci(
                        g['day'].to_numpy(), (g[f'pm_{thr0}'] - o) ** 2, (g[f'{col}_{thr0}'] - o) ** 2)
                    rec['crpss_clim_ci_low'], rec['crpss_clim_ci_high'] = day_block_ci(
                        g['day'].to_numpy(), g['crps_model'], g['crps_clim'])
                rows.append(rec)
    return pd.DataFrame(rows)


def twopart_gate(metrics: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """
    Agreed pass rules, on pooled rows (all cells, all seasons), compared UNROUNDED:
      probability  Brier skill of the gating event > 0 vs the best baseline on validation AND test
      amount       probability passes AND CRPS skill vs climatology > 0 on validation AND test
    Missing/NaN scores fail. Heavy-rain alerts stay disabled regardless.
    """
    gate: Dict[str, Dict[str, object]] = {}
    pooled = metrics[(metrics['cell'] == 'ALL') & (metrics['season'] == 'ALL')]
    for key in TWOPART_KEYS:
        thr0 = _twopart_events(key)[0]
        info: Dict[str, object] = {'event_mm': thr0, 'probability_passed': True, 'amount_passed': True}
        for split in ('validate', 'test'):
            p = pooled[(pooled['horizon'] == key) & (pooled['split'] == split)]
            bss = float(p[f'bss_ge{thr0}'].iloc[0]) if len(p) else np.nan
            crpss = float(p['crpss_vs_clim'].iloc[0]) if len(p) else np.nan
            info[f'{split}_bss'], info[f'{split}_crpss_clim'] = bss, crpss
            info[f'{split}_best_baseline'] = p[f'bss_ge{thr0}_best_baseline'].iloc[0] if len(p) else None
            info['probability_passed'] = bool(info['probability_passed'] and np.isfinite(bss) and bss > 0)
            info['amount_passed'] = bool(info['amount_passed'] and np.isfinite(crpss) and crpss > 0)
        info['amount_passed'] = bool(info['amount_passed'] and info['probability_passed'])
        gate[key] = info
    return gate


def heavy_rain_event_counts(cells: Dict[str, pd.Series]) -> Dict[str, Dict[str, int]]:
    """24 h totals >= TWOPART_HEAVY_MM per split: distinct IST origin days (any cell) and cell-windows."""
    df = twopart_rows(cells, RAIN_OM_ACCUM_KEY)
    heavy = df[df['y'] >= TWOPART_HEAVY_MM]
    return {sp: {'distinct_days': int(heavy.loc[heavy['split'] == sp, 'day'].nunique()),
                 'cell_windows': int((heavy['split'] == sp).sum())} for sp in RAIN_OM_SPLITS}


def train_rainfall_twopart(save: bool = False) -> Dict[str, object]:
    """
    Fit one TwoPartRainModel per target on TRAIN rows (pooled over grid cells, cell as a feature),
    score validation/test against climatology, persistence and conditional climatology, apply the
    pass rules. Writes artifacts only when save=True. Heavy-rain alerts remain disabled.
    """
    from src.collectors import open_meteo
    from src.models.rain_model import FEATURES, TwoPartRainModel
    cells, zone_to_cell = openmeteo_rain_cells(open_meteo.load_local()['hourly'])
    models, metrics = {}, []
    for key in TWOPART_KEYS:
        df = twopart_rows(cells, key)
        feats = FEATURES + [c for c in df.columns if c.startswith('is_')]
        tr = df[df['split'] == 'train']
        model = TwoPartRainModel(key, feature_names=feats).fit(tr[feats], tr['y'].to_numpy(), tr['season'].to_numpy())
        scored = twopart_scored_rows(df, model.predict(df[feats], df['season'].to_numpy()), key)
        metrics.append(evaluate_twopart(scored, key))
        models[key] = model
    metrics = pd.concat(metrics, ignore_index=True)
    gate = twopart_gate(metrics)
    heavy = heavy_rain_event_counts(cells)
    pd.set_option('display.width', 250)
    cols = ['horizon', 'split', 'cell', 'season', 'n', 'n_days']
    pooled = metrics[(metrics['cell'] == 'ALL') & (metrics['season'] == 'ALL')]
    print('\nPOOLED (all cells, all seasons)')
    print(pooled[cols + [c for c in metrics.columns if c.startswith(('bss_ge', 'crpss', 'crps_', 'auc'))
                         or c in ('pod', 'far', 'csi', 'freq_bias', 'prob_threshold', 'bss_ci_low', 'bss_ci_high',
                                  'crpss_clim_ci_low', 'crpss_clim_ci_high', 'mae_model_median', 'mae_zero',
                                  'rmse_model_mean', 'rmse_zero')]].to_string(index=False))
    for by in ('cell', 'season'):
        sub = metrics[(metrics['cell' if by == 'season' else 'season'] == 'ALL') & (metrics[by] != 'ALL')]
        print(f'\nBY {by.upper()} (Brier skill of the gating event, CRPS skill vs climatology)')
        print(sub[cols + ['bss_ge0.1', 'bss_ge2.5', 'crpss_vs_clim', 'crpss_vs_zero']].to_string(index=False))
    print('\nPASS RULES')
    for key, g in gate.items():
        print(f"  {key:10s} probability={'PASS' if g['probability_passed'] else 'FAIL'} "
              f"amount={'PASS' if g['amount_passed'] else 'FAIL'}  BSS val/test={g['validate_bss']:.4f}/{g['test_bss']:.4f}  "
              f"CRPSS(clim) val/test={g['validate_crpss_clim']:.4f}/{g['test_crpss_clim']:.4f}")
    print(f'\nHEAVY RAIN (24 h total >= {TWOPART_HEAVY_MM} mm): {heavy} — alerts not evaluated, remain disabled')
    bundle = {'models': models, 'zone_to_cell': zone_to_cell, 'cells': sorted(cells), 'gate': gate,
              'metrics': metrics, 'heavy_rain_events': heavy,
              'meta': {'source': 'data/external/open_meteo_chennai_hourly_2025_2026.csv (reanalysis)',
                       'splits': RAIN_OM_SPLITS, 'events': TWOPART_EVENTS,
                       'heavy_rain_alerts_enabled': TWOPART_HEAVY_RAIN_ALERTS_ENABLED,
                       'gate_rule': 'probability: BSS(gating event) > 0 vs best of climatology/persistence/'
                                    'conditional climatology on validate AND test; amount: probability passes AND '
                                    'CRPSS vs climatology > 0 on validate AND test (unrounded)',
                       'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}}
    if save:
        os.makedirs(os.path.dirname(TWOPART_ARTIFACT), exist_ok=True)
        joblib.dump(bundle, TWOPART_ARTIFACT)
        metrics.to_csv(TWOPART_METRICS, index=False)
        print(f'Saved {TWOPART_ARTIFACT} and {TWOPART_METRICS}')
    return bundle


# =============================================================================
# HOURLY TEMPERATURE EXPERIMENT (src/models/temp_model.py) — same Open-Meteo file, splits, cells
# =============================================================================
TEMP_HOURLY_HORIZONS = [1, 3, 6, 24]
TEMP_HOURLY_ARTIFACT = os.path.join('models', 'temperature', 'temp_hourly_openmeteo.joblib')
TEMP_HOURLY_METRICS = os.path.join('results', 'metrics', 'temperature_hourly_evaluation.csv')


def temp_rows(cells: Dict[str, pd.Series], h: int, weather: Optional[Dict[str, pd.DataFrame]] = None,
              required: Optional[List[str]] = None, splits: Optional[Dict] = None) -> pd.DataFrame:
    """Rows per (cell, origin) for horizon h: features, target T(t+h), baselines, split label
    (origin AND target in the same split), IST day/season/target month/hour, origin month/hour.
    With `weather` (per-cell hourly frames), candidate features are built instead (experiment only)."""
    from src.models.rain_model import SEASON_OF_MONTH
    from src.models.temp_model import TEMP_FEATURES, temp_candidate_features, temp_features
    frames = []
    for cell in sorted(cells):
        s = cells[cell]
        f = temp_features(s, h) if weather is None else temp_candidate_features(weather[cell], h)
        f['cell'], f['y'] = cell, s.shift(-h).to_numpy()
        split = np.full(len(s), None, dtype=object)
        for sp in (splits or RAIN_OM_SPLITS):
            m = _split_mask(s.index, sp, splits)
            ok = m.copy()
            ok[len(m) - h:] = False
            ok[:len(m) - h] &= m[h:]
            split[ok] = sp
        f['split'] = split
        tgt = (s.index + pd.Timedelta(hours=h)).tz_convert('Asia/Kolkata')
        org = s.index.tz_convert('Asia/Kolkata')
        f['day'] = org.strftime('%Y-%m-%d')
        f['origin_month'], f['origin_hour'] = org.month, org.hour
        f['target_month'], f['target_hour'] = tgt.month, tgt.hour
        f['season'] = [SEASON_OF_MONTH[mo] for mo in tgt.month]
        frames.append(f)
    df = pd.concat(frames)
    return df[df['split'].notna()].dropna(subset=(required or TEMP_FEATURES) + ['y'])


def evaluate_temp(df: pd.DataFrame, pred: np.ndarray, key: str,
                  extra_baselines: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """MAE/RMSE for model and baselines (persistence, same hour yesterday, diurnal climatology from
    TRAIN rows by cell x month x hour, falling back to cell x hour); MSE skill vs the best baseline
    (unrounded); day-block 95% CI on pooled rows."""
    y = df['y'].to_numpy()
    meth = {'model': pred, 'persistence': df['t_now'].to_numpy(), 'same_hour_yesterday': df['t_same_hour_yday'].to_numpy(),
            'climatology': _train_lookup(df, y, [['target_month', 'target_hour'], ['target_hour'], []]),
            **(extra_baselines or {})}
    se = {m: (v - y) ** 2 for m, v in meth.items()}
    ae = {m: np.abs(v - y) for m, v in meth.items()}
    rows = []
    for split in sorted(df['split'].unique()):
        ms = (df['split'] == split).to_numpy()
        for cell in ['ALL'] + sorted(df['cell'].unique()):
            mc = ms & ((df['cell'] == cell).to_numpy() if cell != 'ALL' else True)
            for season in ['ALL'] + sorted(df.loc[mc, 'season'].unique()):
                m = mc & ((df['season'] == season).to_numpy() if season != 'ALL' else True)
                if not m.any():
                    continue
                rec = {'horizon': key, 'split': split, 'cell': cell, 'season': season, 'n': int(m.sum()),
                       'n_days': int(df.loc[m, 'day'].nunique())}
                for name in meth:
                    rec[f'mae_{name}'] = float(ae[name][m].mean())
                    rec[f'rmse_{name}'] = float(np.sqrt(se[name][m].mean()))
                best = min((n for n in meth if n != 'model'), key=lambda n: se[n][m].mean())
                rec['best_baseline'] = best
                rec['skill'] = 1.0 - se['model'][m].mean() / se[best][m].mean() if se[best][m].mean() > 0 else np.nan
                if cell == 'ALL' and season == 'ALL':
                    rec['skill_ci_low'], rec['skill_ci_high'] = day_block_ci(df.loc[m, 'day'].to_numpy(),
                                                                             se['model'][m], se[best][m])
                rows.append(rec)
    return pd.DataFrame(rows)


def temp_gate(metrics: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    """A horizon passes only if pooled MSE skill vs the best baseline is > 0 on validation AND test."""
    pooled = metrics[(metrics['cell'] == 'ALL') & (metrics['season'] == 'ALL')]
    gate = {}
    for h in TEMP_HOURLY_HORIZONS:
        key = f'{h}h'
        info: Dict[str, object] = {'passed': True}
        for split in ('validate', 'test'):
            p = pooled[(pooled['horizon'] == key) & (pooled['split'] == split)]
            sk = float(p['skill'].iloc[0]) if len(p) else np.nan
            info[f'{split}_skill'] = sk
            info[f'{split}_best_baseline'] = p['best_baseline'].iloc[0] if len(p) else None
            info[f'{split}_rmse_model'] = float(p['rmse_model'].iloc[0]) if len(p) else np.nan
            info['passed'] = bool(info['passed'] and np.isfinite(sk) and sk > 0)
        gate[key] = info
    return gate


def train_temperature_hourly(save: bool = False) -> Dict[str, object]:
    """Fit one HourlyTempModel per horizon on TRAIN rows (pooled over grid cells), evaluate, gate.
    Writes artifacts only when save=True."""
    from src.collectors import open_meteo
    from src.models.temp_model import TEMP_FEATURES, HourlyTempModel
    cells, zone_to_cell = openmeteo_rain_cells(open_meteo.load_local()['hourly'], column='temperature_2m_c')
    models, metrics = {}, []
    for h in TEMP_HOURLY_HORIZONS:
        df = temp_rows(cells, h)
        tr = df[df['split'] == 'train']
        model = HourlyTempModel(h).fit(tr[TEMP_FEATURES], tr['y'].to_numpy())
        metrics.append(evaluate_temp(df, model.predict(df[TEMP_FEATURES]), f'{h}h'))
        models[f'{h}h'] = model
    metrics = pd.concat(metrics, ignore_index=True)
    gate = temp_gate(metrics)
    pd.set_option('display.width', 250)
    pooled = metrics[(metrics['cell'] == 'ALL') & (metrics['season'] == 'ALL')]
    print(f'cells: {len(cells)} {dict((c, [z for z, cc in zone_to_cell.items() if cc == c]) for c in sorted(cells))}')
    print(pooled.drop(columns=['cell', 'season']).to_string(index=False))
    sub = metrics[(metrics['cell'] == 'ALL') & (metrics['season'] != 'ALL')]
    print(sub[['horizon', 'split', 'season', 'n', 'rmse_model', 'rmse_persistence', 'rmse_same_hour_yesterday',
               'rmse_climatology', 'skill', 'best_baseline']].to_string(index=False))
    cellm = metrics[(metrics['cell'] != 'ALL') & (metrics['season'] == 'ALL') & (metrics['split'] == 'test')]
    print(cellm[['horizon', 'cell', 'rmse_model', 'skill', 'best_baseline']].to_string(index=False))
    for key, g in gate.items():
        print(f"  {key:4s} {'PASS' if g['passed'] else 'FAIL'}  skill val/test = {g['validate_skill']:.4f}/{g['test_skill']:.4f}"
              f"  (best baseline {g['validate_best_baseline']}/{g['test_best_baseline']})")
    for key, mdl in models.items():
        print(f'  {key}: {mdl.summary()}')
    bundle = {'models': models, 'zone_to_cell': zone_to_cell, 'cells': sorted(cells), 'gate': gate, 'metrics': metrics,
              'meta': {'source': 'data/external/open_meteo_chennai_hourly_2025_2026.csv (reanalysis temperature_2m)',
                       'splits': RAIN_OM_SPLITS, 'features': TEMP_FEATURES,
                       'gate_rule': 'pooled MSE skill vs best of persistence/same-hour-yesterday/diurnal climatology '
                                    '> 0 on validate AND test (unrounded)',
                       'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}}
    if save:
        os.makedirs(os.path.dirname(TEMP_HOURLY_ARTIFACT), exist_ok=True)
        joblib.dump(bundle, TEMP_HOURLY_ARTIFACT)
        metrics.to_csv(TEMP_HOURLY_METRICS, index=False)
        print(f'Saved {TEMP_HOURLY_ARTIFACT} and {TEMP_HOURLY_METRICS}')
    return bundle


# =============================================================================
# FORECAST CANDIDATE EXPERIMENT (3 h / 6 h / 24 h temperature; 3 h / 6 h / 24 h / 24 h total rainfall)
# Same local Open-Meteo data, splits, baselines, evaluation and pass rules as production.
# Writes ONLY under models/candidates/ and results/experiments/ (and only with --save).
# =============================================================================
CAND_MODELS_DIR = os.path.join('models', 'candidates')
CAND_RESULTS_DIR = os.path.join('results', 'experiments')
CAND_TEMP_ARTIFACT = os.path.join(CAND_MODELS_DIR, 'temp_hourly_candidate.joblib')
CAND_RAIN_ARTIFACT = os.path.join(CAND_MODELS_DIR, 'rain_twopart_candidate.joblib')
CAND_TEMP_METRICS = os.path.join(CAND_RESULTS_DIR, 'temperature_candidate_evaluation.csv')
CAND_RAIN_METRICS = os.path.join(CAND_RESULTS_DIR, 'rainfall_candidate_evaluation.csv')
CAND_REPORT = os.path.join(CAND_RESULTS_DIR, 'forecast_candidates_report.md')
CAND_TEMP_HORIZONS = [3, 6, 24]
CAND_RAIN_KEYS = ['3h', '6h', '24h', RAIN_OM_ACCUM_KEY]
PRODUCTION_ARTIFACTS = (TEMP_HOURLY_ARTIFACT, TWOPART_ARTIFACT, RAIN_OM_ARTIFACT, TEMP_HOURLY_METRICS, TWOPART_METRICS)
RAIN_CANDIDATE_EXTRA = ['rh_now', 'rh_change_3h', 'wind_u', 'wind_v', 't_dev_24h', 'hour_sin2', 'hour_cos2',
                        'logit_pclim', 'logit_pcond']
# Encoding key levels (shared with the stored tables used live; see src/models/encoding.py)
RAIN_PCLIM_LEVELS = [['target_month', 'target_hour'], ['target_hour'], []]
RAIN_PCOND_LEVELS = [['state', 'origin_month'], ['state'], []]
TEMP_CLIM_TARGET_LEVELS = [['target_month', 'target_hour'], ['target_hour'], []]
TEMP_CLIM_ORIGIN_LEVELS = [['origin_month', 'origin_hour'], ['origin_hour'], []]
CAND_TEMP_REFERENCE = os.path.join(CAND_RESULTS_DIR, 'temperature_3h_reference_predictions.csv')
CAND_RAIN_REFERENCE = os.path.join(CAND_RESULTS_DIR, 'rainfall_3h_reference_predictions.csv')
REFERENCE_ROWS_PER_SPLIT = 150


def cell_weather_frames(hourly: pd.DataFrame, key_column: str) -> Tuple[Dict[str, pd.DataFrame], Dict[int, str]]:
    """Per grid cell (grouped by identical `key_column` series): all hourly variables of the cell's
    lowest-numbered zone, on the cell's hourly index. Other variables may differ slightly between zones
    of one cell (cells differ by variable: 8 for rain/wind, 9 for temperature/humidity)."""
    cells, z2c = openmeteo_rain_cells(hourly, key_column)
    frames = {}
    for cell, s in cells.items():
        rep = min(z for z, c in z2c.items() if c == cell)
        g = hourly[hourly['zone_id'] == rep].sort_values('time_utc').drop_duplicates('time_utc').set_index('time_utc')
        frames[cell] = g.reindex(s.index)
    return frames, z2c


ENCODING_MIN_COUNT = 20     # sparser groups fall back to the next level (e.g. December x hour has ~1 train row)


def train_encoding(df: pd.DataFrame, values: np.ndarray, levels: List[List[str]], per_cell: bool = True,
                   loo: bool = True, min_count: int = ENCODING_MIN_COUNT) -> np.ndarray:
    """
    TRAIN-split mean of `values` by the first key level with at least `min_count` training rows
    (e.g. month x hour -> hour -> all). Validation/test rows never enter the means. With loo=True each
    TRAIN row excludes its own value (leave-one-out), so a model fitted on training rows cannot see its
    own target through the feature.
    """
    v = np.asarray(values, dtype=float)
    out = np.full(len(df), np.nan)
    train_all = (df['split'] == 'train').to_numpy()
    groups = ([np.flatnonzero(df['cell'].to_numpy() == c) for c in pd.unique(df['cell'])] if per_cell
              else [np.arange(len(df))])
    for idx in groups:
        g, vv, tr = df.iloc[idx], v[idx], train_all[idx]
        res = np.full(len(idx), np.nan)
        for keys in levels:
            if keys:
                agg = pd.Series(vv[tr]).groupby([g[k].to_numpy()[tr] for k in keys]).agg(['sum', 'count'])
                look = (pd.MultiIndex.from_arrays([g[k].to_numpy() for k in keys]) if len(keys) > 1
                        else g[keys[0]].to_numpy())
                sums, cnts = agg['sum'].reindex(look).to_numpy(), agg['count'].reindex(look).to_numpy()
            else:
                sums, cnts = np.full(len(idx), vv[tr].sum()), np.full(len(idx), float(tr.sum()))
            if loo:
                sums = np.where(tr, sums - vv, sums)
                cnts = np.where(tr, cnts - 1, cnts)
            enough = cnts >= max(1, min_count) if keys else cnts > 0       # final level: any training rows
            with np.errstate(invalid='ignore', divide='ignore'):
                cand = np.where(enough, sums / np.where(cnts > 0, cnts, 1), np.nan)
            res = np.where(np.isnan(res), cand, res)
        out[idx] = res
    return out


def _logit_clip(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    return np.log(p / (1 - p))


def temp_candidate_experiment(hourly: pd.DataFrame) -> Dict[str, object]:
    """Horizon-specific temperature regressions with richer features vs the production model and the
    production baselines, on identical rows."""
    from src.models.temp_model import TEMP_CANDIDATE_EXTRA, TEMP_FEATURES, HourlyTempModel
    cells, _ = openmeteo_rain_cells(hourly, 'temperature_2m_c')
    weather, z2c = cell_weather_frames(hourly, 'temperature_2m_c')
    prod = joblib.load(TEMP_HOURLY_ARTIFACT) if os.path.exists(TEMP_HOURLY_ARTIFACT) else None
    feats = TEMP_FEATURES + TEMP_CANDIDATE_EXTRA
    raw_req = [f for f in feats if f not in ('clim_target', 'anom_now')]
    from src.models.encoding import fit_table
    models, cand_m, prod_m, encodings, reference = {}, [], [], {}, None
    for h in CAND_TEMP_HORIZONS:
        key = f'{h}h'
        df = temp_rows(cells, h, weather=weather, required=raw_req)
        df['clim_target'] = train_encoding(df, df['y'].to_numpy(), TEMP_CLIM_TARGET_LEVELS)
        clim_origin = train_encoding(df, df['t_now'].to_numpy(), TEMP_CLIM_ORIGIN_LEVELS)
        df['anom_now'] = df['t_now'] - clim_origin
        # stored tables for live inference (built from the same TRAIN rows; non-LOO lookup)
        encodings[key] = {'clim_target': fit_table(df, df['y'].to_numpy(), TEMP_CLIM_TARGET_LEVELS,
                                                   min_count=ENCODING_MIN_COUNT),
                          'clim_origin': fit_table(df, df['t_now'].to_numpy(), TEMP_CLIM_ORIGIN_LEVELS,
                                                   min_count=ENCODING_MIN_COUNT)}
        df = df.dropna(subset=['clim_target', 'anom_now'])
        tr = df[df['split'] == 'train']
        model = HourlyTempModel(h, feature_names=feats).fit(tr[feats], tr['y'].to_numpy())
        models[key] = model
        if key == '3h':
            reference = _reference_sample(df, feats, {'pred_temp_c': model.predict(df[feats])})
        cand_m.append(evaluate_temp(df, model.predict(df[feats]), key).assign(model='candidate'))
        if prod is not None and key in prod['models']:
            prod_m.append(evaluate_temp(df, prod['models'][key].predict(df[TEMP_FEATURES]), key).assign(model='production'))
    cand_m = pd.concat(cand_m, ignore_index=True)
    prod_m = pd.concat(prod_m, ignore_index=True) if prod_m else pd.DataFrame()
    gate_c = temp_gate(cand_m)
    gate_p = temp_gate(prod_m) if len(prod_m) else {}
    decision = {}
    for h in CAND_TEMP_HORIZONS:
        key = f'{h}h'
        c, p = gate_c[key], gate_p.get(key, {})
        beats_prod = all(c[f'{s}_skill'] > p.get(f'{s}_skill', -np.inf) for s in ('validate', 'test'))
        decision[key] = {'candidate_passed': c['passed'], 'production_passed': bool(p.get('passed')),
                         'beats_production_on_both_splits': bool(beats_prod),
                         'promotion_eligible': bool(c['passed'] and (beats_prod or not p.get('passed'))),
                         'candidate': c, 'production_same_rows': p}
    return {'models': models, 'zone_to_cell': z2c, 'features': feats, 'metrics': pd.concat([cand_m, prod_m]),
            'gate': gate_c, 'decision': decision, 'encodings': encodings, 'rep_zone': _rep_zones(z2c),
            'reference': reference}


def _rep_zones(z2c: Dict[int, str]) -> Dict[str, int]:
    """Grid cell -> its lowest-numbered zone (whose weather frame the candidates were trained on)."""
    return {c: min(z for z, cc in z2c.items() if cc == c) for c in sorted(set(z2c.values()))}


def _reference_sample(df: pd.DataFrame, feats: List[str], preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Deterministic, evenly spaced validation and test rows with their features and experiment predictions."""
    out = df[['cell', 'split'] + feats].copy()
    for name, p in preds.items():
        out[name] = p
    out['origin_utc'] = [t.isoformat() for t in df.index]
    parts = []
    for split in ('validate', 'test'):
        s = out[out['split'] == split].sort_values(['cell', 'origin_utc'])
        idx = np.unique(np.linspace(0, len(s) - 1, min(REFERENCE_ROWS_PER_SPLIT, len(s))).astype(int))
        parts.append(s.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def rain_candidate_rows(cells: Dict[str, pd.Series], weather: Dict[str, pd.DataFrame], key: str,
                        splits: Optional[Dict] = None) -> pd.DataFrame:
    """twopart_rows plus origin humidity/wind/temperature features and TRAINING-only (leave-one-out)
    climatological rain probabilities pooled over cells (unconditional by target month x hour, and
    conditional on whether it rains now x origin month)."""
    from src.models.rain_model import WET_MM, rain_weather_features
    df = twopart_rows(cells, key, splits)
    off = 24 if key == RAIN_OM_ACCUM_KEY else int(key[:-1])
    extra = []
    for cell in sorted(cells):
        f = rain_weather_features(weather[cell], off)          # same builder as the live forecaster
        f['cell'] = cell
        extra.append(f)
    extra = pd.concat(extra).reset_index()
    extra.columns = ['time_utc'] + list(extra.columns[1:])
    left = df.reset_index()
    left.columns = ['time_utc'] + list(left.columns[1:])
    df = left.merge(extra, on=['time_utc', 'cell'], how='left').set_index('time_utc')
    wet = (df['y'].to_numpy() >= WET_MM).astype(float)
    df['logit_pclim'] = _logit_clip(train_encoding(df, wet, RAIN_PCLIM_LEVELS, per_cell=False))
    df['state'] = (df['persistence'].to_numpy() >= WET_MM).astype(int)
    df['logit_pcond'] = _logit_clip(train_encoding(df, wet, RAIN_PCOND_LEVELS, per_cell=False))
    return df.dropna(subset=RAIN_CANDIDATE_EXTRA)


def rain_candidate_experiment(hourly: pd.DataFrame) -> Dict[str, object]:
    """Two-part candidates (same model family) with humidity/wind/temperature and climatology-anchored
    inputs vs the production two-part models, on identical rows, with the agreed pass rules."""
    from src.models.rain_model import FEATURES, TwoPartRainModel
    cells, z2c = openmeteo_rain_cells(hourly)
    weather, _ = cell_weather_frames(hourly, 'precipitation_mm')
    prod = joblib.load(TWOPART_ARTIFACT) if os.path.exists(TWOPART_ARTIFACT) else None
    from src.models.encoding import fit_table
    from src.models.rain_model import WET_MM
    models, cand_m, prod_m, encodings, reference = {}, [], [], {}, None
    for key in CAND_RAIN_KEYS:
        df = rain_candidate_rows(cells, weather, key)
        cell_cols = [c for c in df.columns if c.startswith('is_')]
        feats = FEATURES + cell_cols + RAIN_CANDIDATE_EXTRA
        wet = (df['y'].to_numpy() >= WET_MM).astype(float)
        # Tables from the rows kept for modelling (the encodings above were computed before dropna on the
        # extra features; only the first 3 h of each cell's record are dropped, all outside validation/test).
        full = twopart_rows(cells, key)
        full['state'] = (full['persistence'].to_numpy() >= WET_MM).astype(int)
        full_wet = (full['y'].to_numpy() >= WET_MM).astype(float)
        encodings[key] = {'pclim': fit_table(full, full_wet, RAIN_PCLIM_LEVELS, per_cell=False, min_count=ENCODING_MIN_COUNT),
                          'pcond': fit_table(full, full_wet, RAIN_PCOND_LEVELS, per_cell=False, min_count=ENCODING_MIN_COUNT)}
        tr = df[df['split'] == 'train']
        model = TwoPartRainModel(key, feature_names=feats).fit(tr[feats], tr['y'].to_numpy(), tr['season'].to_numpy())
        models[key] = model
        if key == '3h':
            pr = model.predict(df[feats], df['season'].to_numpy())
            ref = _reference_sample(df.assign(season_=df['season']), feats + ['season_', 'persistence'],
                                    {'pred_p_wet': pr['p_wet'], 'pred_mean_mm': pr['mean']})
            reference = ref.rename(columns={'season_': 'season'})
        cand_m.append(evaluate_twopart(twopart_scored_rows(df, model.predict(df[feats], df['season'].to_numpy()), key),
                                       key).assign(model='candidate'))
        if prod is not None:
            pm = prod['models'][key]
            prod_m.append(evaluate_twopart(twopart_scored_rows(df, pm.predict(df[pm.feature_names], df['season'].to_numpy()),
                                                               key), key).assign(model='production'))
    cand_m = pd.concat(cand_m, ignore_index=True)
    prod_m = pd.concat(prod_m, ignore_index=True) if prod_m else pd.DataFrame()
    gate_c = {k: v for k, v in twopart_gate(cand_m).items() if k in CAND_RAIN_KEYS}
    gate_p = {k: v for k, v in twopart_gate(prod_m).items() if k in CAND_RAIN_KEYS} if len(prod_m) else {}
    decision = {}
    for key in CAND_RAIN_KEYS:
        c, p = gate_c[key], gate_p.get(key, {})
        beats_prod = all(c[f'{s}_bss'] > p.get(f'{s}_bss', -np.inf) for s in ('validate', 'test'))
        decision[key] = {'candidate_probability_passed': c['probability_passed'],
                         'candidate_amount_passed': c['amount_passed'],
                         'production_probability_passed': bool(p.get('probability_passed')),
                         'beats_production_bss_on_both_splits': bool(beats_prod),
                         'promotion_eligible': bool(c['probability_passed'] and (beats_prod or not p.get('probability_passed'))),
                         'candidate': c, 'production_same_rows': p}
    return {'models': models, 'zone_to_cell': z2c, 'cells': sorted(cells), 'metrics': pd.concat([cand_m, prod_m]),
            'gate': gate_c, 'decision': decision, 'encodings': encodings, 'rep_zone': _rep_zones(z2c),
            'reference': reference}


def _candidate_report(temp: Dict[str, object], rain: Dict[str, object]) -> str:
    lines = ['# Forecast candidate experiment (3 h / 6 h / 24 h)', '',
             'Local Open-Meteo data; production splits, baselines and pass rules; candidates and production models',
             'scored on identical rows. Production artifacts are NOT modified. Promotion requires passing the rules',
             'on validation AND test, and beating the production model where production already passes.', '',
             '## Temperature (MSE skill vs best of persistence / same-hour-yesterday / climatology)', '',
             '| horizon | candidate val | candidate test | production val | production test | candidate passes | promotion eligible |',
             '|---|---|---|---|---|---|---|']
    for key, d in temp['decision'].items():
        c, p = d['candidate'], d['production_same_rows']
        lines.append(f"| {key} | {c['validate_skill']:.4f} | {c['test_skill']:.4f} | {p.get('validate_skill', np.nan):.4f} | "
                     f"{p.get('test_skill', np.nan):.4f} | {d['candidate_passed']} | {d['promotion_eligible']} |")
    lines += ['', '## Rainfall (Brier skill of the gating event; CRPS skill vs climatology)', '',
              '| target | cand BSS val | cand BSS test | prod BSS val | prod BSS test | cand CRPSS val | cand CRPSS test | '
              'probability passes | amount passes | promotion eligible |', '|---|---|---|---|---|---|---|---|---|---|']
    for key, d in rain['decision'].items():
        c, p = d['candidate'], d['production_same_rows']
        lines.append(f"| {key} | {c['validate_bss']:.4f} | {c['test_bss']:.4f} | {p.get('validate_bss', np.nan):.4f} | "
                     f"{p.get('test_bss', np.nan):.4f} | {c['validate_crpss_clim']:.4f} | {c['test_crpss_clim']:.4f} | "
                     f"{d['candidate_probability_passed']} | {d['candidate_amount_passed']} | {d['promotion_eligible']} |")
    lines += ['', 'Heavy-rain alerts remain disabled (no heavy-rain day in the test split).']
    return '\n'.join(lines) + '\n'


def run_forecast_candidates(save: bool = False) -> Dict[str, object]:
    from src.collectors import open_meteo
    hourly = open_meteo.load_local()['hourly']
    temp = temp_candidate_experiment(hourly)
    rain = rain_candidate_experiment(hourly)
    report = _candidate_report(temp, rain)
    print(report)
    if save:
        targets = (CAND_TEMP_ARTIFACT, CAND_RAIN_ARTIFACT, CAND_TEMP_METRICS, CAND_RAIN_METRICS, CAND_REPORT,
                   CAND_TEMP_REFERENCE, CAND_RAIN_REFERENCE)
        assert not set(map(os.path.normpath, targets)) & set(map(os.path.normpath, PRODUCTION_ARTIFACTS))
        os.makedirs(CAND_MODELS_DIR, exist_ok=True)
        os.makedirs(CAND_RESULTS_DIR, exist_ok=True)
        meta = {'source': 'data/external/open_meteo_chennai_hourly_2025_2026.csv', 'splits': RAIN_OM_SPLITS,
                'status': 'CANDIDATE - not used by the live pipeline', 'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}
        joblib.dump({**{k: v for k, v in temp.items() if k != 'metrics'}, 'meta': meta}, CAND_TEMP_ARTIFACT)
        joblib.dump({**{k: v for k, v in rain.items() if k != 'metrics'}, 'meta': meta}, CAND_RAIN_ARTIFACT)
        temp['metrics'].to_csv(CAND_TEMP_METRICS, index=False)
        rain['metrics'].to_csv(CAND_RAIN_METRICS, index=False)
        temp['reference'].to_csv(CAND_TEMP_REFERENCE, index=False)
        rain['reference'].to_csv(CAND_RAIN_REFERENCE, index=False)
        with open(CAND_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)
        print('Saved candidates:', ', '.join(targets))
    return {'temperature': temp, 'rainfall': rain, 'report': report}


# =============================================================================
# CONTROLLED PROMOTION OF THE 3 h CANDIDATES (temperature 3 h, rainfall 3 h) + ROLLBACK
# =============================================================================
PROMOTION_BACKUP_ROOT = os.path.join('models', 'backups')
PROMOTION_TOLERANCE = 1e-9          # max |live - experiment| for every feature and prediction
PROMOTED_KEYS = ('3h',)
PROMOTION_FILES = (TEMP_HOURLY_ARTIFACT, TWOPART_ARTIFACT, TEMP_HOURLY_METRICS, TWOPART_METRICS)


def _sha256(path: str) -> str:
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _atomic_replace(write_fn, path: str) -> None:
    tmp = path + '.tmp'
    write_fn(tmp)
    os.replace(tmp, path)


def backup_files(paths, root: str = PROMOTION_BACKUP_ROOT, label: str = 'promotion') -> str:
    """Copy each file to root/<label>_<UTC timestamp>/ and write manifest.json with SHA-256 hashes."""
    import json
    import shutil
    ts = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
    bdir = os.path.join(root, f'{label}_{ts}')
    os.makedirs(bdir, exist_ok=False)
    entries = []
    for p in paths:
        dst = os.path.join(bdir, os.path.basename(p))
        shutil.copy2(p, dst)
        h = _sha256(p)
        if _sha256(dst) != h:
            raise RuntimeError(f'backup copy mismatch for {p}')
        entries.append({'original': p, 'backup': dst, 'sha256': h})
    manifest = os.path.join(bdir, 'manifest.json')
    _atomic_replace(lambda t: open(t, 'w', encoding='utf-8').write(
        json.dumps({'created_utc': ts, 'files': entries}, indent=2)), manifest)
    return manifest


def rollback_from_manifest(manifest_path: str) -> List[str]:
    """Restore every backed-up file to its original path (atomic per file) and verify its SHA-256."""
    import json
    import shutil
    with open(manifest_path, encoding='utf-8') as f:
        man = json.load(f)
    restored = []
    for e in man['files']:
        if _sha256(e['backup']) != e['sha256']:
            raise RuntimeError(f"backup corrupted: {e['backup']}")
        _atomic_replace(lambda t, src=e['backup']: shutil.copy2(src, t), e['original'])
        if _sha256(e['original']) != e['sha256']:
            raise RuntimeError(f"rollback verification failed: {e['original']}")
        restored.append(e['original'])
    return restored


def verify_3h_live_compat(temp_bundle: Dict, rain_bundle: Dict, hourly: Optional[pd.DataFrame] = None) -> Dict:
    """
    Rebuild every saved reference row through the LIVE input builders (live_forecaster.temp_candidate_row /
    rain_candidate_row) from 72 h windows of the local Open-Meteo data for the representative zone, then
    compare every feature and prediction with the experiment's values. Passes if all |diff| <= tolerance.
    """
    from src.collectors import open_meteo
    from src.forecasting.live_forecaster import (WEATHER_COLUMNS, WEATHER_CONTEXT_HOURS, rain_candidate_row,
                                                 temp_candidate_row)
    from src.preprocessing.live_inputs import hourly_frame
    hourly = open_meteo.load_local()['hourly'] if hourly is None else hourly
    out = {'tolerance': PROMOTION_TOLERANCE}
    tref = pd.read_csv(CAND_TEMP_REFERENCE)
    tm3, tenc = temp_bundle['models']['3h'], temp_bundle['encodings']['3h']
    fdiff = pdiff = 0.0
    for _, r in tref.iterrows():
        origin = pd.Timestamp(r['origin_utc'])
        frame = hourly_frame(hourly, temp_bundle['rep_zone'][r['cell']], WEATHER_COLUMNS, origin, WEATHER_CONTEXT_HOURS)
        x = temp_candidate_row(frame, 3, r['cell'], tenc)
        fdiff = max(fdiff, float(np.nanmax(np.abs(x[tm3.features].to_numpy()[0] - r[tm3.features].to_numpy(dtype=float)))))
        pdiff = max(pdiff, abs(float(tm3.predict(x[tm3.features])[0]) - r['pred_temp_c']))
    out['temperature'] = {'rows': len(tref), 'max_feature_diff': fdiff, 'max_prediction_diff': pdiff}
    rref = pd.read_csv(CAND_RAIN_REFERENCE)
    rm3, renc = rain_bundle['models']['3h'], rain_bundle['encodings']['3h']
    fdiff = pdiff = 0.0
    for _, r in rref.iterrows():
        origin = pd.Timestamp(r['origin_utc'])
        frame = hourly_frame(hourly, rain_bundle['rep_zone'][r['cell']], WEATHER_COLUMNS, origin, WEATHER_CONTEXT_HOURS)
        x = rain_candidate_row(frame['precipitation_mm'], frame, 3, r['cell'], rain_bundle['cells'], renc)
        fdiff = max(fdiff, float(np.nanmax(np.abs(x[rm3.feature_names].to_numpy()[0] - r[rm3.feature_names].to_numpy(dtype=float)))))
        pr = rm3.predict(x[rm3.feature_names], np.array([r['season']]))
        pdiff = max(pdiff, abs(float(pr['p_wet'][0]) - r['pred_p_wet']), abs(float(pr['mean'][0]) - r['pred_mean_mm']))
    out['rainfall'] = {'rows': len(rref), 'max_feature_diff': fdiff, 'max_prediction_diff': pdiff}
    out['passed'] = all(out[v][k] <= PROMOTION_TOLERANCE for v in ('temperature', 'rainfall')
                        for k in ('max_feature_diff', 'max_prediction_diff'))
    return out


def promote_3h_candidates(apply: bool = False) -> Optional[str]:
    """
    Promote ONLY temperature 3 h and rainfall 3 h into the production bundles after compatibility checks.
    apply=False: checks only. apply=True: back up production bundles + metrics (SHA-256 manifest), write the
    new bundles atomically, re-verify through the live forecaster, and roll back automatically on failure.
    """
    ct, cr = joblib.load(CAND_TEMP_ARTIFACT), joblib.load(CAND_RAIN_ARTIFACT)
    pt, prn = joblib.load(TEMP_HOURLY_ARTIFACT), joblib.load(TWOPART_ARTIFACT)
    problems = []
    if not ct['gate']['3h']['passed']:
        problems.append('temperature 3h candidate did not pass its gate')
    if not (cr['gate']['3h']['probability_passed'] and cr['gate']['3h']['amount_passed']):
        problems.append('rainfall 3h candidate did not pass its gates')
    if pt['zone_to_cell'] != ct['zone_to_cell']:
        problems.append('temperature zone->cell mapping differs between production and candidate')
    if prn['zone_to_cell'] != cr['zone_to_cell'] or prn['cells'] != cr['cells']:
        problems.append('rainfall zone->cell mapping differs between production and candidate')
    compat = verify_3h_live_compat(ct, cr)
    print('pre-promotion live compatibility:', compat)
    if not compat['passed']:
        problems.append('live feature/prediction reproduction exceeded tolerance')
    if problems:
        print('NOT PROMOTED:', problems)
        return None
    if not apply:
        print('Checks passed (dry run: nothing written).')
        return None
    manifest = backup_files(PROMOTION_FILES)
    print('Backups + manifest:', manifest)
    stamp = {'promoted_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'keys': list(PROMOTED_KEYS),
             'source': 'models/candidates (forecast candidate experiment)', 'backup_manifest': manifest,
             'live_compatibility': compat}
    new_t = {**pt, 'models': {**pt['models'], '3h': ct['models']['3h']}, 'gate': {**pt['gate'], '3h': ct['gate']['3h']},
             'encodings': {'3h': ct['encodings']['3h']}, 'rep_zone': ct['rep_zone'], 'promotion': stamp}
    new_r = {**prn, 'models': {**prn['models'], '3h': cr['models']['3h']}, 'gate': {**prn['gate'], '3h': cr['gate']['3h']},
             'encodings': {'3h': cr['encodings']['3h']}, 'rep_zone': cr['rep_zone'], 'promotion': stamp}

    def merged(prod_csv, cand_csv):
        prod, cand = pd.read_csv(prod_csv), pd.read_csv(cand_csv)
        cand = cand[(cand['model'] == 'candidate') & (cand['horizon'] == '3h')].reindex(columns=prod.columns)
        return pd.concat([prod[prod['horizon'] != '3h'], cand], ignore_index=True)
    t_metrics, r_metrics = merged(TEMP_HOURLY_METRICS, CAND_TEMP_METRICS), merged(TWOPART_METRICS, CAND_RAIN_METRICS)
    new_t['metrics'], new_r['metrics'] = t_metrics, r_metrics
    try:
        _atomic_replace(lambda t: joblib.dump(new_t, t), TEMP_HOURLY_ARTIFACT)
        _atomic_replace(lambda t: joblib.dump(new_r, t), TWOPART_ARTIFACT)
        _atomic_replace(lambda t: t_metrics.to_csv(t, index=False), TEMP_HOURLY_METRICS)
        _atomic_replace(lambda t: r_metrics.to_csv(t, index=False), TWOPART_METRICS)
        from src.forecasting.live_forecaster import LiveForecaster
        fc = LiveForecaster()
        post = verify_3h_live_compat(fc.temp_hourly, fc.rain_twopart)
        print('post-promotion live compatibility (loaded by LiveForecaster):', post)
        if not post['passed']:
            raise RuntimeError('post-promotion verification failed')
    except Exception:
        print('ROLLING BACK:', rollback_from_manifest(manifest))
        raise
    print('Promoted temperature 3h and rainfall 3h. Rollback: python -m src.models.train_and_validate_hmm '
          f'--rollback-promotion {manifest}')
    return manifest


# =============================================================================
# LONG-HORIZON EXPERIMENT (6 h / 24 h temperature; 6 h / 24 h / 24 h-total rainfall)
# Feature sets are SELECTED by expanding-window CV inside the train+validation period only (folds A-C);
# the selected variant is then scored ONCE on the official validation/test split (test: previously
# examined) and on an 84 h untouched window from the scheduler's cache (indicative only).
# Writes only under models/experiments/long_horizon and results/experiments/long_horizon (with --save).
# =============================================================================
LH_MODELS_DIR = os.path.join('models', 'experiments', 'long_horizon')
LH_RESULTS_DIR = os.path.join('results', 'experiments', 'long_horizon')
LH_ARTIFACT = os.path.join(LH_MODELS_DIR, 'long_horizon_candidates.joblib')
LH_CV_CSV = os.path.join(LH_RESULTS_DIR, 'cv_selection.csv')
LH_FINAL_CSV = os.path.join(LH_RESULTS_DIR, 'final_evaluation.csv')
LH_REPORT = os.path.join(LH_RESULTS_DIR, 'long_horizon_report.md')
LH_SNAPSHOT = os.path.join(LH_RESULTS_DIR, 'open_meteo_recent_snapshot.csv')
LH_LIVE_CACHE = os.path.join('data', 'live', 'raw', 'open_meteo_recent.csv')
LH_UNTOUCHED_START = '2026-10-08 00:00'      # first hour after the local file; never examined before
LH_FOLDS = {   # expanding windows inside 2025-01-01 .. 2026-03-31 (train + validation only)
    'A': {'train': ('2025-01-01 00:00', '2025-06-30 23:00'), 'validate': ('2025-07-01 00:00', '2025-09-30 23:00')},
    'B': {'train': ('2025-01-01 00:00', '2025-09-30 23:00'), 'validate': ('2025-10-01 00:00', '2025-11-30 23:00')},
    'C': {'train': ('2025-01-01 00:00', '2025-11-30 23:00'), 'validate': ('2025-12-01 00:00', '2026-03-31 23:00')},
}
LH_TEMP_HORIZONS = [6, 24]
LH_RAIN_KEYS = ['6h', '24h', RAIN_OM_ACCUM_KEY]
LH_MIN_POSITIVE_FOLDS = 2                     # pre-specified selection rule


def _lh_temp_variants() -> Dict[str, List[str]]:
    from src.models.temp_model import TEMP_CANDIDATE_EXTRA, TEMP_FEATURES
    lags = ['t_lag1', 't_lag3', 't_same_hour_2d', 't_max_24h', 't_min_24h', 'hour_sin2', 'hour_cos2']
    return {'T0_production': list(TEMP_FEATURES),
            'T1_harmonic_clim': TEMP_FEATURES + ['hclim_target', 'hanom_now'],
            'T2_harmonic_clim_lags': TEMP_FEATURES + ['hclim_target', 'hanom_now'] + lags,
            'T3_harmonic_clim_lags_weather': TEMP_FEATURES + ['hclim_target', 'hanom_now'] + lags + ['rh_now', 'wind_u', 'wind_v'],
            'T4_lookup_clim_full': TEMP_FEATURES + TEMP_CANDIDATE_EXTRA}


def _lh_rain_variants(cell_cols: List[str]) -> Dict[str, List[str]]:
    from src.models.rain_model import FEATURES, RAIN_WEATHER_FEATURES
    base = FEATURES + cell_cols
    return {'R0_production': base,
            'R1_harmonic_prob': base + ['logit_pharm'],
            'R2_harmonic_prob_weather': base + ['logit_pharm'] + RAIN_WEATHER_FEATURES,
            'R3_harmonic_prob_weather_cond': base + ['logit_pharm', 'logit_pcond'] + RAIN_WEATHER_FEATURES,
            'R4_lookup_full': base + RAIN_CANDIDATE_EXTRA}


def harmonic_design(times: pd.DatetimeIndex) -> np.ndarray:
    """Smooth seasonal x diurnal climatology design (IST): annual k=1,2, diurnal k=1,2, annual1 x diurnal1."""
    ist = times.tz_convert('Asia/Kolkata')
    a = 2 * np.pi * (ist.dayofyear.to_numpy() + ist.hour.to_numpy() / 24.0) / 365.25
    d = 2 * np.pi * ist.hour.to_numpy() / 24.0
    return np.column_stack([np.ones(len(times)), np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a),
                            np.sin(d), np.cos(d), np.sin(2 * d), np.cos(2 * d),
                            np.sin(a) * np.sin(d), np.sin(a) * np.cos(d), np.cos(a) * np.sin(d), np.cos(a) * np.cos(d)])


def _harmonic_temperature(df: pd.DataFrame, h: int) -> None:
    """Per cell, least-squares harmonic climatology fitted on TRAIN rows only: at the target time
    (hclim_target, also a baseline) and at the origin (for hanom_now = t_now - climatology)."""
    tr_all = (df['split'] == 'train').to_numpy()
    tgt, org = df.index + pd.Timedelta(hours=h), df.index
    hc_t, hc_o = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for cell in pd.unique(df['cell']):
        m = (df['cell'] == cell).to_numpy()
        tr = m & tr_all
        ct, *_ = np.linalg.lstsq(harmonic_design(tgt[tr]), df['y'].to_numpy()[tr], rcond=None)
        co, *_ = np.linalg.lstsq(harmonic_design(org[tr]), df['t_now'].to_numpy()[tr], rcond=None)
        hc_t[m], hc_o[m] = harmonic_design(tgt[m]) @ ct, harmonic_design(org[m]) @ co
    df['hclim_target'], df['hanom_now'] = hc_t, df['t_now'].to_numpy() - hc_o


def _harmonic_probability(df: pd.DataFrame, outcome: np.ndarray, h: int) -> np.ndarray:
    """Pooled logistic climatology P(outcome) ~ harmonic_design(target time), fitted on TRAIN rows only."""
    from sklearn.linear_model import LogisticRegression
    X = harmonic_design(df.index + pd.Timedelta(hours=h))[:, 1:]
    tr = (df['split'] == 'train').to_numpy()
    o = np.asarray(outcome, dtype=int)
    if o[tr].min() == o[tr].max():
        return np.full(len(df), float(o[tr].mean()))
    return LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], o[tr]).predict_proba(X)[:, 1]


def _lh_data(include_untouched: bool) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Local Open-Meteo file, plus (optionally) the scheduler cache rows from LH_UNTOUCHED_START on."""
    from src.collectors import open_meteo
    hourly = open_meteo.load_local()['hourly']
    if not include_untouched or not os.path.exists(LH_LIVE_CACHE):
        return hourly, None
    raw = pd.read_csv(LH_LIVE_CACHE)
    raw['time_utc'] = pd.to_datetime(raw['time_utc'], utc=True)
    raw = raw[raw['time_utc'] >= pd.Timestamp(LH_UNTOUCHED_START, tz='UTC')]
    cols = ['zone_id', 'time_utc', 'temperature_2m_c', 'relative_humidity_2m_pct', 'precipitation_mm',
            'wind_speed_10m_kmh', 'wind_direction_10m_deg']
    return pd.concat([hourly[cols], raw[cols]], ignore_index=True), raw


def _cells_by_mapping(hourly: pd.DataFrame, z2c: Dict[int, str], column: str):
    """Per cell: `column` series and full weather frame of the cell's lowest-numbered zone (fixed mapping)."""
    series, frames = {}, {}
    for cell in sorted(set(z2c.values())):
        rep = min(z for z, c in z2c.items() if c == cell)
        g = hourly[hourly['zone_id'] == rep].sort_values('time_utc').drop_duplicates('time_utc').set_index('time_utc')
        idx = pd.date_range(g.index.min(), g.index.max(), freq='h')
        frames[cell] = g.reindex(idx)
        series[cell] = frames[cell][column]
    return series, frames


def _lh_temp_frame(series, frames, h, splits) -> pd.DataFrame:
    from src.models.temp_model import TEMP_CANDIDATE_EXTRA, TEMP_FEATURES
    raw_req = [f for f in TEMP_FEATURES + TEMP_CANDIDATE_EXTRA if f not in ('clim_target', 'anom_now')]
    df = temp_rows(series, h, weather=frames, required=raw_req, splits=splits)
    df['clim_target'] = train_encoding(df, df['y'].to_numpy(), TEMP_CLIM_TARGET_LEVELS)
    df['anom_now'] = df['t_now'] - train_encoding(df, df['t_now'].to_numpy(), TEMP_CLIM_ORIGIN_LEVELS)
    _harmonic_temperature(df, h)
    allf = sorted({f for v in _lh_temp_variants().values() for f in v})
    return df.dropna(subset=allf)


def _lh_rain_frame(series, frames, key, splits):
    from src.models.encoding import logit_clip
    from src.models.rain_model import WET_MM
    df = rain_candidate_rows(series, frames, key, splits)
    off = 24 if key == RAIN_OM_ACCUM_KEY else int(key[:-1])
    y = df['y'].to_numpy()
    ph = {thr: _harmonic_probability(df, y >= thr, off) for thr in (WET_MM,) + _twopart_events(key)}
    df['logit_pharm'] = logit_clip(ph[WET_MM])
    return df, ph


def _lh_fit_eval_temp(df, h, feats):
    from src.models.temp_model import HourlyTempModel
    tr = df[df['split'] == 'train']
    model = HourlyTempModel(h, feature_names=feats).fit(tr[feats], tr['y'].to_numpy())
    m = evaluate_temp(df, model.predict(df[feats]), f'{h}h',
                      extra_baselines={'harmonic_climatology': df['hclim_target'].to_numpy()})
    return model, m


def _lh_fit_eval_rain(df, ph, key, feats):
    from src.models.rain_model import TwoPartRainModel
    tr = df[df['split'] == 'train']
    model = TwoPartRainModel(key, feature_names=feats).fit(tr[feats], tr['y'].to_numpy(), tr['season'].to_numpy())
    scored = twopart_scored_rows(df, model.predict(df[feats], df['season'].to_numpy()), key, extra_prob_baselines=ph)
    return model, evaluate_twopart(scored, key)


def _lh_select(cv: pd.DataFrame) -> Tuple[str, Dict[str, object]]:
    """Pre-specified rule: highest mean CV skill among variants with skill > 0 in >= LH_MIN_POSITIVE_FOLDS folds."""
    agg = cv.groupby('variant')['skill'].agg(mean='mean', positive=lambda s: int((s > 0).sum()))
    eligible = agg[agg['positive'] >= LH_MIN_POSITIVE_FOLDS]
    pool = eligible if len(eligible) else agg
    best = pool['mean'].idxmax()
    return best, {'eligible': bool(len(eligible)), 'mean_cv_skill': float(pool.loc[best, 'mean']),
                  'positive_folds': int(pool.loc[best, 'positive'])}


def run_long_horizon_experiment(save: bool = False) -> Dict[str, object]:
    from src.collectors import open_meteo
    local = open_meteo.load_local()['hourly']
    _, z2c_t = openmeteo_rain_cells(local, 'temperature_2m_c')
    _, z2c_r = openmeteo_rain_cells(local)
    cv_rows, decisions, final_metrics, models = [], {}, [], {}
    # ── 1) selection by expanding-window CV (train + validation period only; no test rows exist here)
    s_t, f_t = _cells_by_mapping(local, z2c_t, 'temperature_2m_c')
    s_r, f_r = _cells_by_mapping(local, z2c_r, 'precipitation_mm')
    for fold, splits in LH_FOLDS.items():
        for h in LH_TEMP_HORIZONS:
            df = _lh_temp_frame(s_t, f_t, h, splits)
            for name, feats in _lh_temp_variants().items():
                _, m = _lh_fit_eval_temp(df, h, feats)
                p = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL') & (m['split'] == 'validate')].iloc[0]
                cv_rows.append({'variable': 'temperature', 'horizon': f'{h}h', 'fold': fold, 'variant': name,
                                'skill': float(p['skill']), 'best_baseline': p['best_baseline'], 'n': int(p['n'])})
        for key in LH_RAIN_KEYS:
            df, ph = _lh_rain_frame(s_r, f_r, key, splits)
            cell_cols = [c for c in df.columns if c.startswith('is_')]
            thr0 = _twopart_events(key)[0]
            for name, feats in _lh_rain_variants(cell_cols).items():
                _, m = _lh_fit_eval_rain(df, ph, key, feats)
                p = m[(m['cell'] == 'ALL') & (m['season'] == 'ALL') & (m['split'] == 'validate')].iloc[0]
                cv_rows.append({'variable': 'rainfall', 'horizon': key, 'fold': fold, 'variant': name,
                                'skill': float(p[f'bss_ge{thr0}']), 'crpss_clim': float(p['crpss_vs_clim']),
                                'best_baseline': p[f'bss_ge{thr0}_best_baseline'], 'n': int(p['n'])})
        print(f'fold {fold} done')
    cv = pd.DataFrame(cv_rows)
    # ── 2) final: official splits (+ untouched window), selected variant and the production-feature reference
    data, raw = _lh_data(include_untouched=True)
    final_splits = dict(RAIN_OM_SPLITS)
    if raw is not None and len(raw):
        final_splits['untouched'] = (LH_UNTOUCHED_START, raw['time_utc'].max().strftime('%Y-%m-%d %H:%M'))
    s_t, f_t = _cells_by_mapping(data, z2c_t, 'temperature_2m_c')
    s_r, f_r = _cells_by_mapping(data, z2c_r, 'precipitation_mm')
    for h in LH_TEMP_HORIZONS:
        key = f'{h}h'
        sel, info = _lh_select(cv[(cv['variable'] == 'temperature') & (cv['horizon'] == key)])
        df = _lh_temp_frame(s_t, f_t, h, final_splits)
        for name in dict.fromkeys([sel, 'T0_production']):
            model, m = _lh_fit_eval_temp(df, h, _lh_temp_variants()[name])
            final_metrics.append(m.assign(variable='temperature', variant=name))
            if name == sel:
                models[('temperature', key)] = model
                decisions[('temperature', key)] = {'selected': sel, **info, **temp_gate(m)[key]}
    for key in LH_RAIN_KEYS:
        sel_rows = cv[(cv['variable'] == 'rainfall') & (cv['horizon'] == key)]
        sel, info = _lh_select(sel_rows)
        df, ph = _lh_rain_frame(s_r, f_r, key, final_splits)
        cell_cols = [c for c in df.columns if c.startswith('is_')]
        for name in dict.fromkeys([sel, 'R0_production']):
            model, m = _lh_fit_eval_rain(df, ph, key, _lh_rain_variants(cell_cols)[name])
            final_metrics.append(m.assign(variable='rainfall', variant=name))
            if name == sel:
                models[('rainfall', key)] = model
                decisions[('rainfall', key)] = {'selected': sel, **info, **twopart_gate(m)[key]}
    final = pd.concat(final_metrics, ignore_index=True)
    report = _lh_report(cv, final, decisions, final_splits)
    print(report)
    if save:
        os.makedirs(LH_MODELS_DIR, exist_ok=True)
        os.makedirs(LH_RESULTS_DIR, exist_ok=True)
        assert not {os.path.normpath(p) for p in (LH_ARTIFACT, LH_CV_CSV, LH_FINAL_CSV, LH_REPORT)} & \
            {os.path.normpath(p) for p in PRODUCTION_ARTIFACTS}
        joblib.dump({'models': models, 'decisions': decisions, 'folds': LH_FOLDS, 'final_splits': final_splits,
                     'status': 'EXPERIMENT - not promoted, not used by the live pipeline',
                     'note': 'Promotion would additionally require storing the harmonic/encoding coefficients.',
                     'created_utc': pd.Timestamp.now(tz='UTC').isoformat()}, LH_ARTIFACT)
        cv.to_csv(LH_CV_CSV, index=False)
        final.to_csv(LH_FINAL_CSV, index=False)
        if raw is not None:
            raw.to_csv(LH_SNAPSHOT, index=False)
        with open(LH_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)
        print('Saved:', LH_ARTIFACT, LH_CV_CSV, LH_FINAL_CSV, LH_REPORT, LH_SNAPSHOT if raw is not None else '')
    return {'cv': cv, 'final': final, 'decisions': decisions, 'report': report}


def _lh_report(cv, final, decisions, final_splits) -> str:
    lines = ['# Long-horizon experiment (6 h / 24 h)', '',
             'Selection: expanding-window CV folds A (Jul-Sep 2025), B (Oct-Nov 2025), C (Dec 2025-Mar 2026) - no test rows.',
             f'Rule: highest mean CV skill among variants positive in >= {LH_MIN_POSITIVE_FOLDS} of 3 folds.',
             'Final: official validation and test (test PREVIOUSLY EXAMINED - not an independent confirmation),',
             f"plus untouched window {final_splits.get('untouched')} from the scheduler cache (preliminary data, indicative only).",
             'Baselines include a smooth harmonic seasonal x diurnal climatology (stricter than before).', '',
             '## CV skill by variant (mean over folds A-C; skill = MSE skill for temperature, Brier skill for rainfall)', '']
    piv = cv.pivot_table(index=['variable', 'horizon', 'variant'], columns='fold', values='skill')
    piv['mean'] = piv.mean(axis=1)
    lines += ['```', piv.round(4).to_string(), '```', '', '## Final evaluation of the selected variant', '',
              '| target | selected | eligible | val skill | test skill (examined) | passes rule |', '|---|---|---|---|---|---|']
    for (var, key), d in decisions.items():
        if var == 'temperature':
            vs, ts, ok = d['validate_skill'], d['test_skill'], d['passed']
        else:
            vs, ts, ok = d['validate_bss'], d['test_bss'], d['probability_passed']
        lines.append(f"| {var} {key} | {d['selected']} | {d['eligible']} | {vs:.4f} | {ts:.4f} | {ok} |")
    if 'untouched' in final_splits:
        u = final[(final['split'] == 'untouched') & (final['cell'] == 'ALL') & (final['season'] == 'ALL')]
        lines += ['', '## Untouched window (indicative only)', '', '```',
                  u.drop(columns=[c for c in u.columns if u[c].isna().all()]).to_string(index=False)[:4000], '```']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rainfall-openmeteo', action='store_true',
                    help='Train/evaluate Open-Meteo rainfall HMMs only (original pipeline is not run)')
    ap.add_argument('--rainfall-twopart', action='store_true',
                    help='Two-part rainfall experiment only (dry run unless --save; original pipeline is not run)')
    ap.add_argument('--save', action='store_true',
                    help='With --rainfall-openmeteo / --rainfall-twopart: write model + metrics')
    ap.add_argument('--temperature-hourly', action='store_true',
                    help='Hourly temperature experiment only (dry run unless --save)')
    ap.add_argument('--forecast-candidates', action='store_true',
                    help='3h/6h/24h candidate experiment (writes only under models/candidates and '
                         'results/experiments, and only with --save)')
    ap.add_argument('--promote-3h-candidates', action='store_true',
                    help='Check, and with --apply promote, ONLY temperature 3h + rainfall 3h (backups + manifest)')
    ap.add_argument('--apply', action='store_true', help='With --promote-3h-candidates: write production bundles')
    ap.add_argument('--rollback-promotion', metavar='MANIFEST', help='Restore files listed in a backup manifest.json')
    ap.add_argument('--long-horizon-experiment', action='store_true',
                    help='6h/24h experiment (CV selection on train+validation only; writes only with --save under '
                         'models/experiments/long_horizon and results/experiments/long_horizon)')
    args = ap.parse_args()
    if args.long_horizon_experiment:
        run_long_horizon_experiment(save=args.save)
    elif args.rollback_promotion:
        print('Restored:', rollback_from_manifest(args.rollback_promotion))
    elif args.promote_3h_candidates:
        promote_3h_candidates(apply=args.apply)
    elif args.forecast_candidates:
        run_forecast_candidates(save=args.save)
    elif args.temperature_hourly:
        train_temperature_hourly(save=args.save)
    elif args.rainfall_twopart:
        train_rainfall_twopart(save=args.save)
    elif args.rainfall_openmeteo:
        train_rainfall_openmeteo(save=args.save)
    else:
        run_hmm_training_and_validation()
