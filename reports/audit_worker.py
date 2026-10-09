import glob
import json
import os
import pandas as pd
import numpy as np

def audit_aqi_files():
    aqi_files = sorted(glob.glob('data/raw/AQI/*.csv'))
    results = {}
    
    for fpath in aqi_files:
        fname = os.path.basename(fpath)
        print(f"Auditing {fname}...")
        df = pd.read_csv(fpath)
        n_rows, n_cols = df.shape
        col_names = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Station info
        st_ids = df['Station ID'].dropna().unique().tolist() if 'Station ID' in df.columns else []
        st_names = df['Station Name'].dropna().unique().tolist() if 'Station Name' in df.columns else []
        cities = df['City'].dropna().unique().tolist() if 'City' in df.columns else []
        states = df['State'].dropna().unique().tolist() if 'State' in df.columns else []
        
        # Timestamp
        ts_raw = df['Timestamp']
        # Try parsing timestamps
        ts = pd.to_datetime(ts_raw, errors='coerce')
        n_null_ts = int(ts.isnull().sum())
        ts_min = str(ts.min()) if not ts.dropna().empty else None
        ts_max = str(ts.max()) if not ts.dropna().empty else None
        
        # Frequency / intervals
        ts_sorted = ts.dropna().sort_values()
        diffs = ts_sorted.diff()
        top_diffs = {str(k): int(v) for k, v in diffs.value_counts().head(5).items()}
        
        # Duplicates
        n_exact_dups = int(df.duplicated().sum())
        n_ts_dups = int(ts.duplicated().sum())
        
        # Missing values and column summary
        col_summary = {}
        for col in col_names:
            series = df[col]
            null_count = int(series.isnull().sum())
            null_pct = float(null_count / n_rows * 100)
            
            # Numeric stats
            num_series = pd.to_numeric(series, errors='coerce')
            num_null = int(num_series.isnull().sum())
            
            summary = {
                'null_count': null_count,
                'null_pct': round(null_pct, 2),
                'dtype': str(series.dtype)
            }
            if num_null < n_rows:
                # has numeric data
                non_null_num = num_series.dropna()
                summary['min'] = float(non_null_num.min())
                summary['max'] = float(non_null_num.max())
                summary['mean'] = round(float(non_null_num.mean()), 2)
                summary['q01'] = round(float(non_null_num.quantile(0.01)), 2)
                summary['q99'] = round(float(non_null_num.quantile(0.99)), 2)
                summary['negative_count'] = int((non_null_num < 0).sum())
            col_summary[col] = summary
            
        results[fname] = {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'columns': col_names,
            'dtypes': dtypes,
            'station_ids': st_ids,
            'station_names': st_names,
            'cities': cities,
            'states': states,
            'ts_min': ts_min,
            'ts_max': ts_max,
            'null_ts': n_null_ts,
            'top_diffs': top_diffs,
            'exact_duplicates': n_exact_dups,
            'timestamp_duplicates': n_ts_dups,
            'col_summary': col_summary
        }
    return results

def audit_weather_files():
    weather_results = {}
    
    # Rainfall
    rf_path = 'data/raw/chennai_daily_rainfall_1991_2023.csv'
    print(f"Auditing {rf_path}...")
    df_rf = pd.read_csv(rf_path)
    n_rows, n_cols = df_rf.shape
    exact_dups = int(df_rf.duplicated().sum())
    
    # Date parsing
    dates = pd.to_datetime(df_rf['Date'], format='%d-%m-%Y', errors='coerce')
    date_min = str(dates.min())
    date_max = str(dates.max())
    null_dates = int(dates.isnull().sum())
    stations = df_rf['Station'].unique().tolist()
    
    # Check date frequency per station
    st_date_counts = df_rf.groupby('Station')['Date'].count().to_dict()
    
    weather_results['rainfall'] = {
        'path': rf_path,
        'n_rows': n_rows,
        'n_cols': n_cols,
        'columns': df_rf.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df_rf.dtypes.items()},
        'exact_duplicates': exact_dups,
        'date_min': date_min,
        'date_max': date_max,
        'null_dates': null_dates,
        'stations': stations,
        'st_counts': st_date_counts,
        'rainfall_stats': {
            'min': float(df_rf['Rainfall'].min()),
            'max': float(df_rf['Rainfall'].max()),
            'mean': round(float(df_rf['Rainfall'].mean()), 2),
            'null_count': int(df_rf['Rainfall'].isnull().sum()),
            'negative_count': int((df_rf['Rainfall'] < 0).sum()),
            'zero_count': int((df_rf['Rainfall'] == 0).sum())
        }
    }
    
    # Temperature
    temp_path = 'data/raw/chennai_daily_temperature_1951_2024.csv'
    print(f"Auditing {temp_path}...")
    df_temp = pd.read_csv(temp_path)
    n_rows_t, n_cols_t = df_temp.shape
    exact_dups_t = int(df_temp.duplicated().sum())
    
    dates_t = pd.to_datetime(df_temp['Date'], format='%d-%m-%Y', errors='coerce')
    date_min_t = str(dates_t.min())
    date_max_t = str(dates_t.max())
    null_dates_t = int(dates_t.isnull().sum())
    
    # Analyze non-numeric values in Temp Max, Temp Min, Rain
    def check_col(s):
        null_raw = int(s.isnull().sum())
        num = pd.to_numeric(s, errors='coerce')
        invalid = s[num.isnull() & s.notnull()].value_counts().to_dict()
        return {
            'null_raw': null_raw,
            'invalid_values': invalid,
            'numeric_valid_count': int(num.notnull().sum()),
            'min': float(num.min()) if num.notnull().sum() > 0 else None,
            'max': float(num.max()) if num.notnull().sum() > 0 else None,
            'mean': round(float(num.mean()), 2) if num.notnull().sum() > 0 else None
        }
    
    weather_results['temperature'] = {
        'path': temp_path,
        'n_rows': n_rows_t,
        'n_cols': n_cols_t,
        'columns': df_temp.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df_temp.dtypes.items()},
        'exact_duplicates': exact_dups_t,
        'date_min': date_min_t,
        'date_max': date_max_t,
        'null_dates': null_dates_t,
        'Rain': check_col(df_temp['Rain']),
        'Temp_Max': check_col(df_temp['Temp Max']),
        'Temp_Min': check_col(df_temp['Temp Min'])
    }
    
    return weather_results

def audit_geojson():
    with open('data/raw/chennai_zones_2025.geojson', 'r') as f:
        gj = json.load(f)
    features = gj.get('features', [])
    zone_list = []
    for feat in features:
        props = feat.get('properties', {})
        geom = feat.get('geometry', {})
        zone_list.append({
            'properties': props,
            'geom_type': geom.get('type'),
            'coord_len': len(geom.get('coordinates', []))
        })
    return {
        'type': gj.get('type'),
        'feature_count': len(features),
        'zones': zone_list
    }

if __name__ == '__main__':
    all_audit = {
        'aqi': audit_aqi_files(),
        'weather': audit_weather_files(),
        'geojson': audit_geojson()
    }
    with open('reports/audit_raw_results.json', 'w') as f:
        json.dump(all_audit, f, indent=2)
    print("Audit completed successfully. Saved to reports/audit_raw_results.json")
