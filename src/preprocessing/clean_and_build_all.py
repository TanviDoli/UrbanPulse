"""
Comprehensive Preprocessing & Master Dataset Builder for UrbanPulse.
1. Cleans all 9 AQI station CSVs (removes '2025' artifact row, reindexes to exact 15-min grid).
2. Computes hourly CPCB AQI using rolling 24h / 8h windows and piecewise sub-index breakpoints.
3. Integrates station-level observed hourly rainfall (tipping bucket) for monitored zones.
4. Segregates historical IMD daily rainfall (1993-2023) into climatological reference.
5. Cleans and integrates IMD daily temperature (1951 - 2024-06-22) with explicit daily provenance flags.
6. Builds data/processed/master_table.csv covering all 15 GCC zones on the hourly operational timeline (2024-2025).
"""

import glob
import os
import re
import pandas as pd
import numpy as np
from src.preprocessing.cpcb_aqi import calculate_sub_index, compute_cpcb_aqi_row
from src.graph.build_zone_graph import GCC_ZONE_MAPPING

def clean_station_file(fpath: str) -> pd.DataFrame:
    """Clean a single raw CAAQMS station file."""
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath)
    
    # 1. Filter out concatenation artifact row ('2025' or null timestamp)
    valid_mask = df['Timestamp'].notnull() & (df['Station ID'].astype(str).str.strip() != '2025')
    df = df[valid_mask].copy()
    
    # 2. Parse timestamps and sort
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])
    
    # 3. Rename columns cleanly by stripping unit corruptions (\ufffd)
    col_map = {}
    for c in df.columns:
        if c.startswith('PM2.5'): col_map[c] = 'PM2.5'
        elif c.startswith('PM10'): col_map[c] = 'PM10'
        elif c.startswith('NO2'): col_map[c] = 'NO2'
        elif c.startswith('SO2'): col_map[c] = 'SO2'
        elif c.startswith('CO'): col_map[c] = 'CO'
        elif c.startswith('Ozone'): col_map[c] = 'Ozone'
        elif c.startswith('NH3'): col_map[c] = 'NH3'
        elif c.startswith('AT'): col_map[c] = 'AT'
        elif c.startswith('RF'): col_map[c] = 'RF'
        elif c.startswith('TOT-RF'): col_map[c] = 'TOT-RF'
        elif c.startswith('RH'): col_map[c] = 'RH'
        elif c.startswith('WS'): col_map[c] = 'WS'
        
    df = df.rename(columns=col_map)
    
    # 4. Set numeric types and clip negative values to NaN
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']
    for p in pollutant_cols:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors='coerce')
            df.loc[df[p] < 0, p] = np.nan
            
    # 5. Sanitize weather columns
    for w in ['AT', 'RH', 'WS', 'RF', 'TOT-RF']:
        if w in df.columns:
            df[w] = pd.to_numeric(df[w], errors='coerce')
            
    # Invalidate Manali Village rainfall sensor drift
    if 'manali-village' in fname:
        if 'RF' in df.columns: df['RF'] = np.nan
        if 'TOT-RF' in df.columns: df['TOT-RF'] = np.nan
        
    # 6. Reindex to uniform 15-minute grid from 2024-01-01 to 2025-12-31
    df = df.set_index('Timestamp')
    full_15m = pd.date_range('2024-01-01 00:00:00+00:00', '2025-12-31 23:45:00+00:00', freq='15min')
    df_clean = df.reindex(full_15m)
    df_clean.index.name = 'timestamp'
    
    return df_clean

def compute_hourly_station_aqi_and_weather(df_15m: pd.DataFrame, station_file: str) -> pd.DataFrame:
    """
    Aggregates 15-minute data to hourly:
    - Computes hourly mean pollutant concentrations (min 2 of 4 15-min intervals).
    - Computes rolling 24-hr means for PM2.5, PM10, NO2, SO2, NH3 (min 16 hours).
    - Computes rolling 8-hr means for CO, Ozone (min 6 hours).
    - Calculates CPCB sub-indices and overall AQI.
    - Computes hourly sum of rainfall (for tipping bucket).
    """
    pollutants_24h = ['PM2.5', 'PM10', 'NO2', 'SO2', 'NH3']
    pollutants_8h = ['CO', 'Ozone']
    all_pollutants = pollutants_24h + pollutants_8h
    
    # 1. Resample to hourly means (min_count=2 out of 4)
    # For rainfall, sum the 15-minute values in the hour (min_count=2)
    hourly_pollutants = df_15m[all_pollutants].resample('1h').agg(
        lambda s: s.mean() if s.count() >= 2 else np.nan
    )
    
    # Rainfall aggregation: sum 15-min rainfall within the hour
    if 'RF' in df_15m.columns and not df_15m['RF'].dropna().empty:
        hourly_rf = df_15m['RF'].resample('1h').agg(
            lambda s: s.sum() if s.count() >= 2 else np.nan
        )
    else:
        hourly_rf = pd.Series(np.nan, index=hourly_pollutants.index)
        
    # 2. Rolling averages according to CPCB specifications
    # 24-hour rolling mean (min_periods=16 hours)
    rolling_24h = hourly_pollutants[pollutants_24h].rolling(window=24, min_periods=16).mean()
    
    # 8-hour rolling mean (min_periods=6 hours)
    rolling_8h = hourly_pollutants[pollutants_8h].rolling(window=8, min_periods=6).mean()
    
    df_rolling = pd.concat([rolling_24h, rolling_8h], axis=1)
    
    # 3. Calculate sub-indices and overall AQI
    aqi_values = []
    aqi_observed = []
    dominant_pollutants = []
    
    for _, row in df_rolling.iterrows():
        sub_dict = {}
        for p in all_pollutants:
            val = row[p]
            if pd.notna(val):
                sub_dict[p] = calculate_sub_index(p, val)
            else:
                sub_dict[p] = np.nan
                
        aqi_val, dom, cnt = compute_cpcb_aqi_row(sub_dict)
        if pd.notna(aqi_val):
            aqi_values.append(round(aqi_val, 2))
            aqi_observed.append(True)
            dominant_pollutants.append(dom)
        else:
            aqi_values.append(np.nan)
            aqi_observed.append(False)
            dominant_pollutants.append(None)
            
    df_hourly = pd.DataFrame({
        'AQI': aqi_values,
        'aqi_observed': aqi_observed,
        'dominant_pollutant': dominant_pollutants,
        'rainfall': hourly_rf,
        'rainfall_observed': hourly_rf.notnull()
    }, index=hourly_pollutants.index)
    
    return df_hourly

def clean_imd_temperature() -> pd.DataFrame:
    """Clean IMD daily temperature dataset (1951 - 2024-06-22)."""
    t_path = 'data/raw/chennai_daily_temperature_1951_2024.csv'
    df = pd.read_csv(t_path)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
    
    # Replace non-numeric tokens
    df['Temp Max'] = pd.to_numeric(df['Temp Max'].replace('-----', np.nan), errors='coerce')
    df['Temp Min'] = pd.to_numeric(df['Temp Min'].replace('-----', np.nan), errors='coerce')
    
    # Fix outlier on 2023-12-01 (Temp Min = 92.0 -> seasonal median 22.5)
    outlier_idx = pd.Timestamp('2023-12-01')
    if outlier_idx in df.index and df.loc[outlier_idx, 'Temp Min'] > 40:
        df.loc[outlier_idx, 'Temp Min'] = 22.5
        
    df['Temp Mean'] = (df['Temp Max'] + df['Temp Min']) / 2.0
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/chennai_temperature_daily_1951_2024.csv')
    return df

def clean_imd_rainfall() -> pd.DataFrame:
    """Clean historical IMD daily rainfall (1993 - 2023)."""
    r_path = 'data/raw/chennai_daily_rainfall_1991_2023.csv'
    df = pd.read_csv(r_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Drop exact duplicate rows
    df = df.drop_duplicates()
    
    # Save clean climatological reference dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/chennai_rainfall_climatology_1993_2023.csv', index=False)
    print(f"Saved {len(df)} historical rainfall records to data/processed/chennai_rainfall_climatology_1993_2023.csv")
    return df

def build_master_table():
    """Build the final master dataset covering all 15 zones over 2024-2025."""
    print("Step 1: Processing IMD Temperature & Rainfall reference files...")
    df_temp_daily = clean_imd_temperature()
    clean_imd_rainfall()
    
    print("Step 2: Cleaning and computing hourly AQI for 9 CAAQMS stations...")
    aqi_files = sorted(glob.glob('data/raw/AQI/*.csv'))
    station_hourly_data = {}
    
    os.makedirs('data/processed/clean_aqi_stations', exist_ok=True)
    for f in aqi_files:
        fname = os.path.basename(f)
        print(f"  Cleaning {fname}...")
        df_15m = clean_station_file(f)
        # Save clean 15-min file
        df_15m.to_csv(f'data/processed/clean_aqi_stations/{fname}')
        
        # Compute hourly AQI and weather
        df_h = compute_hourly_station_aqi_and_weather(df_15m, fname)
        station_hourly_data[fname] = df_h
        
    print("Step 3: Assembling Master Table for 15 GCC Zones across 2024-2025...")
    # Full hourly timeline for 2 years: 2024 (366 days) + 2025 (365 days) = 731 days * 24 = 17,544 hours
    hourly_index = pd.date_range('2024-01-01 00:00:00+00:00', '2025-12-31 23:00:00+00:00', freq='1h')
    
    # Zone to primary station mapping
    zone_station_info = {
        1: ('Thiruvottiyur', 'gandhi-nagar-2024-25.csv', 'Gandhi Nagar_Ennore, Chennai - TNPCB'),
        2: ('Manali', 'manali-2024-25.csv', 'Manali, Chennai - CPCB'),
        3: ('Madhavaram', None, 'None (Unmonitored)'),
        4: ('Tondiarpet', 'kodungaiyur-2024-25.csv', 'Kodungaiyur, Chennai - TNPCB'),
        5: ('Royapuram', 'royapuram-2024-25.csv', 'Royapuram, Chennai - TNPCB'),
        6: ('Thiru-Vi-Ka Nagar', None, 'None (Unmonitored)'),
        7: ('Ambattur', None, 'None (Unmonitored)'),
        8: ('Anna Nagar', None, 'None (Unmonitored)'),
        9: ('Teynampet', None, 'None (Unmonitored)'),
        10: ('Kodambakkam', 'arumbakkam-2024-25.csv', 'Arumbakkam, Chennai - TNPCB'),
        11: ('Valasaravakkam', None, 'None (Unmonitored)'),
        12: ('Alandur', 'alandur-2024-25.csv', 'Alandur Bus Depot, Chennai - CPCB'),
        13: ('Adyar', 'velachery-2024-25.csv', 'Velachery Res. Area, Chennai - CPCB'),
        14: ('Perungudi', 'perungudi-2024-25.csv', 'Perungudi, Chennai - TNPCB'),
        15: ('Sholinganallur', None, 'None (Unmonitored)')
    }
    
    # Pre-extract temperature mapping for each calendar date
    # Daily temperature is observed only up to 2024-06-22
    temp_by_date = {}
    for dt_idx, row in df_temp_daily.iterrows():
        d_str = dt_idx.strftime('%Y-%m-%d')
        temp_by_date[d_str] = {
            'temp_mean': row['Temp Mean'],
            'temp_max': row['Temp Max'],
            'temp_min': row['Temp Min'],
            'observed': True
        }
        
    all_master_rows = []
    
    for zone_id in range(1, 16):
        zone_label, st_file, st_name = zone_station_info[zone_id]
        print(f"  Building Zone {zone_id:02d}: {zone_label} (Station: {st_name})...")
        
        has_station = (st_file is not None) and (st_file in station_hourly_data)
        st_data = station_hourly_data[st_file] if has_station else None
        
        for ts in hourly_index:
            d_str = ts.strftime('%Y-%m-%d')
            
            # 1. AQI
            if has_station and (ts in st_data.index):
                row_st = st_data.loc[ts]
                aqi_val = row_st['AQI']
                aqi_obs = bool(row_st['aqi_observed'])
                dom_pollutant = row_st['dominant_pollutant']
                
                # Rainfall from station gauge
                rf_val = row_st['rainfall']
                rf_obs = bool(row_st['rainfall_observed'])
                rf_res = 'hourly' if rf_obs else 'unobserved'
            else:
                aqi_val = np.nan
                aqi_obs = False
                dom_pollutant = None
                rf_val = np.nan
                rf_obs = False
                rf_res = 'unobserved'
                
            # 2. Temperature (IMD daily background)
            if d_str in temp_by_date and ts <= pd.Timestamp('2024-06-22 23:59:59', tz='UTC'):
                t_info = temp_by_date[d_str]
                temp_val = t_info['temp_mean']
                temp_max = t_info['temp_max']
                temp_min = t_info['temp_min']
                temp_obs = True
                temp_res = 'daily'
            else:
                temp_val = np.nan
                temp_max = np.nan
                temp_min = np.nan
                temp_obs = False
                temp_res = 'unobserved'
                
            all_master_rows.append({
                'timestamp': ts.isoformat(),
                'zone_id': zone_id,
                'zone_label': zone_label,
                'station_name': st_name,
                'AQI': aqi_val,
                'aqi_observed': aqi_obs,
                'dominant_pollutant': dom_pollutant,
                'rainfall': rf_val,
                'rainfall_observed': rf_obs,
                'rainfall_resolution': rf_res,
                'temperature': temp_val,
                'temp_max_daily': temp_max,
                'temp_min_daily': temp_min,
                'temperature_observed': temp_obs,
                'temperature_resolution': temp_res
            })
            
    df_master = pd.DataFrame(all_master_rows)
    master_path = 'data/processed/master_table.csv'
    df_master.to_csv(master_path, index=False)
    print(f"\nSuccessfully generated {master_path} with {len(df_master)} total rows ({len(hourly_index)} hours x 15 zones).")
    return df_master

if __name__ == '__main__':
    build_master_table()
