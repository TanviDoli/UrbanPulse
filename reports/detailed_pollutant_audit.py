import glob
import os
import pandas as pd
import numpy as np

def audit_pollutants():
    aqi_files = sorted(glob.glob('data/raw/AQI/*.csv'))
    target_prefixes = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3', 'AT', 'RH', 'WS', 'RF', 'TOT-RF']
    
    records = []
    for f in aqi_files:
        fname = os.path.basename(f)
        df = pd.read_csv(f)
        # Drop the artifact row where Station ID == '2025' or Timestamp is null
        valid_mask = df['Timestamp'].notnull() & (df['Station ID'] != '2025')
        df_valid = df[valid_mask]
        
        row_dict = {'file': fname, 'total_rows': len(df), 'valid_rows': len(df_valid)}
        
        # Match columns by prefix
        col_map = {}
        for col in df.columns:
            for p in target_prefixes:
                if col.startswith(p):
                    col_map[p] = col
                    break
                    
        for p in target_prefixes:
            col = col_map.get(p)
            if col is not None:
                series = pd.to_numeric(df_valid[col], errors='coerce')
                non_null = int(series.notnull().sum())
                pct = round(non_null / len(df_valid) * 100, 2)
                min_v = round(float(series.min()), 2) if non_null > 0 else None
                max_v = round(float(series.max()), 2) if non_null > 0 else None
                q01 = round(float(series.quantile(0.01)), 2) if non_null > 0 else None
                q99 = round(float(series.quantile(0.99)), 2) if non_null > 0 else None
                neg_count = int((series < 0).sum()) if non_null > 0 else 0
                row_dict[f'{p}_non_null'] = non_null
                row_dict[f'{p}_pct'] = pct
                row_dict[f'{p}_min'] = min_v
                row_dict[f'{p}_max'] = max_v
                row_dict[f'{p}_q01'] = q01
                row_dict[f'{p}_q99'] = q99
                row_dict[f'{p}_neg'] = neg_count
            else:
                row_dict[f'{p}_non_null'] = 0
                row_dict[f'{p}_pct'] = 0.0
        records.append(row_dict)
    
    df_out = pd.DataFrame(records)
    return df_out

if __name__ == '__main__':
    df_res = audit_pollutants()
    # print summary table
    cols_to_show = ['file', 'valid_rows'] + [c for c in df_res.columns if c.endswith('_pct')]
    print(df_res[cols_to_show].to_string(index=False))
    df_res.to_csv('reports/pollutant_completeness_audit.csv', index=False)
    print("\nSaved detailed pollutant completeness to reports/pollutant_completeness_audit.csv")
