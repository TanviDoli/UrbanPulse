import glob
import os
import pandas as pd

aqi_files = sorted(glob.glob('data/raw/AQI/*.csv'))
expected_steps = 70176 # 366*96 + 365*96

for f in aqi_files:
    fname = os.path.basename(f)
    df = pd.read_csv(f)
    # clean artifact row
    df = df[df['Timestamp'].notnull() & (df['Station ID'] != '2025')]
    ts = pd.to_datetime(df['Timestamp'])
    full_idx = pd.date_range('2024-01-01 00:00:00+00:00', '2025-12-31 23:45:00+00:00', freq='15min')
    missing_in_grid = full_idx.difference(ts)
    print(f"{fname:26s} | Valid rows: {len(df):5d} | Expected: {expected_steps} | Missing grid steps: {len(missing_in_grid):5d} | Dups in ts: {ts.duplicated().sum()}")
