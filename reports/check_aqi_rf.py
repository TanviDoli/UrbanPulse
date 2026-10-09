import glob
import os
import pandas as pd

aqi_files = sorted(glob.glob('data/raw/AQI/*.csv'))
for f in aqi_files:
    fname = os.path.basename(f)
    df = pd.read_csv(f)
    df = df[df['Timestamp'].notnull() & (df['Station ID'] != '2025')]
    rf_col = [c for c in df.columns if c.startswith('RF')][0]
    s = pd.to_numeric(df[rf_col], errors='coerce')
    non_null = s.notnull().sum()
    pos = (s > 0).sum()
    print(f"{fname:26s} | non-null={non_null:5d} ({non_null/len(df)*100:5.1f}%) | pos={pos:5d} | max={s.max()} | sum={s.sum():.1f} mm")
