import pandas as pd
df = pd.read_csv('reports/pollutant_completeness_audit.csv')
for _, r in df.iterrows():
    print(f"=== {r['file']} ===")
    for p in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']:
        print(f"  {p:6s}: min={r[p+'_min']}, max={r[p+'_max']}, q01={r[p+'_q01']}, q99={r[p+'_q99']}, neg_count={r[p+'_neg']}")
