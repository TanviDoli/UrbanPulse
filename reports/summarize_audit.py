import json

with open('reports/audit_raw_results.json') as f:
    d = json.load(f)

print('=== AQI SUMMARY ===')
for fname, info in d['aqi'].items():
    print(f"{fname}: {info['n_rows']} rows, cols={info['n_cols']}, st={info['station_names']}, id={info['station_ids']}")
    print(f"   dates: {info['ts_min']} to {info['ts_max']}, null_ts={info['null_ts']}, exact_dups={info['exact_duplicates']}, ts_dups={info['timestamp_duplicates']}")
    print(f"   top diffs: {info['top_diffs']}")
    
    # Check pollutants and meteorological features
    cols_check = [c for c in info['columns'] if any(k in c for k in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3', 'AT', 'RF', 'RH', 'WS'])]
    null_info = {}
    for c in cols_check:
        clean_name = c.split()[0]
        null_info[clean_name] = f"{info['col_summary'].get(c, {}).get('null_pct')}%"
    print(f"   null pct: {null_info}")

print('\n=== WEATHER SUMMARY ===')
rf = d['weather']['rainfall']
print(f"Rainfall: rows={rf['n_rows']}, dates: {rf['date_min']} to {rf['date_max']}, exact_dups={rf['exact_duplicates']}, stations={len(rf['stations'])}")
print(f"Stations ({len(rf['stations'])}): {rf['stations']}")
print(f"Rainfall stats: {rf['rainfall_stats']}")

temp = d['weather']['temperature']
print(f"Temperature: rows={temp['n_rows']}, dates: {temp['date_min']} to {temp['date_max']}, exact_dups={temp['exact_duplicates']}")
print(f"Temp Max: {temp['Temp_Max']}")
print(f"Temp Min: {temp['Temp_Min']}")
print(f"Rain in temp file: {temp['Rain']}")

print('\n=== GEOJSON SUMMARY ===')
gj = d['geojson']
print(f"GeoJSON type: {gj['type']}, features: {gj['feature_count']}")
for z in gj['zones']:
    print(f"  Zone: {z['properties']['name_of_th']}")
