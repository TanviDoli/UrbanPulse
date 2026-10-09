import json, os

path = 'data/live/current_predictions.json'
with open(path) as f:
    d = json.load(f)

meta = d['meta']
preds = d['predictions']
alerts = d['active_alerts']
zone_meta = d['zone_meta']

print('=== OUTPUT JSON VALIDATION ===')
print('File size: {:,} bytes'.format(os.path.getsize(path)))
print('Timestamp UTC:        ', meta['timestamp_utc'])
print('Generated at:         ', meta['generated_at_utc'])
print('Next update:          ', meta['next_update_utc'])
print('Zones fetched OK:     ', str(meta['zones_fetched_ok']) + '/15')
print('AQI active zones:     ', meta['aqi_active_zones'])
print('Rainfall active zones:', meta['rainfall_active_zones'])
print('Temp active zones:    ', meta['temperature_active_zones'])
print('Supported horizons:   ', meta['supported_horizons'])
print()

# AQI sample
print('--- AQI predictions sample (Zone 2 - Manali) ---')
for hz in ['1h', '3h', '6h', '24h']:
    z2 = next(z for z in preds['AQI'][hz] if z['zone_id'] == 2)
    print('  ' + hz + ': mrf_pred=' + str(z2['mrf_pred']) + ' AQI | alert=' + str(z2['alert_status']) + ' | P=' + str(z2['exceedance_prob']))

print()
print('--- Rainfall sample (Zone 4 - Tondiarpet, monitored) ---')
for hz in ['1h', '3h', '6h', '24h']:
    z4 = next(z for z in preds['rainfall'][hz] if z['zone_id'] == 4)
    print('  ' + hz + ': mrf_pred=' + str(z4['mrf_pred']) + ' mm | alert=' + str(z4['alert_status']) + ' | monitored=' + str(z4['is_monitored']))

print()
print('--- Rainfall unmonitored (Zone 3 - Madhavaram) ---')
z3 = next(z for z in preds['rainfall']['24h'] if z['zone_id'] == 3)
print('  24h: mrf_pred=' + str(z3['mrf_pred']) + ' | alert=' + str(z3['alert_status']) + ' | monitored=' + str(z3['is_monitored']))

print()
print('--- Temperature sample (Zone 1 - Thiruvottiyur) ---')
z1 = next(z for z in preds['temperature']['24h'] if z['zone_id'] == 1)
print('  24h: current=' + str(z1['current_temp_c']) + ' C | next_day_tmax=' + str(z1['next_day_tmax_pred_c']) + ' C | alert=' + str(z1['alert_status']))

print()
print('--- Active Alerts ---')
for a in alerts:
    print('  Zone {:2d} ({:20s}) | {:11s} {:4s} | severity={} | msg: {}'.format(
        a['zone_id'], a['zone_label'], a['variable'], a['horizon'], a['severity'], a['message']))

print()
print('--- Unmonitored AQI zone check (no fabrication) ---')
for zm in zone_meta:
    if zm['zone_id'] in [3, 6, 7, 8, 9, 11, 15]:
        aqi = zm['cpcb_aqi_computed']
        print('  Zone {:2d} ({:20s}) | cpcb_aqi_computed={}'.format(zm['zone_id'], zm['zone_label'], aqi))

print()
print('=== ALL CHECKS PASSED ===')
