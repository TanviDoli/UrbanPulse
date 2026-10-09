"""
Phase D — Combined audit: local CAAQMS files vs OpenAQ vs Open-Meteo, plus checks of the
frozen models on the inputs the live pipeline will use. Nothing here fits a model.

Run (after src.audit.openaq_audit and src.audit.open_meteo_audit):
    python -m src.audit.combined_audit
Outputs:
    reports/unit_validation.csv                    (consumed by the live pipeline)
    reports/coverage/*.csv                         machine-readable coverage tables
    reports/coverage/combined_audit_results.json   numbers quoted in final_data_audit.md
"""

import glob
import json
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.audit.openaq_audit import load_zone_polygons
from src.collectors.open_meteo import load_local, daily_mean_temperature
from src.preprocessing.cpcb_aqi import aqi_from_hourly_concentrations

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE = os.path.join(ROOT, 'data', 'external', 'openaq', 'hours')
COV_DIR = os.path.join(ROOT, 'reports', 'coverage')
IST = 'Asia/Kolkata'

LOCAL_TO_OPENAQ = {
    'manali-2024-25.csv': 2586, 'manali-village-2024-25.csv': 10780,
    'kodungaiyur-2024-25.csv': 11579, 'royapuram-2024-25.csv': 11578,
    'arumbakkam-2024-25.csv': 11581, 'alandur-2024-25.csv': 3409319,
    'velachery-2024-25.csv': 5655, 'perungudi-2024-25.csv': 12046,
    'gandhi-nagar-2024-25.csv': None,
}
OPENAQ_TO_CPCB = {'pm25': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO2', 'so2': 'SO2',
                  'co': 'CO', 'o3': 'Ozone', 'nh3': 'NH3'}
MW = {'NO2': 46.0055, 'SO2': 64.066, 'CO': 28.010, 'Ozone': 47.997, 'NH3': 17.031}
ZONE_RAIN_STATION = {4: 'kodungaiyur-2024-25.csv', 5: 'royapuram-2024-25.csv',
                     10: 'arumbakkam-2024-25.csv', 14: 'perungudi-2024-25.csv'}
ZONE_AQI_STATION = {2: 2586, 4: 11579, 5: 11578, 10: 11581, 12: 3409319, 13: 5655, 14: 12046}
TEST_START, TEST_END = '2025-09-14', '2025-12-31'   # existing test partition (reports/data_split.md)


def local_hourly(fname: str, as_tz: str = IST) -> pd.DataFrame:
    """Local clean 15-min file -> hourly means (>=2 of 4), timestamps re-labelled in `as_tz`."""
    df = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'clean_aqi_stations', fname), index_col=0)
    idx = pd.to_datetime(df.index.str.slice(0, 19))           # drop the (incorrect) +00:00 label
    df.index = idx.tz_localize(as_tz)
    cols = [c for c in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3', 'AT', 'RF'] if c in df]
    num = df[cols].apply(pd.to_numeric, errors='coerce')
    agg = num.drop(columns=[c for c in ['RF'] if c in num]).resample('1h').agg(
        lambda s: s.mean() if s.count() >= 2 else np.nan)
    if 'RF' in num:
        agg['RF'] = num['RF'].resample('1h').agg(lambda s: s.sum() if s.count() >= 2 else np.nan)
    return agg


def openaq_series(sensor_id: int) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(CACHE, f'sensor_{sensor_id}_*.csv')))
    frames = [f for f in (pd.read_csv(p) for p in fs) if len(f)]
    if not frames:
        return pd.DataFrame(columns=['t', 'value', 'observed_count', 'units'])
    d = pd.concat(frames, ignore_index=True).dropna(subset=['value'])
    d['t'] = pd.to_datetime(d['hour_start_utc'], utc=True).dt.tz_convert(IST).dt.floor('h')
    return d.drop_duplicates('t')


def unit_validation(sensors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fname, lid in LOCAL_TO_OPENAQ.items():
        if lid is None:
            continue
        loc_ist = local_hourly(fname, IST)
        loc_utc = local_hourly(fname, 'UTC')
        for _, s in sensors[(sensors['location_id'] == lid) & sensors['parameter'].isin(OPENAQ_TO_CPCB)].iterrows():
            pol = OPENAQ_TO_CPCB[s['parameter']]
            o = openaq_series(int(s['sensor_id']))
            rec = {'sensor_id': int(s['sensor_id']), 'location_id': lid, 'local_file': fname,
                   'parameter': s['parameter'], 'cpcb_pollutant': pol, 'openaq_units_label': s['units'],
                   'openaq_last_utc': s['metadata_last_utc']}
            if pol not in loc_ist or o.empty:
                rec.update(n_overlap=0, verdict='insufficient_overlap')
                rows.append(rec)
                continue
            oo = o.set_index('t')['value'].astype(float)
            j = pd.concat([loc_ist[pol].rename('local'), oo.rename('oaq')], axis=1, join='inner').dropna()
            lu = loc_utc[pol].tz_convert(IST)
            lu.index = lu.index.floor('h')
            lu = lu.groupby(level=0).mean()
            ju = pd.concat([lu.rename('local'), oo.rename('oaq')], axis=1,
                           join='inner').dropna()
            rec['n_overlap'] = len(j)
            if len(j) < 200:
                rec['verdict'] = 'insufficient_overlap'
                rows.append(rec)
                continue
            jj = j[j['local'] > j['local'].quantile(0.1)]
            ratio = float((jj['oaq'] / jj['local']).median())
            corr = float(j.corr().iloc[0, 1])
            rec.update(corr_local_as_ist=round(corr, 3),
                       corr_local_as_utc=round(float(ju.corr().iloc[0, 1]), 3) if len(ju) > 200 else None,
                       median_ratio_openaq_over_local=round(ratio, 4),
                       ppb_to_mass_factor=(round(MW[pol] / 24.45 / (1000 if pol == 'CO' else 1), 4)
                                           if pol in MW else None))
            if corr >= 0.8 and 0.9 <= ratio <= 1.1:
                rec['verdict'] = 'native_cpcb_units'
            elif pol in MW and corr >= 0.8 and abs(ratio * rec['ppb_to_mass_factor'] - 1) < 0.1:
                rec['verdict'] = 'true_ppb'
            else:
                rec['verdict'] = 'mismatch'
            rows.append(rec)
    return pd.DataFrame(rows)


def station_mapping_reconciliation(locations: pd.DataFrame) -> pd.DataFrame:
    polys = load_zone_polygons()
    mp = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'station_zone_mapping.csv'))
    rows = []
    for _, r in mp[mp['station_file'].notna() & (mp['station_file'] != 'None')].iterrows():
        def pip(lat, lon):
            for (z, _), poly in polys:
                if poly.contains(Point(lon, lat)):
                    return z
            return None
        lid = LOCAL_TO_OPENAQ.get(r['station_file'])
        oq = locations[locations['location_id'] == lid].iloc[0] if lid in set(locations['location_id']) else None
        rows.append({
            'station_name': r['station_name'], 'station_file': r['station_file'],
            'mapped_zone_id': int(r['zone_id']), 'local_lat': r['latitude'], 'local_lon': r['longitude'],
            'local_coord_pip_zone': pip(r['latitude'], r['longitude']),
            'openaq_location_id': lid,
            'openaq_lat': None if oq is None else oq['latitude'], 'openaq_lon': None if oq is None else oq['longitude'],
            'openaq_coord_pip_zone': None if oq is None else pip(oq['latitude'], oq['longitude']),
            'separation_km': None if oq is None else round(float(np.hypot(
                (r['latitude'] - oq['latitude']) * 111, (r['longitude'] - oq['longitude']) * 108)), 2),
        })
    df = pd.DataFrame(rows)
    df['consistent'] = df.apply(lambda x: x['local_coord_pip_zone'] == x['mapped_zone_id'] and
                                (x['openaq_coord_pip_zone'] in (None, x['mapped_zone_id']) or pd.isna(x['openaq_location_id'])),
                                axis=1)
    return df


def local_raw_rf(fname: str) -> pd.Series:
    df = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'clean_aqi_stations', fname), usecols=['RF'])
    return pd.to_numeric(df['RF'], errors='coerce').dropna()


def rain_comparison(om_hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z, fname in ZONE_RAIN_STATION.items():
        g = local_hourly(fname, IST)['RF']
        # Open-Meteo precip is the sum over the hour ENDING at `time`; shift to hour start, IST hour
        o = om_hourly[om_hourly['zone_id'] == z].set_index('time_utc')['precipitation_mm']
        o.index = (o.index - pd.Timedelta(hours=1)).tz_convert(IST).floor('h')   # :30 IST -> IST hour (30 min offset)
        j = pd.concat([g.rename('gauge'), o.rename('om')], axis=1).loc['2025-01-01':'2025-12-31']
        jh = j.dropna()
        gd = g.loc['2025-01-01':'2025-12-31'].resample('D').agg(lambda s: s.sum() if s.count() >= 20 else np.nan)
        od = o.loc['2025-01-01':'2025-12-31'].resample('D').sum(min_count=24)
        jd = pd.concat([gd.rename('gauge'), od.rename('om')], axis=1).dropna()
        rows.append({
            'zone_id': z, 'gauge_station': fname, 'hours_overlap': len(jh), 'days_overlap': len(jd),
            'daily_corr': round(float(jd.corr().iloc[0, 1]), 3),
            'gauge_total_mm_overlap_days': round(float(jd['gauge'].sum()), 1),
            'om_total_mm_overlap_days': round(float(jd['om'].sum()), 1),
            'gauge_wet_hour_frac': round(float((jh['gauge'] > 0.1).mean()), 3),
            'om_wet_hour_frac': round(float((jh['om'] > 0.1).mean()), 3),
            'gauge_max_hour_mm': round(float(jh['gauge'].max()), 1), 'om_max_hour_mm': round(float(jh['om'].max()), 1),
            'gauge_days_ge_64_5': int((jd['gauge'] >= 64.5).sum()), 'om_days_ge_64_5': int((jd['om'] >= 64.5).sum()),
            'gauge_15min_frac_nonzero': round(float((local_raw_rf(fname) > 0).mean()), 3),
            'om_max_day_mm': round(float(jd['om'].max()), 1),
            'gauge_on_om_max_day_mm': round(float(jd.loc[jd['om'].idxmax(), 'gauge']), 1),
            'om_max_day': str(jd['om'].idxmax().date()),
        })
    return pd.DataFrame(rows)


def temperature_comparison(om_hourly: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    """Open-Meteo 2 m temperature vs measured station temperature (OpenAQ 'temperature' sensors)."""
    loc_zone = {2586: 2, 10780: 2, 11579: 4, 11578: 5, 11581: 10, 3409319: 12, 5655: 13, 12046: 14}
    rows = []
    for _, s in sensors[(sensors['parameter'] == 'temperature') & sensors['location_id'].isin(loc_zone)].iterrows():
        o = openaq_series(int(s['sensor_id']))
        if o.empty:
            continue
        z = loc_zone[int(s['location_id'])]
        st = o.set_index('t')['value'].astype(float)
        st = st[(st > 5) & (st < 50)]
        om = om_hourly[om_hourly['zone_id'] == z].set_index('time_utc')['temperature_2m_c']
        om.index = om.index.tz_convert(IST).floor('h')        # instantaneous value at :00Z (= :30 IST)
        j = pd.concat([st.rename('station'), om.rename('om')], axis=1, join='inner').dropna()
        if len(j) < 200:
            continue
        dj = j.resample('D').mean()
        rows.append({'sensor_id': int(s['sensor_id']), 'location_name': s['location_name'], 'zone_id': z,
                     'hours_overlap': len(j), 'hourly_corr': round(float(j.corr().iloc[0, 1]), 3),
                     'mean_bias_om_minus_station_c': round(float((j['om'] - j['station']).mean()), 2),
                     'hourly_mae_c': round(float((j['om'] - j['station']).abs().mean()), 2),
                     'daily_mean_mae_c': round(float((dj['om'] - dj['station']).abs().mean()), 2)})
    return pd.DataFrame(rows)


def eval_rain_hmm(om_hourly: pd.DataFrame) -> pd.DataFrame:
    """Frozen rainfall HMM on the existing TEST partition: gauge input vs Open-Meteo input, gauge truth."""
    rows = []
    for z, fname in ZONE_RAIN_STATION.items():
        m = joblib.load(os.path.join(ROOT, 'models', 'hmm', f'hmm_rainfall_zone_{z}.joblib'))
        g = local_hourly(fname, IST)['RF'].loc[TEST_START:TEST_END]
        o = om_hourly[om_hourly['zone_id'] == z].set_index('time_utc')['precipitation_mm']
        o.index = (o.index - pd.Timedelta(hours=1)).tz_convert(IST) - pd.Timedelta(minutes=30)
        o = o.reindex(g.index)                                 # align to the gauge IST hour grid
        for label, inp in [('gauge', g), ('open_meteo', o)]:
            for h in [1, 24]:
                yhat, _ = m.filter_and_forecast_sequence(inp.values.astype(float), horizon=h)
                pred = pd.Series(yhat, index=g.index).shift(h)       # forecast issued at t for t+h
                j = pd.concat([g.rename('y'), pred.rename('p')], axis=1).dropna()
                rows.append({'zone_id': z, 'input': label, 'horizon_h': h, 'n': len(j),
                             'mae_mm': round(float((j['y'] - j['p']).abs().mean()), 3),
                             'rmse_mm': round(float(np.sqrt(((j['y'] - j['p']) ** 2).mean())), 3),
                             'corr': round(float(j.corr().iloc[0, 1]), 3)})
    return pd.DataFrame(rows)


def eval_temp_hmm(om_hourly: pd.DataFrame) -> Dict:
    """Frozen daily-mean temperature HMM, next-day forecasts vs Open-Meteo daily means (2025-2026)."""
    m = joblib.load(os.path.join(ROOT, 'models', 'hmm', 'hmm_temperature_zone_1.joblib'))
    d = daily_mean_temperature(om_hourly)
    out = {}
    for z in [1, 7, 12]:
        s = d[d['zone_id'] == z].sort_values('date_ist')['tmean_c'].values.astype(float)
        yhat, _ = m.filter_and_forecast_sequence(s, horizon=1)
        y, p = s[1:], yhat[:-1]
        ok = ~np.isnan(y) & ~np.isnan(p)
        pers = np.abs(s[1:] - s[:-1])[ok]
        out[z] = {'n_days': int(ok.sum()), 'mae_c': round(float(np.abs(y[ok] - p[ok]).mean()), 3),
                  'rmse_c': round(float(np.sqrt(((y[ok] - p[ok]) ** 2).mean())), 3),
                  'reference_day_to_day_change_mae_c': round(float(pers.mean()), 3),
                  'om_tmean_range_c': [round(float(np.nanmin(s)), 1), round(float(np.nanmax(s)), 1)],
                  'hmm_state_means_c': [round(float(x), 1) for x in m.means_]}
    return out


def aqi_checks(validated: set) -> Dict:
    """(a) OpenAQ-derived AQI vs master_table AQI for 2025; (b) frozen AQI HMM on 2026 OpenAQ AQI."""
    master = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'master_table.csv'),
                         usecols=['timestamp', 'zone_id', 'AQI'])
    master['t'] = pd.to_datetime(master['timestamp'].str.slice(0, 19)).dt.tz_localize(IST)
    sens = pd.read_csv(os.path.join(ROOT, 'reports', 'openaq_audit', 'sensors.csv'))
    res = {}
    for z, lid in ZONE_AQI_STATION.items():
        frames = {}
        for _, s in sens[(sens['location_id'] == lid) & sens['parameter'].isin(OPENAQ_TO_CPCB)].iterrows():
            if int(s['sensor_id']) not in validated:
                continue
            o = openaq_series(int(s['sensor_id']))
            o = o[(o['value'] >= 0) & (pd.to_numeric(o['observed_count'], errors='coerce').fillna(4) >= 2)]
            pol = OPENAQ_TO_CPCB[s['parameter']]
            ser = o.set_index('t')['value'].astype(float)
            frames[pol] = ser if pol not in frames else frames[pol].combine_first(ser)
        if not frames:
            res[z] = {'note': 'no validated sensors'}
            continue
        conc = pd.DataFrame(frames)
        conc = conc.reindex(pd.date_range(conc.index.min(), conc.index.max(), freq='h'))
        aqi = aqi_from_hourly_concentrations(conc)['AQI']
        mz = master[master['zone_id'] == z].set_index('t')['AQI']
        j = pd.concat([aqi.rename('oaq'), mz.rename('master')], axis=1, join='inner').dropna()
        m = joblib.load(os.path.join(ROOT, 'models', 'hmm', f'hmm_aqi_zone_{z}.joblib'))
        a26 = aqi.loc['2026-01-01':]
        ev = {}
        for h in [1, 24]:
            if a26.notna().sum() < 100:
                break
            yhat, _ = m.filter_and_forecast_sequence(a26.values.astype(float), horizon=h)
            p = pd.Series(yhat, index=a26.index).shift(h)
            jj = pd.concat([a26.rename('y'), p.rename('p')], axis=1).dropna()
            ev[f'{h}h'] = {'n': len(jj), 'mae': round(float((jj['y'] - jj['p']).abs().mean()), 2),
                           'rmse': round(float(np.sqrt(((jj['y'] - jj['p']) ** 2).mean())), 2)}
        res[z] = {'station_location_id': lid,
                  'overlap_hours_2025': len(j),
                  'mae_vs_master_2025': round(float((j['oaq'] - j['master']).abs().mean()), 2) if len(j) else None,
                  'corr_vs_master_2025': round(float(j.corr().iloc[0, 1]), 3) if len(j) > 10 else None,
                  'aqi_hours_2026': int(a26.notna().sum()),
                  'aqi_last_valid_ist': str(aqi.dropna().index.max()) if aqi.notna().any() else None,
                  'aqi_2026_mean': round(float(a26.mean()), 1) if a26.notna().any() else None,
                  'aqi_2026_hours_gt_200': int((a26 > 200).sum()),
                  'hmm_eval_2026_out_of_sample': ev}
    return res


def zone_variable_coverage(unit_val: pd.DataFrame, aqi_res: Dict) -> pd.DataFrame:
    labels = {1: 'Thiruvottiyur', 2: 'Manali', 3: 'Madhavaram', 4: 'Tondiarpet', 5: 'Royapuram',
              6: 'Thiru-Vi-Ka Nagar', 7: 'Ambattur', 8: 'Anna Nagar', 9: 'Teynampet', 10: 'Kodambakkam',
              11: 'Valasaravakkam', 12: 'Alandur', 13: 'Adyar', 14: 'Perungudi', 15: 'Sholinganallur'}
    rows = []
    for z in range(1, 16):
        ar = aqi_res.get(z, {})
        last = ar.get('aqi_last_valid_ist')
        if z in ZONE_AQI_STATION:
            aq_hist = 'measured (local CAAQMS 2024-2025)'
            aq_live = f'measured (OpenAQ, last valid {last[:16]} IST)' if last else 'unavailable'
            aq_model = 'AQI HMM + MRF'
        elif z == 1:
            aq_hist, aq_live, aq_model = ('measured but ~90% missing (Gandhi Nagar)',
                                          'unavailable (station not on OpenAQ)', 'MRF estimate only (no HMM)')
        else:
            aq_hist, aq_live, aq_model = 'unavailable (no station)', 'unavailable (no station)', 'MRF estimate only'
        rows.append({
            'zone_id': z, 'zone_label': labels[z],
            'aq_2024_2025': aq_hist, 'aq_2026_live': aq_live, 'aq_model': aq_model,
            'rain_measured': 'gauge 2024-2025 (local CAAQMS RF)' if z in ZONE_RAIN_STATION else 'unavailable',
            'rain_reanalysis': 'Open-Meteo hourly 2025-01-01 -> present',
            'rain_model': 'rainfall HMM (reanalysis-driven)' if z in ZONE_RAIN_STATION else 'none (no trained model)',
            'temp_measured': 'IMD daily (city-wide) to 2024-06-22; station AT where OpenAQ has a sensor',
            'temp_reanalysis': 'Open-Meteo hourly 2025-01-01 -> present',
            'temp_model': 'daily-mean HMM, next day only',
        })
    return pd.DataFrame(rows)


def run():
    os.makedirs(COV_DIR, exist_ok=True)
    sensors = pd.read_csv(os.path.join(ROOT, 'reports', 'openaq_audit', 'sensors.csv'))
    locations = pd.read_csv(os.path.join(ROOT, 'reports', 'openaq_audit', 'locations.csv'))
    om = load_local()['hourly']

    print('[combined] unit validation ...')
    uv = unit_validation(sensors)
    uv.to_csv(os.path.join(ROOT, 'reports', 'unit_validation.csv'), index=False)
    validated = set(uv.loc[uv['verdict'] == 'native_cpcb_units', 'sensor_id'].astype(int))
    print('[combined] station mapping ...')
    sm = station_mapping_reconciliation(locations)
    sm.to_csv(os.path.join(COV_DIR, 'station_mapping_reconciliation.csv'), index=False)
    print('[combined] rainfall comparison ...')
    rc = rain_comparison(om)
    rc.to_csv(os.path.join(COV_DIR, 'rain_open_meteo_vs_gauge_2025.csv'), index=False)
    print('[combined] temperature comparison ...')
    tc = temperature_comparison(om, sensors)
    tc.to_csv(os.path.join(COV_DIR, 'temperature_open_meteo_vs_station.csv'), index=False)
    print('[combined] rainfall HMM evaluation ...')
    rh = eval_rain_hmm(om)
    rh.to_csv(os.path.join(COV_DIR, 'rain_hmm_test_eval_by_input.csv'), index=False)
    print('[combined] temperature HMM evaluation ...')
    th = eval_temp_hmm(om)
    print('[combined] AQI checks ...')
    aq = aqi_checks(validated)
    zc = zone_variable_coverage(uv, aq)
    zc.to_csv(os.path.join(COV_DIR, 'zone_variable_coverage.csv'), index=False)
    results = {'temperature_hmm_eval_vs_open_meteo': th, 'aqi_checks': aq,
               'n_validated_sensors': len(validated),
               'unit_verdict_counts': uv['verdict'].value_counts().to_dict()}
    with open(os.path.join(COV_DIR, 'combined_audit_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print('[combined] done')


if __name__ == '__main__':
    run()
