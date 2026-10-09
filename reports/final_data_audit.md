# UrbanPulse Chennai — Final Data Audit and Implementation Plan

Audit date: 2026-10-09. Every number below was computed from the files and API responses named
here; scripts are in `src/audit/`, machine-readable outputs in `reports/coverage/`,
`reports/openaq_audit/`, `reports/open_meteo_audit/` and `reports/unit_validation.csv`.

Provenance vocabulary used throughout:

| label | meaning |
|---|---|
| **measured** | a physical instrument reading (CAAQMS station via local CSV or OpenAQ; IMD record) |
| **reanalysis** | Open-Meteo Historical Weather API (ERA5 family / ECMWF IFS analysis; most recent ~5 days preliminary) |
| **model estimate** | produced by UrbanPulse HMM/MRF, never shown as an observation |
| **unavailable** | no data or no supported model; shown as unavailable, never as 0 or "low risk" |

---

## 1. Inventory (Phase A)

| item | path | status |
|---|---|---|
| Zone polygons | `data/raw/chennai_zones_2025.geojson` | 15 valid polygons, property `name_of_th`. Feature order ≠ zone number. Zone 12's centroid lies 0.2 km outside its (concave) polygon, in no other zone. |
| 9 station CSVs | `data/raw/AQI/*.csv` → cleaned `data/processed/clean_aqi_stations/` | 15-min, 2024-01-01 → 2025-12-31. **Timestamps are IST although labelled `+0000`** (§4.1). |
| IMD temperature | `data/raw/chennai_daily_temperature_1951_2024.csv` → `data/processed/chennai_temperature_daily_1951_2024.csv` | daily, ends 2024-06-22 |
| IMD rainfall | `data/raw/chennai_daily_rainfall_1991_2023.csv` → `..._climatology_1993_2023.csv` | daily, ends 2023-12-26 |
| Open-Meteo hourly | `data/external/open_meteo_chennai_hourly_2025_2026.csv` | two-block CSV, 2025-01-01 00:00 → 2026-10-07 23:00 UTC (§2) |
| Master table / splits | `data/processed/{master_table,train,validate,test}.csv` | hourly 2024-01-01 → 2025-12-31; train → 2025-05-26, validate 2025-05-27 → 09-13, test 2025-09-14 → 12-31. **Unchanged.** |
| AQI HMMs | `models/hmm/hmm_aqi_zone_{2,4,5,10,12,13,14}.joblib` | K = 6, hourly CPCB AQI |
| Rainfall HMMs | `models/hmm/hmm_rainfall_zone_{4,5,10,14}.joblib` | K = 3–4, hourly gauge rainfall |
| Temperature HMMs | `models/hmm/hmm_temperature_zone_{1..15}.joblib` | **15 identical copies**, K = 4, fit on 171 daily **mean** temperatures (IMD Tmean, Jan–Jun 2024) |
| MRFs | `models/hmm_mrf/mrf_{aqi,rainfall,temperature}.joblib` | Gaussian MRF on 29 shared-boundary edges; **β = 1e-4 (AQI), 0 (rainfall), 0 (temperature)** |
| Previous live output / dashboard | `data/live/current_predictions.json`, `dashboard/app.py` | backed up to `backups/2026-10-09_pre_final_audit/` |
| Previous audits | `reports/data_audit.md`, `reports/historical_data_audit.md`, … | kept. Two claims there are wrong: AQI MRF β is 1e-4 (not 0.65), and the temperature HMM is a daily-*mean* model (not Tmax). |
| OpenAQ audit script | — | **none existed**; written as `src/audit/openaq_audit.py` |

No models were retrained. No data files were overwritten.

## 2. Open-Meteo audit (Phase B) — `reports/open_meteo_audit/summary.md`

- Format detected from content: CSV with two blank-line-separated blocks (location metadata, hourly data). Not JSON.
- 232,200 rows = 15 locations × 15,480 hours. 0 duplicate timestamps, 0 missing hours, 0 missing values. UTC (`utc_offset_seconds = 0`).
- Variables/units: `temperature_2m` °C, `relative_humidity_2m` %, `precipitation` mm (sum over the preceding hour), `wind_speed_10m` km/h, `wind_direction_10m` °. All values within plausible ranges.
- **Does not reach the audit date**: the last record is 2026-10-07 23:00 UTC; 2026-10-08 is absent. The live collector fetches newer hours from the same API.
- The file does not store requested coordinates. Location *i* = GeoJSON feature *i* (mean 3.5 km from that zone's centroid vs 9.5 km under zone-number order).
- **Only 8 distinct grid cells for 15 zones**: {4, 5, 6, 8}, {2, 3}, {10, 11}, {12, 14}, {9, 13} share cells (identical or near-identical series). Zones 1 and 15 resolve to cells outside the GCC polygons.

## 3. OpenAQ audit (Phase C) — `reports/openaq_audit/summary.md`

- API checked against live responses: `/v3/locations?bbox`, `/v3/locations/{id}/sensors`, `/v3/sensors/{id}/hours` (`datetime_from`/`datetime_to`, `limit` ≤ 1000, `page`), `X-API-Key` header, 60 req/min (`X-Ratelimit-*` headers). The key was never printed or logged.
- 13 locations in the GCC bounding box, 146 sensors, of which 95 overlap 2025-01-01 → 2026-10-08 (IST). **1,985 requests, 0 failed, 0 sensor-months missing.** All 95 returned data; 0 duplicate hours, 0 sentinel values.
- 8 of the 9 local stations are on OpenAQ (Gandhi Nagar / Ennore is not). New-generation sensors start 2025-02-19 01:00 IST and run to **2026-10-07 19:00 IST**, except Perungudi (ends 2026-09-29 00:00) and Alandur (ends 2025-12-02 16:00). Active-period coverage is 72–90 %, full-window coverage 66–83 % (Alandur 37 %).
- No station publishes **NH3** on OpenAQ, so live AQI uses at most 6 of the 7 CPCB pollutants. Arumbakkam has no CO/O3 (as in the local file).
- AirNow "Chennai" (PM2.5 only, 51 % coverage, no data Jul 2025–Apr 2026) is not usable for CPCB AQI (needs ≥3 pollutants). It is reported but not used.
- Sensor metadata coverage is unreliable (e.g. `percentCoverage` = 1,257,925 %). Coverage was computed from downloaded hours.

## 4. Combined audit (Phase D)

### 4.1 Timezones
Local station timestamps are **IST mislabelled as UTC**. Aligning local files with OpenAQ (true UTC) across 44 sensors gives median correlation 0.977 when local times are read as IST, and 0.468 when read as UTC. This does not affect the trained HMMs (they use only hourly ordering), but any join with UTC sources must shift by 5 h 30 m. Open-Meteo is UTC on :00. OpenAQ hours start at :30 UTC (IST hour boundaries).

### 4.2 Units — `reports/unit_validation.csv`
OpenAQ labels the current CPCB gas sensors (CO, NO2, SO2) as **ppb**, but the values equal the local CPCB-native values (CO mg/m³, others µg/m³): median ratio 0.993–1.005, correlation 0.86–0.999 for **44 sensors → verdict `native_cpcb_units`**. Applying a ppb conversion would have inflated NO2 by 1.88× and CO by 1,145×. No conversion is applied. Two SO2 sensors (Kodungaiyur, Alandur) have the right scale but correlation 0.50, so they are **excluded**. 34 legacy µg/m³ sensors have no data in the window. The live pipeline uses only `native_cpcb_units` sensors.

### 4.3 Implausible values
Local files: PM values cap at 999/1000 (instrument ceiling). OpenAQ: 7 hours of Manali PM2.5 exceed the existing physical bound in `src/live/schemas.py` (max 5,870 µg/m³). The live pipeline drops them. Negative pollutant values are invalid. An hour is valid only with ≥2 of 4 sub-hourly readings (same rule as training).

### 4.4 Station → zone mapping — `reports/coverage/station_mapping_reconciliation.csv`
| station | mapped zone | local coords | OpenAQ coords | note |
|---|---|---|---|---|
| Manali, Manali Village | 2 | zone 2 | zone 2 | consistent |
| Kodungaiyur | 4 | zone 4 | zone 4 | consistent |
| Royapuram | 5 | zone 5 | zone 5 | consistent |
| Velachery Res. Area | 13 | zone 13 | zone 13 | coords 3.7 km apart, same zone |
| Arumbakkam | 10 | zone 10 | **zone 8** (0.23 km from zone 10) | boundary-ambiguous; kept as 10 |
| Perungudi | 14 | zone 14 | **zone 15** (0.04 km from zone 14) | boundary-ambiguous; kept as 14 |
| Alandur Bus Depot | 12 | zone 12 | **10 km outside GCC** | OpenAQ coordinate error; matched by name |
| Gandhi Nagar / Ennore | 1 | zone 1 | not on OpenAQ | no live data |

The existing mapping is kept because the zone HMMs were trained on these station assignments.

### 4.5 Weather: reanalysis vs measurement
- **Temperature** (`temperature_open_meteo_vs_station.csv`): versus OpenAQ station thermometers (2025–26), daily-mean MAE is 0.6–1.1 °C at Manali, Velachery, Kodungaiyur, Arumbakkam, Perungudi and Alandur. Royapuram shows a constant +3.75 °C offset and Manali Village a correlation of 0.14, which look like station-sensor problems.
- **Rainfall** (`rain_open_meteo_vs_gauge_2025.csv`): the CAAQMS gauges are not a reliable reference. **Perungudi's gauge is drifting** (non-zero in 17 % of 15-min intervals, 12,578 mm over 319 days of 2025; Chennai's annual normal is ~1,400 mm). Kodungaiyur and Arumbakkam recorded 0.7 and 0.0 mm on 2025-12-02, when reanalysis shows 152 and 111 mm. Full-year daily correlation between gauge and reanalysis is −0.01 to 0.24.

### 4.6 Existing models on the inputs available live
- **AQI HMMs** (2026, truly out of sample, OpenAQ-derived AQI; `combined_audit_results.json`). OpenAQ AQI reproduces the master-table AQI for 2025 (MAE 1.0–8.5, r 0.79–0.99). Frozen HMM 1 h MAE: 5.8 (Royapuram), 9.9 (Kodungaiyur), 12.1 (Perungudi), 13.9 (Velachery), 17.8 (Manali), 18.1 (Arumbakkam). 24 h MAE: 9.5–36.8.
- **Rainfall HMMs** (existing test partition, gauge truth; `rain_hmm_test_eval_by_input.csv`). Driving them with Open-Meteo raises 1 h MAE from 0.10–0.25 mm to 0.79–1.16 mm (zones 4, 5, 10), with correlation 0.05–0.23. 24 h forecasts are identical for either input (they revert to the stationary mean). Zone 14's model was trained on the drifting gauge. **Input requirement not satisfied → rainfall forecasts disabled.**
- **Temperature HMM** (next-day daily mean vs Open-Meteo daily means, 644 days): MAE 0.91–1.07 °C. The typical day-to-day change is 0.49–0.57 °C, so the model has **low skill**. It is kept because it is the trained model for this variable/horizon, and it is labelled as such.

### 4.7 Defects found in the previous live pipeline (v2, WeatherAPI)
1. WeatherAPI air-quality concentrations (model output, instantaneous) were fed in and displayed as "CPCB AQI" observations.
2. An instantaneous temperature was fed into a daily-mean HMM, and the output was labelled `next_day_tmax_pred_c`.
3. `evaluate_aqi_alert(NaN)` returned probability 0 → **NORMAL**, so missing forecasts showed as low risk.
4. With no active zone, the MRF returns mean 0 with very large variance → P(AQI > 200) ≈ 0.5 → spurious **WATCH**.
5. The dashboard mapped unknown statuses to "Low" (`get_risk_level`), started zone colour at "Low", and showed `"0.0"` mm rainfall when the value was missing.
6. HMMs were filtered on a single observation instead of the available history.

All fixed in v3 (§7); the old code is in `backups/`.

## 5. Coverage by zone — `reports/coverage/zone_variable_coverage.csv`

| zone | AQ measured 2024-25 | AQ live (OpenAQ) | AQ model | weather | rain forecast | temp forecast |
|---|---|---|---|---|---|---|
| 1 Thiruvottiyur | ~90 % missing | none | MRF estimate only | reanalysis (own cell, outside GCC) | — | next-day mean |
| 2 Manali | yes (+ Manali Village) | to 2026-10-07 | HMM + MRF | reanalysis (cell shared with 3) | — | next-day mean |
| 3 Madhavaram | none | none | MRF estimate only | reanalysis (= zone 2) | — | next-day mean |
| 4 Tondiarpet | yes | to 2026-10-07 | HMM + MRF | reanalysis (cell shared with 5, 6, 8) | disabled | next-day mean |
| 5 Royapuram | yes | to 2026-10-07 | HMM + MRF | reanalysis (= 4) | disabled | next-day mean |
| 6 Thiru-Vi-Ka Nagar | none | none | MRF estimate only | reanalysis (= 4) | — | next-day mean |
| 7 Ambattur | none | none | MRF estimate only | reanalysis (own cell) | — | next-day mean |
| 8 Anna Nagar | none | none | MRF estimate only | reanalysis (= 4) | — | next-day mean |
| 9 Teynampet | none | none | MRF estimate only | reanalysis (≈ 13) | — | next-day mean |
| 10 Kodambakkam | yes (no CO/O3) | to 2026-10-07 | HMM + MRF | reanalysis (= 11) | disabled | next-day mean |
| 11 Valasaravakkam | none | none | MRF estimate only | reanalysis (= 10) | — | next-day mean |
| 12 Alandur | yes to 2025-12-02 | **stale** (ends 2025-12-02) | MRF estimate only while stale | reanalysis (= 14) | — | next-day mean |
| 13 Adyar | yes | to 2026-10-07 | HMM + MRF | reanalysis (≈ 9) | — | next-day mean |
| 14 Perungudi | yes | **stale** (ends 2026-09-29) | MRF estimate only while stale | reanalysis (= 12) | disabled | next-day mean |
| 15 Sholinganallur | none | none | estimate only if zone 14 is live | reanalysis (own cell, outside GCC) | — | next-day mean |

"MRF estimate" is produced only when at least one adjacent zone has a live measured input; otherwise the zone is unavailable.

## 6. Decision table (Phase E)

| variable / horizon | live source (provenance) | zones | existing model | known limitations | code |
|---|---|---|---|---|---|
| AQI now | OpenAQ CAAQMS, CPCB NAQI method (**measured**) | 2, 4, 5, 10, 13 live; 12, 14 stale | — (computed with the same function as training) | ~1–2 day OpenAQ lag; no NH3; 2 SO2 sensors excluded | `live_inputs.build_aqi_inputs`, `cpcb_aqi.aqi_from_hourly_concentrations` |
| AQI 1/3/6/24 h | as above | HMM + MRF where measured input ≤ 6 h old; MRF estimate for neighbours | `hmm_aqi_zone_*`, `mrf_aqi` (β = 1e-4) | target times are relative to the last measured hour; 2026 1 h MAE 6–18 | `live_forecaster.run_aqi` |
| AQI alert | P(AQI > 200) from the MRF posterior | as above | `alert_engine.evaluate_aqi_alert` | 20 %/60 % cut-offs are UrbanPulse decision rules, not CPCB | unchanged except the NaN guard |
| Rainfall now / last 24 h | Open-Meteo (**reanalysis**) | all 15 (8 cells) | — | not a gauge; smoother, lower peaks | `live_inputs.weather_current` |
| Rainfall 1/3/6/24 h, heavy-rain alert | — | **none (disabled)** | `hmm_rainfall_zone_*` exists but input not valid | §4.5–4.6 | `RAINFALL_FORECAST_ENABLED = False` |
| Temperature now | Open-Meteo (**reanalysis**) | all 15 (8 cells) | — | Royapuram/Manali Village station disagreement | `weather_current` |
| Temperature 1/3/6 h | — | none | no sub-daily model exists | not fabricated | — |
| Temperature next day | Open-Meteo complete-day IST means (**reanalysis**) | all 15 | `hmm_temperature_zone_*` (daily mean), `mrf_temperature` (β = 0) | daily **mean**, not Tmax; low skill (MAE ~1 °C) | `live_forecaster.run_temperature` |
| Heatwave alert | — | none | — | IMD criterion needs Tmax + departure from normal; model gives mean | status `NOT_EVALUATED_MODEL_IS_DAILY_MEAN` |

No model needed retraining to support this table. Retraining would only be justified to add a rainfall forecast (e.g. HMMs fit on reanalysis precipitation) or a Tmax model, and both are outside this scope.

## 7. Implementation summary (Phase F)

```
collect      src/collectors/{openaq_client,open_meteo,live_collect}.py  -> data/live/raw/
preprocess   src/preprocessing/live_inputs.py (+ cpcb_aqi.aqi_from_hourly_concentrations)
infer        src/forecasting/live_forecaster.py   (frozen HMM + MRF; no fitting)
orchestrate  src/live/live_pipeline.py            -> data/live/current_predictions.json (atomic, strict JSON)
present      dashboard/app.py + data_loader.py    (reads JSON only)
```

- Failure handling: a failed source keeps its previous file. `collection_status.json` records the error and the last success. Data is flagged stale after 3 h (OpenAQ) / 6 h (Open-Meteo). Archive hours after "now" (forecast-filled) are dropped. If no source has usable data, the previous output is left untouched.
- Every zone/variable/horizon carries `availability` (`MODEL_FORECAST`, `MODEL_ESTIMATE`, `UNAVAILABLE`), a `basis`, and an explicit status. Unavailable is never mapped to a value or to NORMAL.
- Dashboard: map coloured by evaluated alerts only (grey = not evaluated, dashed outline = MRF estimate); zone panel separates measured AQ from reanalysis weather; per-source data age and stale flags; last successful and next scheduled update; Alert Center; new **Data & Coverage** page.
- Credentials stay in `.env`, now listed in `.gitignore`.
- Tests: `tests/test_final_pipeline.py` (data validation, zone mapping, missing data, API failures, output validation).
