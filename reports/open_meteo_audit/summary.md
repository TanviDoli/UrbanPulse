# Open-Meteo Local File Audit

- File: `data/external/open_meteo_chennai_hourly_2025_2026.csv`
- Detected format (from content): **csv** — two blank-line-separated CSV blocks (location metadata block + hourly data block). Not JSON; not a single-table CSV.
- Rows: 232,200 hourly rows, 15 locations, 5 variables.
- Timezone: `timezone=GMT`, `utc_offset_seconds=0` for all locations → timestamps are UTC.
- Period: 2025-01-01 00:00 → 2026-10-07 23:00 UTC (2026-10-08 04:30 IST); 15,480 expected hourly steps per location.
- Reaches intended audit date 2026-10-08 (full UTC day)? **no** — the last record is 2026-10-07 23:00 UTC; 2026-10-08 is not in the file.
- Provenance: Open-Meteo Historical Weather API = reanalysis/model-derived (ERA5 family / ECMWF IFS). These are **not measured station observations**.

## Units

| variable | in file | expected | status |
|---|---|---|---|
| temperature_2m | '°C' | °C | OK |
| relative_humidity_2m | '%' | % | OK |
| precipitation | 'mm' | mm | OK |
| wind_speed_10m | 'km/h' | km/h | OK |
| wind_direction_10m | '°' | ° | OK |

The degree sign is stored as UTF-8 `°` but some readers decode it as `�`; values are unaffected.

## Completeness

- Duplicate timestamps (any location): 0
- Missing hourly timestamps (any location): 0
- Missing values per variable (all locations): temperature_2m_c=0, relative_humidity_2m_pct=0, precipitation_mm=0, wind_speed_10m_kmh=0, wind_direction_10m_deg=0

## Location → zone mapping

The file does not record the requested coordinates; returned coordinates are model grid cells. Location *i* matches GeoJSON feature *i* (mean grid-to-centroid distance 3.46 km) and not zone number *i+1* (mean 9.48 km).

| location | zone | grid lat | grid lon | elev (m) | km to zone centroid | grid cell inside zone |
|---|---|---|---|---|---|---|
| 0 | 1 Thiruvottiyur | 13.2513 | 80.3102 | 4 | 8.14 | outside GCC |
| 1 | 2 Manali | 13.1810 | 80.2370 | 3 | 3.2 | same zone |
| 14 | 3 Madhavaram | 13.1810 | 80.2370 | 8 | 4.09 | zone 2 |
| 7 | 4 Tondiarpet | 13.1107 | 80.2459 | 5 | 3.15 | zone 6 |
| 8 | 5 Royapuram | 13.1107 | 80.2459 | 11 | 4.38 | zone 6 |
| 3 | 6 Thiru-Vi-Ka Nagar | 13.1107 | 80.2459 | 12 | 0.82 | same zone |
| 2 | 7 Ambattur | 13.1107 | 80.1639 | 16 | 0.83 | same zone |
| 10 | 8 Anna Nagar | 13.1107 | 80.2459 | 13 | 3.67 | zone 6 |
| 9 | 9 Teynampet | 13.0404 | 80.2548 | 11 | 0.84 | same zone |
| 4 | 10 Kodambakkam | 13.0404 | 80.1729 | 17 | 4.06 | zone 11 |
| 6 | 11 Valasaravakkam | 13.0404 | 80.1729 | 14 | 0.99 | same zone |
| 5 | 12 Alandur | 12.9701 | 80.1818 | 11 | 3.41 | same zone |
| 11 | 13 Adyar | 13.0404 | 80.2548 | 14 | 4.75 | zone 9 |
| 13 | 14 Perungudi | 12.9701 | 80.1818 | 3 | 4.64 | zone 12 |
| 12 | 15 Sholinganallur | 12.8998 | 80.1907 | 6 | 4.92 | outside GCC |

## Shared grid cells — 8 distinct cells for 15 zones

| grid cell | zones | series |
|---|---|---|
| 13.1107,80.2459 | 4, 5, 6, 8 | identical |
| 12.9701,80.1818 | 12, 14 | identical |
| 13.0404,80.2548 | 9, 13 | max abs diff temperature_2m_c=0.10, relative_humidity_2m_pct=1.00 |
| 13.0404,80.1729 | 10, 11 | identical |
| 13.181,80.237 | 2, 3 | identical |
| 12.8998,80.1907 | 15 | single zone |
| 13.1107,80.1639 | 7 | single zone |
| 13.2513,80.3102 | 1 | single zone |

Implication: zones sharing a cell are **not independent weather observations**. Spatial differences between them are zero (or elevation-adjusted only).

## Plausibility

| variable | min | p01 | median | p99 | max | below | above |
|---|---|---|---|---|---|---|---|
| temperature_2m_c | 17.8 | 21.7 | 28.6 | 37.7 | 41.8 | 0 | 0 |
| relative_humidity_2m_pct | 15.0 | 35.0 | 73.0 | 98.0 | 100.0 | 0 | 0 |
| precipitation_mm | 0.0 | 0.0 | 0.0 | 3.4 | 34.0 | 0 | 0 |
| wind_speed_10m_kmh | 0.0 | 1.5 | 10.6 | 22.6 | 41.6 | 0 | 0 |
| wind_direction_10m_deg | 0.0 | 8.0 | 178.0 | 352.0 | 360.0 | 0 | 0 |

Hours with precipitation > 0: 16.2% of all location-hours.

## Inconsistencies with the existing weather pipeline

- The existing live pipeline (`src/live/live_pipeline.py`) reads WeatherAPI *current* conditions, not Open-Meteo.
- Open-Meteo wind speed is km/h; CAAQMS station `WS` is m/s.
- Open-Meteo `precipitation` is the sum over the *preceding* hour (timestamp = hour end); station `RF` is per 15-minute interval and OpenAQ hours are IST-aligned (:30 UTC).
- The temperature HMM was trained on IMD *daily mean* temperature (Nungambakkam/Meenambakkam record, 1951–2024-06-22); Open-Meteo hourly temperature must be aggregated to complete IST-day means before use.
- The local file starts 2025-01-01, so it does not overlap the IMD daily temperature record (ends 2024-06-22) or the IMD rainfall record (ends 2023-12-26); it overlaps the CAAQMS station files for calendar 2025.
