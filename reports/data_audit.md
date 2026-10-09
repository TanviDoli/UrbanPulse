# UrbanPulse: Comprehensive Raw Data Audit Report
**Project:** Spatio-Temporal Early Warning System for Chennai — AQI, Rainfall & Temperature Forecasting using HMM + MRF  
**Date of Audit:** October 3, 2026  
**Author:** Lead ML / Data-Engineering Developer  
**Status:** Complete — Model training halted pending user review and confirmation.

---

## Executive Summary

A comprehensive, zero-assumption audit was executed across all raw data assets in `data/raw/`, comprising:
1. 1 GeoJSON file (`chennai_zones_2025.geojson`) defining Greater Chennai Corporation (GCC) administrative boundaries.
2. 9 Continuous Ambient Air Quality Monitoring Station (CAAQMS) CSV files in `data/raw/AQI/`.
3. 1 IMD daily rainfall historical dataset (`chennai_daily_rainfall_1991_2023.csv`).
4. 1 IMD daily temperature historical dataset (`chennai_daily_temperature_1951_2024.csv`).

### Critical Discoveries & Scientific Realities
1. **Severe Temporal Misalignment Between Modalities:**
   - **AQI station files:** Span **2024-01-01 00:00:00 to 2025-12-31 23:45:00** at 15-minute intervals (2 full calendar years).
   - **IMD Daily Rainfall:** Spans **1993-03-03 to 2023-12-26**. It **terminates 6 days before the AQI period begins** (zero direct temporal overlap).
   - **IMD Daily Temperature:** Spans **1951-01-01 to 2024-06-22**. It overlaps with the AQI period for only **174 calendar days** (January 1, 2024 to June 22, 2024).
2. **Weather Variables Inside AQI Station Files:**
   - **Ambient Temperature (`AT`):** 100% missing in 7 of the 9 stations (Alandur, Arumbakkam, Kodungaiyur, Manali, Perungudi, Royapuram, Velachery). Only Gandhi Nagar (11.0% non-null) and Manali Village (66.1% non-null) record temperature.
   - **Rainfall (`RF`):** 100% missing in 4 stations (Alandur, Gandhi Nagar, Manali, Velachery). Present in Arumbakkam, Kodungaiyur, Perungudi, Royapuram, and Manali Village. However, Manali Village exhibits severe sensor drift/failure, recording positive rainfall on 66.4% of all 15-minute intervals totaling 95,079 mm (unphysical).
3. **AQI Concatenation Artifact in Every CSV:**
   - Every single AQI station CSV contains a corrupted boundary row at the 2024/2025 transition (e.g., line 35137) where the column contains the string literal `"2025"` with all other fields set to `NaN`.
4. **Spatial Coverage Disparity:**
   - Chennai has **15 administrative zones**, but only **9 CAAQMS stations**.
   - Zone 2 (Manali) contains 2 stations (`manali` and `manali-village`).
   - 7 administrative zones have no direct air quality monitoring stations.
5. **Operational Resolution Feasibility:**
   - Hourly operational resolution is defensible for AQI (aggregating four 15-min intervals into 1-hour means and calculating rolling CPCB sub-indices).
   - For daily weather variables, hourly operational modeling requires an explicit, provenance-preserving representation (e.g., day-level step functions with boolean observation flags `rainfall_observed` and `temperature_observed`), strictly adhering to the prohibition against data fabrication.

---

## 1. File-by-File Data Audit

### 1.1 GCC Zone Polygons: `data/raw/chennai_zones_2025.geojson`

| Attribute | Value |
|---|---|
| **File Format** | GeoJSON (`FeatureCollection`) |
| **Coordinate Reference System** | EPSG:4326 (WGS84 Longitude, Latitude) |
| **Number of Features** | 15 (Exactly corresponding to GCC Zones) |
| **Geometry Types** | All 15 are `Polygon` |
| **Geometry Validity** | 100% valid (`is_valid = True` for all features, self-intersections = 0) |
| **Property Schema** | `{'name_of_th': string}` |
| **Bounding Box** | Longitude: [80.134, 80.331], Latitude: [12.852, 13.235] |
| **Topology** | Fully connected planar graph with 29 shared-boundary adjacency edges |

#### Zone Geometry Inventory

| Index | Zone Label (`name_of_th`) | Official GCC Zone # | Centroid (Lat, Lon) | Area ($deg^2$) | Vertex Count |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | THIRUVOTTRIYUR | Zone 1 | (13.1690 N, 80.2980 E) | 0.002118 | 262 |
| 2 | MANALI | Zone 2 | (13.1808 N, 80.2463 E) | 0.003526 | 594 |
| 3 | AMBATTUR | Zone 7 | (13.1006 N, 80.1856 E) | 0.003211 | 1358 |
| 4 | THIRU-VIKA-NAGAR | Zone 6 | (13.1084 N, 80.2391 E) | 0.001427 | 2736 |
| 5 | KODAMBAKKAM | Zone 10 | (13.0478 N, 80.2146 E) | 0.001865 | 2146 |
| 6 | ALANDUR | Zone 12 | (12.9968 N, 80.1905 E) | 0.001700 | 1339 |
| 7 | VALASARVAKKAM | Zone 11 | (13.0548 N, 80.1712 E) | 0.001894 | 1157 |
| 8 | TONDIARPET | Zone 4 | (13.1215 N, 80.2662 E) | 0.001739 | 1647 |
| 9 | ROYAPURAM | Zone 5 | (13.0918 N, 80.2719 E) | 0.001819 | 1979 |
| 10 | TEYNAMPET | Zone 9 | (13.0509 N, 80.2474 E) | 0.002101 | 1945 |
| 11 | ANNA NAGAR | Zone 8 | (13.0871 N, 80.2207 E) | 0.002109 | 2226 |
| 12 | ADYAR | Zone 13 | (13.0066 N, 80.2353 E) | 0.003365 | 1611 |
| 13 | SHOZHANGANALLUR | Zone 15 | (12.9270 N, 80.2346 E) | 0.003630 | 371 |
| 14 | PERUNKUDI | Zone 14 | (12.9607 N, 80.2217 E) | 0.002901 | 1133 |
| 15 | MADHAVARAM | Zone 3 | (13.1416 N, 80.2186 E) | 0.002625 | 1474 |

---

### 1.2 Daily Rainfall: `data/raw/chennai_daily_rainfall_1991_2023.csv`

| Metric | Value | Notes |
|---|---|---|
| **Rows** | 21,416 | Raw rows |
| **Columns** | 4 | `['District', 'Station', 'Rainfall', 'Date']` |
| **Data Types** | `District`: object, `Station`: object, `Rainfall`: float64, `Date`: object (DD-MM-YYYY) |
| **Temporal Range** | `1993-03-03` to `2023-12-26` | 30.8 years (no 2024 or 2025 data) |
| **Frequency** | Event-based daily | Sparse: only wet days ($Rainfall \ge 0.01$ mm) are recorded. Days with 0 mm are omitted. |
| **Missing Values** | 0 nulls across all columns |
| **Exact Duplicates** | 1,361 rows (6.35%) | Complete duplicate rows requiring deduplication. |
| **Duplicate Station-Dates** | 2,844 rows | Multiple conflicting or repeated measurements on the same date for a station. |
| **Station Count** | 62 distinct stations | Mix of legacy IMD gauges (`Chennai nungambakkam`, `Anna university`) and GCC zone tags (`Zone 08 Anna Nagar`). |
| **Rainfall Statistics** | Min: 0.01 mm, Mean: 18.28 mm, Median: 7.40 mm, 75%: 22.50 mm, Max: 348.30 mm | No negative values. Highly skewed distribution. |

---

### 1.3 Daily Temperature: `data/raw/chennai_daily_temperature_1951_2024.csv`

| Metric | Value | Notes |
|---|---|---|
| **Rows** | 26,837 | Raw rows |
| **Columns** | 4 | `['Date', 'Rain', 'Temp Max', 'Temp Min']` |
| **Data Types (Raw)** | All 4 columns read as `object` | Due to text sentinel tokens |
| **Temporal Range** | `1951-01-01` to `2024-06-22` | Exactly 26,837 calendar days (0 missing days in the daily sequence) |
| **Frequency** | Daily regular | 1 record per calendar day |
| **Exact Duplicates** | 0 |
| **Sentinels / Non-numerics** | `Rain`: 83 occurrences of `'Tr'` (trace rain), 1 occurrence of `'----'`. `Temp Max`: 9 occurrences of `'-----'`. `Temp Min`: 28 occurrences of `'-----'`. |
| **Missing Values (Nulls)** | `Date`: 0, `Rain`: 25,568 nulls (95.27%), `Temp Max`: 1 null, `Temp Min`: 1 null |
| **Numeric Statistics** | **Temp Max:** Min: 23.55 °C, Mean: 33.68 °C, Max: 44.96 °C.<br>**Temp Min:** Min: 15.30 °C, Mean: 23.97 °C, Max: 92.00 °C (Anomaly). |
| **Anomalies / Outliers** | Row 26632 (`01-12-2023`): `Temp Min = 92.0 °C`. This is a known data entry typo that must be sanitized (imputed or replaced with valid seasonal minimum). |

---

### 1.4 CAAQMS Air Quality Files (`data/raw/AQI/*.csv`)

All 9 files share an identical 29-column schema representing continuous ambient monitoring data:
- **Identifiers:** `Station ID`, `State`, `City`, `Station Name`, `Timestamp`
- **Criteria Pollutants:** `PM2.5 (µg/m³)`, `PM10 (µg/m³)`, `NO (µg/m³)`, `NO2 (µg/m³)`, `NOx (ppb)`, `NH3 (µg/m³)`, `SO2 (µg/m³)`, `CO (mg/m³)`, `Ozone (µg/m³)`
- **VOCs:** `Benzene`, `Toluene`, `Xylene`, `O Xylene`, `Eth-Benzene`, `MP-Xylene`
- **Meteorological Sensors:** `AT (°C)` (Ambient Temp), `RH (%)` (Relative Humidity), `WS (m/s)` (Wind Speed), `WD (deg)` (Wind Direction), `RF (mm)` (Rainfall), `TOT-RF (mm)`, `SR (W/mt2)` (Solar Rad), `BP (mmHg)` (Barometric Pressure), `VWS (m/s)` (Vertical Wind Speed)

*Note on encoding:* In the raw CSV headers, non-ASCII characters (`µ`, `°`) appear corrupted as `\ufffd` depending on the parser encoding. Standard normalization strips these units cleanly.

#### Complete Station-by-Station Audit Table

| Station File | Station Name & ID | Total Rows | Valid Rows | Missing Grid Steps | Date Range (UTC) | Top Pollutant Completeness | Weather Sensors Availability |
|:---|:---|:---:|:---:|:---:|:---|:---|:---|
| **alandur-2024-25.csv** | Alandur Bus Depot (`site_293`) | 62,790 | 62,789 | 7,387 | 2024-01-01 00:00 to 2025-12-02 16:30 | PM2.5: 92.1%, PM10: 92.5%, NO2: 90.0%, SO2: 96.3%, CO: 99.5%, O3: 98.3%, NH3: 82.9% | AT: 0.0%, RF: 0.0%, RH: 46.7%, WS: 99.5% |
| **arumbakkam-2024-25.csv** | Arumbakkam (`site_5361`) | 70,177 | 70,176 | 0 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 76.1%, PM10: 79.6%, NO2: 61.7%, SO2: 81.0%, **CO: 0.0%**, **O3: 0.0%**, NH3: 61.2% | AT: 0.0%, RF: 82.8%, RH: 81.8%, WS: 83.0% |
| **gandhi-nagar-2024-25.csv** | Gandhi Nagar_Ennore (`site_5618`) | 70,177 | 70,176 | 0 | 2024-01-01 00:00 to 2025-12-31 23:45 | **PM2.5: 9.9%**, **PM10: 8.5%**, **NO2: 10.3%**, SO2: 12.6%, CO: 3.8%, O3: 8.6%, NH3: 8.7% | **AT: 11.0%**, RF: 0.0%, RH: 0.03%, WS: 11.0% |
| **kodungaiyur-2024-25.csv** | Kodungaiyur (`site_5365`) | 70,177 | 70,176 | 0 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 92.5%, PM10: 92.1%, NO2: 94.5%, SO2: 88.7%, CO: 88.7%, O3: 75.1%, NH3: 95.1% | AT: 0.0%, RF: 96.9%, RH: 96.3%, WS: 96.7% |
| **manali-2024-25.csv** | Manali (`site_306`) | 63,740 | 63,739 | 6,437 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 82.4%, PM10: 87.5%, NO2: 88.1%, SO2: 86.7%, CO: 98.0%, O3: 98.3%, NH3: 89.8% | AT: 0.0%, RF: 0.0%, RH: 99.2%, WS: 99.2% |
| **manali-village-2024-25.csv** | Manali Village (`site_5092`) | 70,177 | 70,176 | 0 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 57.2%, PM10: 58.5%, NO2: 56.8%, SO2: 66.5%, CO: 64.5%, O3: 66.0%, NH3: 64.2% | **AT: 66.2%**, **RF: 67.5% (Corrupted)**, RH: 66.6%, WS: 66.6% |
| **perungudi-2024-25.csv** | Perungudi (`site_5363`) | 70,177 | 70,176 | 0 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 92.1%, PM10: 95.6%, NO2: 97.5%, SO2: 96.7%, CO: 97.3%, O3: 96.1%, NH3: 97.3% | AT: 0.0%, RF: 93.5%, RH: 93.4%, WS: 96.3% |
| **royapuram-2024-25.csv** | Royapuram (`site_5364`) | 70,177 | 70,176 | 0 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 82.9%, PM10: 85.4%, NO2: 83.1%, SO2: 71.6%, CO: 75.7%, O3: 80.5%, NH3: 83.2% | AT: 0.0%, RF: 88.1%, RH: 88.1%, WS: 88.1% |
| **velachery-2024-25.csv** | Velachery Res. Area (`site_288`) | 66,245 | 66,244 | 3,932 | 2024-01-01 00:00 to 2025-12-31 23:45 | PM2.5: 96.4%, PM10: 95.7%, NO2: 96.7%, SO2: 98.6%, CO: 99.2%, O3: 98.6%, NH3: 95.1% | AT: 0.0%, RF: 0.0%, RH: 97.4%, WS: 99.9% |

---

## 2. Pollutant Concentration Ranges & CPCB Sub-Index Viability

Under the official CPCB methodology:
1. To calculate an official AQI value, at least **3 pollutants** must be monitored, of which **at least one MUST be PM2.5 or PM10**.
2. **PM2.5, PM10, NO2, SO2, NH3** require 24-hour averaging (minimum 16 hours of valid data).
3. **CO, Ozone** require 8-hour averaging (minimum 6 hours of valid data).

### Minimum Criteria Assessment by Station
- **Alandur, Kodungaiyur, Manali, Perungudi, Royapuram, Velachery:** Possess **7 out of 7** CPCB criteria pollutants with high data completeness (>80–98%). 100% compliant with CPCB calculation standards.
- **Arumbakkam:** CO and Ozone sensors are completely unequipped (0% data). However, PM2.5, PM10, NO2, SO2, and NH3 are fully present (5 criteria pollutants including both PM fractions). Thus, Arumbakkam **fully satisfies the CPCB criteria** ($\ge 3$ pollutants with PM).
- **Manali Village:** Possess all 7 pollutants with moderate completeness (~57–66%). Fully satisfies the CPCB criteria.
- **Gandhi Nagar:** Possesses all 7 pollutants but suffers from ~90% missingness across the entire 2-year period. CPCB daily sub-indices can only be computed during active monitoring windows.

### Observed Pollutant Value Ranges & Truncation
- **Negative Values:** Exactly **0** negative concentration values detected across all 9 stations for all pollutants.
- **Sensor Saturation/Capping:**
  - PM2.5: Observed max values range from 316.0 to 1,000.0 µg/m³. The 99th percentile across stations is 75–225 µg/m³.
  - PM10: Observed max values cap at 1,000.0 µg/m³. The 99th percentile is 163–365 µg/m³.
  - CO: Capped at 10.0 mg/m³ (99th percentile: 1.5–4.5 mg/m³).
  - Ozone: Capped at 500.0 µg/m³ (99th percentile: 32–207 µg/m³).

---

## 3. Station-to-Zone Spatial Mapping

Using geometric point-in-polygon containment against `chennai_zones_2025.geojson`, the 9 CAAQMS stations map precisely into GCC zones:

| Station Name | Station ID | Station Coordinates | Mapped Zone ID | Mapped Zone Label | Mapping Method | Confidence | Notes |
|---|---|:---:|:---:|:---|:---|:---:|:---|
| **Alandur Bus Depot** | `site_293` | (12.9972 N, 80.2005 E) | Zone 12 | ALANDUR | Point-in-Polygon | High (1.0) | Direct inside zone |
| **Arumbakkam** | `site_5361` | (13.0640 N, 80.2078 E) | Zone 10 | KODAMBAKKAM | Point-in-Polygon | High (1.0) | Situated inside Kodambakkam boundary |
| **Gandhi Nagar_Ennore** | `site_5618` | (13.2081 N, 80.3204 E) | Zone 1 | THIRUVOTTRIYUR | Point-in-Polygon | High (1.0) | Located in Northern Ennore industrial belt |
| **Kodungaiyur** | `site_5365` | (13.1367 N, 80.2458 E) | Zone 4 | TONDIARPET | Point-in-Polygon | High (1.0) | Direct inside zone |
| **Manali** | `site_306` | (13.1656 N, 80.2588 E) | Zone 2 | MANALI | Point-in-Polygon | High (1.0) | Core industrial zone |
| **Manali Village** | `site_5092` | (13.1709 N, 80.2642 E) | Zone 2 | MANALI | Point-in-Polygon | High (1.0) | Second station in Zone 2 |
| **Perungudi** | `site_5363` | (12.9660 N, 80.2452 E) | Zone 14 | PERUNKUDI | Point-in-Polygon | High (1.0) | Direct inside zone |
| **Royapuram** | `site_5364` | (13.1118 N, 80.2941 E) | Zone 5 | ROYAPURAM | Point-in-Polygon | High (1.0) | Coastal port zone |
| **Velachery Res. Area** | `site_288` | (12.9786 N, 80.2185 E) | Zone 13 | ADYAR | Point-in-Polygon | High (1.0) | Inside Adyar zone boundary |

### Zones Without Direct Continuous Air Quality Monitoring
The following **7 administrative zones** have no direct CAAQMS station:
- **Zone 3: MADHAVARAM**
- **Zone 6: THIRU-VIKA-NAGAR**
- **Zone 7: AMBATTUR**
- **Zone 8: ANNA NAGAR**
- **Zone 9: TEYNAMPET**
- **Zone 11: VALASARVAKKAM**
- **Zone 15: SHOZHANGANALLUR**

*Scientific Note:* These unmonitored zones must be explicitly represented as unobserved (`aqi_observed = False`) in raw observation tables. Their state inference will be handled by the spatial Markov Random Field (MRF) leveraging adjacency to monitored neighbors.

---

## 4. Proposed Common Operational Resolution

### The Fundamental Dilemma
1. **Air Quality:** Available at **15-minute intervals** for **2024–2025**.
2. **IMD Rainfall:** Available at **daily event resolution** for **1993–2023** (terminating in Dec 2023).
3. **IMD Temperature:** Available at **daily resolution** for **1951–2024** (terminating on June 22, 2024).

### Operational Resolution Analysis: Is Hourly Defensible?
- **For AQI:** **Yes.** Aggregating 15-minute raw pollutant concentrations into hourly means ($H = 4 \times 15\text{ min}$) and calculating rolling CPCB sub-indices at an hourly step is standard CPCB operational practice. This supports 1h, 3h, 6h, and 24h forecast horizons.
- **For Rainfall & Temperature:**
  - If hourly modeling is adopted, **converting daily rainfall into fake hourly observations is strictly prohibited by Critical Rule #2**.
  - Similarly, pretending daily $T_{\max}$ and $T_{\min}$ are independently observed hourly temperatures violates scientific provenance.

### The Scientifically Defensible Solution
1. **Primary Operational Grid: Hourly Resolution (`freq='1h'`) over 2024-01-01 to 2025-12-31.**
   - Total hourly steps per zone: $731 \text{ days} \times 24 \text{ hours} = 17,544 \text{ timesteps}$.
   - Across 15 zones: $15 \times 17,544 = 263,160 \text{ rows}$ in `master_table.csv`.
2. **Provenance-Preserving Multimodal Representation:**
   - **AQI:** Computed hourly via official CPCB piecewise sub-index equations over rolling windows. Set `aqi_observed = True` for monitored zones during valid hours, and `False` for missing intervals or unmonitored zones.
   - **Temperature:**
     - For `2024-01-01` to `2024-06-22` (174 days), the official IMD daily temperature is available. To preserve provenance without fabrication, the daily temperature metrics ($T_{\text{mean}} = \frac{T_{\max} + T_{\min}}{2}$) represent the day's ambient background, mapped across the hours of that day with `temperature_observed = True`.
     - For dates after June 22, 2024, `temperature_observed = False` and `temperature = NaN`.
     - Alternatively, where valid station-level `AT` exists (Gandhi Nagar and Manali Village), station readings can be integrated with explicit station provenance.
   - **Rainfall:**
     - In the AQI station network, high-resolution rain gauges exist at Arumbakkam, Kodungaiyur, Perungudi, and Royapuram. These provide hourly cumulative precipitation ($RF_{\text{hourly}} = \sum_{4} RF_{15\min}$) over 2024–2025 with `rainfall_observed = True`.
     - For stations with no rain gauge or unmonitored zones, `rainfall_observed = False` and `rainfall = NaN`.
     - The historical IMD daily rainfall (1993–2023) serves for baseline climatic distribution and threshold calibration (64.5 mm/day IMD heavy rainfall threshold).

---

## 5. Critical Data Quality Issues & Pre-Modeling Remediation Plan

| # | Data Problem | Impact | Remediation Action |
|:---:|:---|:---|:---|
| 1 | **Year-Transition Artifact (`'2025'` row)** | Found at line ~35137 in all 9 AQI files; causes corrupted null timestamps and string pollution in Station ID. | Filter out rows where `Station ID == '2025'` or `Timestamp` is null before any temporal parsing. |
| 2 | **Manali Village Rainfall Gauge Corruption** | 46,582 positive rainfall readings totaling 95,079 mm; stuck gauge drift. | Mask Manali Village `RF` column to `NaN` and set `rainfall_observed = False`. |
| 3 | **December 2023 Temperature Outlier** | Row 26632 (`01-12-2023`): `Temp Min = 92.0 °C`. | Clean string sentinels (`'-----'`, `'Tr'`), replace 92.0 °C with seasonal median (22.5 °C). |
| 4 | **Rainfall Duplicates & Sparsity** | 1,361 exact duplicate rows and 2,844 duplicate station-date pairs in IMD rainfall file. | Remove exact duplicates (`df.drop_duplicates()`) and aggregate multi-readings per station-date by mean. |
| 5 | **Missing Pollutants in Specific Stations** | Arumbakkam lacks CO and Ozone; Gandhi Nagar has ~90% missingness. | Apply CPCB rules allowing AQI calculation if $\ge 3$ pollutants are present including PM. For Gandhi Nagar, retain unobserved masks. |
| 6 | **Unmonitored Zone Representation** | 7 of 15 GCC zones lack direct CAAQMS monitors. | Formulate `station_zone_mapping.csv` with explicit unmonitored status; master table keeps `aqi_observed = False`; spatial inference performed via MRF graph. |

---

## 6. Pre-Modeling Data Preparation Completed

All planned data preparation remediation steps have been executed and verified:
1. **Clean CAAQMS Stations:** The concatenation artifact row (where `Station ID == '2025'`) was dropped across all 9 station files. Timestamps were aligned to a uniform 15-minute UTC grid.
2. **Official CPCB AQI Calculation:** Computed hourly concentrations and rolling 24h / 8h windows using official CPCB piecewise sub-index equations. Enforced the $\ge 3$ pollutants requirement with at least one PM fraction.
3. **Station-to-Zone Spatial Mapping:** Produced `data/processed/station_zone_mapping.csv` explicitly tagging all 15 zones with mapping methods and confidence.
4. **15-Zone Adjacency Graph:** Produced `data/processed/zone_adjacency.csv` with 29 verified planar shared-boundary edges.
5. **Master Dataset:** Generated `data/processed/master_table.csv` ($263,160\text{ rows}$) with explicit observation flags (`aqi_observed`, `rainfall_observed`, `temperature_observed`).
6. **Chronological Splitting:** Partitioned into `train.csv` (70.04%), `validate.csv` (15.05%), and `test.csv` (14.91%) with zero leakage. Documented in `reports/data_split.md`.

---

## 7. Pre-Modeling Verification & Data Governance Matrix

| Variable | Raw Source | Operational Resolution | Provenance & Representation | Total Observed Hours |
|---|---|---|---|:---:|
| **AQI** | 9 CAAQMS CSVs (2024–2025) | Hourly ($1\text{h}$) | Computed via official CPCB piecewise equations over rolling 24h/8h windows. Monitored in 8 zones; 7 zones unmonitored (`aqi_observed = False`). | 111,239 zone-hours |
| **Rainfall** | CAAQMS tipping buckets (2024–2025) | Hourly ($1\text{h}$) | Tipping bucket hourly sums in monitored zones (Zones 4, 5, 10, 14) with `rainfall_observed = True`. Manali Village corrupted gauge masked to NaN. | 63,951 zone-hours |
| **Rainfall (Historical Climatology)** | IMD Daily (1993–2023) | Daily | Segregated in `chennai_rainfall_climatology_1993_2023.csv`. Used exclusively for climatological priors and heavy rainfall threshold ($64.5\text{ mm/day}$) calibration. Not mixed into 2024–2025 operational rows. | 20,055 records |
| **Temperature** | IMD Daily (1951 – 2024-06-22) | Daily background mapped to timeline | Daily mean background ($T_{\text{mean}} = \frac{T_{\max} + T_{\min}}{2}$) mapped across hours of each calendar day up to `2024-06-22 23:00:00` with `temperature_resolution = 'daily'`. After June 22, 2024: `temperature_observed = False` and `temperature = NaN`. | 62,640 zone-hours (174 days $\times$ 24h $\times$ 15 zones) |

**Status:** All pre-modeling data preparation complete. All unit tests passing. HMM + MRF training halted pending user approval.

