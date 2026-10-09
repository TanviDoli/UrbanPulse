# UrbanPulse Chennai — Phase 4A: Historical Data Audit Report

**Date of Audit:** October 9, 2026  
**Auditor:** UrbanPulse Environmental Intelligence System (Autonomous Audit)  
**Scope:** Complete inventory, missing-data quantification, station-to-zone mapping, model coverage audit, and supplementary data recommendations for all 15 Greater Chennai Corporation (GCC) zones.

---

## Executive Summary

An exhaustive inspection of all raw and processed historical datasets, spatial boundaries, and trained model artifacts was conducted. The UrbanPulse historical data infrastructure currently comprises **9 continuous ambient air quality monitoring (CAAQMS) station datasets**, **2 long-term IMD meteorological reference datasets**, **1 official GCC 2025 GeoJSON boundary file**, and a **263,160-row multi-zone operational master table**.

### Key Audit Findings:
1. **Critical Temperature Vacuum (76.6% Missing in Master Table):**
   - In raw CAAQMS station data, **ambient temperature (AT) is 91.2% missing overall** (valid in only 2 of 9 stations: Gandhi Nagar at 11.0% and Manali Village at 66.2%; **0.0% valid in the other 7 stations**).
   - In the IMD temperature archive (`chennai_daily_temperature_1951_2024.csv`), observations are daily-only and **abruptly end on 22 June 2024**.
   - Consequently, from **23 June 2024 through 31 December 2025 (and into 2026)**, there is **zero measured temperature ground truth** for any Chennai zone in the operational master table. All 15 zones share the same 4,176 hours (23.8%) of historical daily background data.
2. **Severe Rainfall Spatial Disparity (75.7% Missing in Master Table):**
   - Raw tipping-bucket hourly rainfall is recorded at only **4 CAAQMS stations** (Kodungaiyur/Zone 4: 96.9%, Royapuram/Zone 5: 88.1%, Arumbakkam/Zone 10: 82.8%, Perungudi/Zone 14: 93.5%).
   - The remaining **11 GCC zones have zero continuous hourly rainfall observations** (Alandur, Manali, Velachery, Gandhi Nagar have 0.0% gauge records; Manali Village has severe hardware drift and is masked).
   - The historical IMD daily rainfall archive (`chennai_daily_rainfall_1991_2023.csv`) contains 21,416 daily records across 45 local rain gauges from 1993 to 2023, but **completely terminates in December 2023**.
3. **CAAQMS Air Quality Spatial Gaps:**
   - **8 of 15 GCC zones** have a physical CAAQMS monitoring station mapped to their polygon.
   - **7 of 15 GCC zones** (Madhavaram, Thiru-Vi-Ka Nagar, Ambattur, Anna Nagar, Teynampet, Valasaravakkam, Sholinganallur) are **completely unmonitored** by CAAQMS ground sensors.
   - Zone 1 (Gandhi Nagar/Ennore) suffers from **~90% sensor missingness** across criteria pollutants, leaving it insufficient for direct HMM training.
   - Therefore, valid HMM temporal air quality models exist for **7 monitored zones** (Zones 2, 4, 5, 10, 12, 13, 14), while the other 8 zones rely entirely on Spatial Markov Random Field (MRF) graph inference.

---

## 1. Inventory of Historical Datasets

| Dataset Identifier | File Path | Format | Record Count | Date Range (Earliest to Latest) | Native Resolution | Spatial Entity / Zone Mapping | Data Category & Provenance |
|---|---|---|---|---|---|---|---|
| **Raw CAAQMS: Alandur** | `data/raw/AQI/alandur-2024-25.csv` | CSV | 62,790 | 2024-01-01 00:00 to 2025-12-02 16:30 | 15-min | Alandur Bus Depot (CPCB) $\rightarrow$ **Zone 12 (Alandur)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Arumbakkam** | `data/raw/AQI/arumbakkam-2024-25.csv` | CSV | 70,177 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Arumbakkam (TNPCB) $\rightarrow$ **Zone 10 (Kodambakkam)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Gandhi Nagar** | `data/raw/AQI/gandhi-nagar-2024-25.csv` | CSV | 70,177 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Gandhi Nagar Ennore (TNPCB) $\rightarrow$ **Zone 1 (Thiruvottiyur)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Kodungaiyur** | `data/raw/AQI/kodungaiyur-2024-25.csv` | CSV | 70,177 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Kodungaiyur (TNPCB) $\rightarrow$ **Zone 4 (Tondiarpet)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Manali** | `data/raw/AQI/manali-2024-25.csv` | CSV | 63,740 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Manali (CPCB) $\rightarrow$ **Zone 2 (Manali)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Manali Village** | `data/raw/AQI/manali-village-2024-25.csv` | CSV | 70,177 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Manali Village (TNPCB) $\rightarrow$ **Zone 2 (Manali - Secondary)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Perungudi** | `data/raw/AQI/perungudi-2024-25.csv` | CSV | 70,177 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Perungudi (TNPCB) $\rightarrow$ **Zone 14 (Perungudi)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Royapuram** | `data/raw/AQI/royapuram-2024-25.csv` | CSV | 70,177 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Royapuram (TNPCB) $\rightarrow$ **Zone 5 (Royapuram)** | Measured CAAQMS ground sensor |
| **Raw CAAQMS: Velachery** | `data/raw/AQI/velachery-2024-25.csv` | CSV | 66,245 | 2024-01-01 00:00 to 2025-12-31 23:45 | 15-min | Velachery Res. Area (CPCB) $\rightarrow$ **Zone 13 (Adyar)** | Measured CAAQMS ground sensor |
| **Raw IMD Rainfall** | `data/raw/chennai_daily_rainfall_1991_2023.csv` | CSV | 21,416 | 1993-03-03 to 2023-12-26 | Daily | 45 IMD district/taluk/zone stations across Chennai | Historical IMD ground observations |
| **Raw IMD Temperature** | `data/raw/chennai_daily_temperature_1951_2024.csv` | CSV | 26,837 | 1951-01-01 to 2024-06-22 | Daily | Chennai Regional IMD Station | Historical IMD ground observations |
| **Chennai Zones GeoJSON** | `data/raw/chennai_zones_2025.geojson` | GeoJSON | 15 features | 2025 Administrative | Vector Polygons | 15 GCC Administrative Zones (Zones 1 to 15) | Official GCC Ward & Zone boundaries |
| **Processed Clean CAAQMS** | `data/processed/clean_aqi_stations/*.csv` (9 files) | CSV | 70,176 / file | 2024-01-01 00:00 to 2025-12-31 23:45 | Exact 15-min grid | 9 CAAQMS stations | Cleaned, reindexed, negative values clipped |
| **IMD Rainfall Climatology** | `data/processed/chennai_rainfall_climatology_1993_2023.csv` | CSV | 21,349 | 1993-03-03 to 2023-12-26 | Daily | Deduplicated IMD multi-station rainfall records | Historical reference climatology |
| **IMD Clean Temperature** | `data/processed/chennai_temperature_daily_1951_2024.csv` | CSV | 26,837 | 1951-01-01 to 2024-06-22 | Daily | Chennai Regional IMD (Temp Mean, Max, Min) | Cleaned, outlier-corrected temperature series |
| **Operational Master Table** | `data/processed/master_table.csv` | CSV | 263,160 | 2024-01-01 00:00 to 2025-12-31 23:00 | Hourly | 15 GCC Zones (17,544 hours $\times$ 15 zones) | Assembled operational training & validation table |
| **Train Dataset** | `data/processed/train.csv` | CSV | 184,320 | 2024-01-01 00:00 to 2025-05-26 23:00 | Hourly | 15 GCC Zones (70.0% chronological split) | Strict training partition |
| **Validation Dataset** | `data/processed/validate.csv` | CSV | 39,600 | 2025-05-27 00:00 to 2025-09-13 23:00 | Hourly | 15 GCC Zones (15.0% chronological split) | Hyperparameter / K selection partition |
| **Test Dataset** | `data/processed/test.csv` | CSV | 39,240 | 2025-09-14 00:00 to 2025-12-31 23:00 | Hourly | 15 GCC Zones (15.0% chronological split) | Isolated hold-out test partition |
| **Station-Zone Mapping** | `data/processed/station_zone_mapping.csv` | CSV | 15 | Static metadata | N/A | 15 GCC Zones $\leftrightarrow$ 9 CAAQMS stations | Point-in-polygon containment mapping |
| **Zone Adjacency Graph** | `data/processed/zone_adjacency.csv` | CSV | 105 | Static topology | N/A | Pairwise zone adjacencies (29 boundary edges) | Spatial MRF graph connectivity weights |

---

## 2. Quantitative Coverage & Missing-Data Analysis

### 2.1 Environmental Variables in Raw CAAQMS Stations (613,837 Total 15-Minute Records)

Across the 9 raw CAAQMS station files, 29 columns are present. The table below documents valid observations, missing counts, and coverage across all monitoring stations:

| Environmental Variable | Common Units | Total Records | Valid Numeric Observations | Missing / Corrupt Records | Valid Coverage (%) | Reporting Stations (Count / 9) | Observed Value Range (Min / Mean / Max) |
|---|---|---|---|---|---|---|---|
| **PM2.5** | $\mu\text{g/m}^3$ | 613,837 | 462,366 | 151,471 | **75.3%** | 9 / 9 | 0.00 / 41.22 / 999.00 |
| **PM10** | $\mu\text{g/m}^3$ | 613,837 | 471,799 | 142,038 | **76.9%** | 9 / 9 | 0.00 / 84.15 / 999.00 |
| **$\text{NO}_2$** | $\mu\text{g/m}^3$ | 613,837 | 460,051 | 153,786 | **75.0%** | 9 / 9 | 0.00 / 22.14 / 498.00 |
| **$\text{NO}_x$** | $\text{ppb}$ | 613,837 | 467,319 | 146,518 | **76.1%** | 9 / 9 | 0.00 / 26.85 / 500.00 |
| **$\text{SO}_2$** | $\mu\text{g/m}^3$ | 613,837 | 473,839 | 139,998 | **77.2%** | 9 / 9 | 0.00 / 11.42 / 380.00 |
| **$\text{CO}$** | $\text{mg/m}^3$ | 613,837 | 422,207 | 191,630 | **68.8%** | 8 / 9 (Arumbakkam = 0%) | 0.00 / 0.74 / 25.00 |
| **$\text{Ozone}$ ($\text{O}_3$)** | $\mu\text{g/m}^3$ | 613,837 | 418,649 | 195,188 | **68.2%** | 8 / 9 (Arumbakkam = 0%) | 0.00 / 24.31 / 250.00 |
| **$\text{NH}_3$** | $\mu\text{g/m}^3$ | 613,837 | 459,754 | 154,083 | **74.9%** | 9 / 9 | 0.00 / 28.16 / 400.00 |
| **Ambient Temp (AT)** | $^\circ\text{C}$ | 613,837 | 54,159 | 559,678 | **8.8%** | **2 / 9** (7 stations = 0%) | 18.20 / 30.15 / 44.80 |
| **Rainfall (RF)** | $\text{mm}$ | 613,837 | 300,875 | 312,962 | **49.0%** | 5 / 9 (4 monitored + 1 drift) | 0.00 / 0.12 / 85.00 |
| **Cumulative Rain (TOT-RF)**| $\text{mm}$ | 613,837 | 421,056 | 192,781 | **68.6%** | 6 / 9 | 0.00 / 412.50 / 9999.00 |
| **Relative Humidity (RH)** | $\%$ | 613,837 | 456,106 | 157,731 | **74.3%** | 9 / 9 | 5.00 / 72.40 / 100.00 |
| **Wind Speed (WS)** | $\text{m/s}$ | 613,837 | 501,770 | 112,067 | **81.7%** | 9 / 9 | 0.00 / 2.14 / 25.00 |
| **Wind Direction (WD)** | $\text{deg}$ | 613,837 | 446,168 | 167,669 | **72.7%** | 9 / 9 | 0.00 / 164.20 / 360.00 |
| **Barometric Pressure (BP)**| $\text{mmHg}$ | 613,837 | 343,819 | 270,018 | **56.0%** | 6 / 9 | 740.0 / 755.2 / 770.0 |
| **Solar Radiation (SR)** | $\text{W/m}^2$ | 613,837 | 435,501 | 178,336 | **71.0%** | 8 / 9 | 0.00 / 185.40 / 1200.0 |
| **Benzene** | $\mu\text{g/m}^3$ | 613,837 | 397,975 | 215,862 | **64.8%** | 8 / 9 | 0.00 / 1.85 / 50.00 |
| **Toluene** | $\mu\text{g/m}^3$ | 613,837 | 307,645 | 306,192 | **50.1%** | 7 / 9 | 0.00 / 4.12 / 100.00 |
| **O-Xylene** | $\mu\text{g/m}^3$ | 613,837 | 0 | 613,837 | **0.0%** | 0 / 9 (100% missing) | N/A |
| **Total Xylene** | $\mu\text{g/m}^3$ | 613,837 | 27,601 | 586,236 | **4.5%** | 2 / 9 | 0.00 / 2.45 / 45.00 |

---

### 2.2 Station-by-Station Coverage Breakdown (Raw CAAQMS)

| Station File Name | Official CAAQMS Station Name | Mapped GCC Zone | Total Records | Date Span | $\text{PM}_{2.5}$ Valid (%) | $\text{PM}_{10}$ Valid (%) | $\text{NO}_2$ Valid (%) | $\text{SO}_2$ Valid (%) | $\text{CO}$ Valid (%) | $\text{O}_3$ Valid (%) | $\text{AT}$ Valid (%) | $\text{RF}$ Valid (%) | Operational Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `alandur-2024-25.csv` | Alandur Bus Depot (CPCB) | Zone 12 | 62,789 | 2024-01-01 to 2025-12-02 | 92.1% | 92.5% | 90.0% | 96.3% | 99.5% | 98.3% | **0.0%** | **0.0%** | Complete pollutants; missing weather; truncated on 02 Dec 2025 |
| `arumbakkam-2024-25.csv` | Arumbakkam (TNPCB) | Zone 10 | 70,176 | 2024-01-01 to 2025-12-31 | 76.1% | 79.6% | 61.7% | 81.0% | **0.0%** | **0.0%** | **0.0%** | **82.8%** | Excellent rainfall & PM; missing CO, $\text{O}_3$, AT |
| `gandhi-nagar-2024-25.csv`| Gandhi Nagar, Ennore (TNPCB) | Zone 1 | 70,176 | 2024-01-01 to 2025-12-31 | **9.9%** | **8.5%** | **10.3%** | **12.6%** | **3.8%** | **8.6%** | **11.0%** | **0.0%** | Severe missingness (~90% missing across all sensors) |
| `kodungaiyur-2024-25.csv` | Kodungaiyur (TNPCB) | Zone 4 | 70,176 | 2024-01-01 to 2025-12-31 | 92.5% | 92.1% | 94.5% | 88.7% | 88.7% | 75.1% | **0.0%** | **96.9%** | High completeness criteria pollutants & rainfall |
| `manali-2024-25.csv` | Manali (CPCB) | Zone 2 | 63,739 | 2024-01-01 to 2025-12-31 | 82.4% | 87.5% | 88.1% | 86.7% | 97.9% | 98.3% | **0.0%** | **0.0%** | High completeness pollutants; weather unequipped |
| `manali-village-2024-25.csv`| Manali Village (TNPCB) | Zone 2 | 70,176 | 2024-01-01 to 2025-12-31 | 57.2% | 58.5% | 56.8% | 66.5% | 64.5% | 66.0% | **66.2%** | **67.5%** | Secondary station; rainfall sensor has severe drift |
| `perungudi-2024-25.csv` | Perungudi (TNPCB) | Zone 14 | 70,176 | 2024-01-01 to 2025-12-31 | 92.1% | 95.6% | 97.5% | 96.7% | 97.3% | 96.1% | **0.0%** | **93.5%** | High completeness criteria pollutants & rainfall |
| `royapuram-2024-25.csv` | Royapuram (TNPCB) | Zone 5 | 70,176 | 2024-01-01 to 2025-12-31 | 82.9% | 85.4% | 83.1% | 71.6% | 75.7% | 80.5% | **0.0%** | **88.1%** | Complete criteria pollutants & rainfall |
| `velachery-2024-25.csv` | Velachery Res. Area (CPCB) | Zone 13 | 66,244 | 2024-01-01 to 2025-12-31 | 96.4% | 95.7% | 96.7% | 98.6% | 99.2% | 98.6% | **0.0%** | **0.0%** | Highest pollutant completeness; missing weather |

---

### 2.3 Master Operational Table Coverage (All 15 GCC Zones over 17,544 Hourly Steps: 263,160 Records)

The operational master table spans 2 years (2024-01-01 00:00 to 2025-12-31 23:00 UTC). The table below quantifies exact valid hours and percentages by zone:

| Zone ID | Zone Label | Mapped Primary Station | Total Hours | Valid AQI Hours | AQI Valid (%) | Valid Rain Hours | Rain Valid (%) | Valid Temp Hours | Temp Valid (%) | Temperature Ground Truth Source & Period |
|---|---|---|---|---|---|---|---|---|---|---|
| **Zone 1** | Thiruvottiyur | Gandhi Nagar Ennore | 17,544 | 1,537 | **8.8%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 2** | Manali | Manali (CPCB) | 17,544 | 15,080 | **86.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 3** | Madhavaram | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 4** | Tondiarpet | Kodungaiyur | 17,544 | 16,725 | **95.3%** | 17,068 | **97.3%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 5** | Royapuram | Royapuram | 17,544 | 15,059 | **85.8%** | 15,537 | **88.6%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 6** | Thiru-Vi-Ka Nagar | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 7** | Ambattur | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 8** | Anna Nagar | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 9** | Teynampet | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 10**| Kodambakkam | Arumbakkam | 17,544 | 14,536 | **82.9%** | 14,918 | **85.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 11**| Valasaravakkam | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 12**| Alandur | Alandur Bus Depot | 17,544 | 14,751 | **84.1%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 13**| Adyar | Velachery Res. Area | 17,544 | 16,382 | **93.4%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 14**| Perungudi | Perungudi | 17,544 | 17,169 | **97.9%** | 16,428 | **93.6%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **Zone 15**| Sholinganallur | *None (Unmonitored)* | 17,544 | 0 | **0.0%** | 0 | **0.0%** | 4,176 | **23.8%** | IMD Daily (ends 2024-06-22) |
| **TOTAL** | **All 15 Zones** | **9 Stations** | **263,160** | **111,239** | **42.3%** | **63,951** | **24.3%** | **61,560** | **23.4%** | **13,368 hours missing per zone** |

---

## 3. Station-to-Zone Mapping and Coverage Gaps

```
Zone 01 [Thiruvottiyur]   ── CAAQMS Gandhi Nagar (9.9% PM2.5, 0% Rain, 11% AT) -> Severe Missingness Gap
Zone 02 [Manali]           ── CAAQMS Manali (82.4% PM2.5, 0% Rain) + Manali Village (Drift masked)
Zone 03 [Madhavaram]       ── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
Zone 04 [Tondiarpet]       ── CAAQMS Kodungaiyur (92.5% PM2.5, 96.9% Rain) -> MONITORED
Zone 05 [Royapuram]        ── CAAQMS Royapuram (82.9% PM2.5, 88.1% Rain) -> MONITORED
Zone 06 [Thiru-Vi-Ka Nagar]── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
Zone 07 [Ambattur]         ── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
Zone 08 [Anna Nagar]       ── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
Zone 09 [Teynampet]        ── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
Zone 10 [Kodambakkam]      ── CAAQMS Arumbakkam (76.1% PM2.5, 82.8% Rain) -> MONITORED (No CO/O3)
Zone 11 [Valasaravakkam]   ── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
Zone 12 [Alandur]          ── CAAQMS Alandur (92.1% PM2.5, 0% Rain) -> MONITORED (Truncated Dec 2025)
Zone 13 [Adyar]            ── CAAQMS Velachery (96.4% PM2.5, 0% Rain) -> MONITORED (No Rain/Temp)
Zone 14 [Perungudi]        ── CAAQMS Perungudi (92.1% PM2.5, 93.5% Rain) -> MONITORED
Zone 15 [Sholinganallur]   ── UNMONITORED (0 CAAQMS Stations) -> Complete Gap
```

### Mapping Topology Summary:
- **8 Monitored Zones:** Zones 1, 2, 4, 5, 10, 12, 13, 14.
- **7 Unmonitored Zones:** Zones 3, 6, 7, 8, 9, 11, 15.
- **Dual-Station Zone:** Zone 2 contains both `manali-2024-25.csv` (Primary, CPCB) and `manali-village-2024-25.csv` (Secondary, TNPCB).
- **Rainfall Monitoring Deficit:** Only Zones 4, 5, 10, 14 have operational tipping-bucket rain gauges in CAAQMS. Zones 1, 2, 12, 13 lack rain gauge hardware.

---

## 4. Existing Model Audit and Input Requirements

The UrbanPulse model architecture consists of **Zone-Specific Temporal Hidden Markov Models (HMM)** coupled with an **All-Zone Spatial Markov Random Field (MRF)**:

### 4.1 Model Inventory & Training Provenance

| Target Variable | Model Type | Checkpoint Artifact Path | Trained Zones | Untrained / Skipped Zones | Training Dataset Source | Selection Metric | Required Input Features | Native Resolution |
|---|---|---|---|---|---|---|---|---|
| **AQI** | Temporal HMM | `models/hmm/hmm_aqi_zone_{z}.joblib`, `models/hmm/hmm_aqi.joblib` | **7 Zones:** 2, 4, 5, 10, 12, 13, 14 ($K=6$ states each) | **8 Zones:** 1 (insufficient $N<2000$), 3, 6, 7, 8, 9, 11, 15 (unmonitored) | `data/processed/train.csv` (AQI column) | Val MAE (1h) | 24-hr historical AQI sequence ($y_{t-23:t}$) | Hourly (CPCB rolling breakpoint sub-index) |
| **Rainfall** | Temporal HMM | `models/hmm/hmm_rainfall_zone_{z}.joblib`, `models/hmm/hmm_rainfall.joblib` | **4 Zones:** Zone 4 ($K=3$), Zones 5, 10, 14 ($K=4$) | **11 Zones:** 1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 15 ($N_{\text{obs}} < 2000$) | `data/processed/train.csv` (rainfall column) | Val MAE (1h) | 24-hr historical rainfall sequence ($y_{t-23:t}$) | Hourly sum ($\text{mm}$) |
| **Temperature** | Temporal HMM | `models/hmm/hmm_temperature_zone_{z}.joblib`, `models/hmm/hmm_temperature.joblib` | **All 15 Zones:** Zones 1 through 15 ($K=4$ states each) | **0 Zones** (All zones share identical daily background) | `data/processed/train.csv` (daily aggregated $T_{\text{mean}}$) | Train BIC (0 validation records exist) | Daily historical temperature sequence ($T_{d-24:d}$) | **Daily regime** ($^\circ\text{C}$); sub-daily is unavailable |
| **Spatial AQI** | Spatial MRF | `models/hmm_mrf/mrf_aqi.joblib` | All 15 Zones coupled via `zone_adjacency.csv` ($\beta = 0.65$) | None | `data/processed/train.csv` | Spatial loss | Zone HMM emissions for 7 monitored zones; graph priors for 8 unmonitored | Hourly |
| **Spatial Rainfall**| Spatial MRF | `models/hmm_mrf/mrf_rainfall.joblib` | 15 Zones ($\beta = 0.00$, uncoupled) | 11 Zones receive `UNMONITORED_INSUFFICIENT_DATA` | `data/processed/train.csv` | Spatial loss | Station rainfall; spatial coupling zeroed due to localized convection | Hourly |
| **Spatial Temperature**| Spatial MRF | `models/hmm_mrf/mrf_temperature.joblib` | All 15 Zones ($\beta = 0.50$) | None | `data/processed/train.csv` | Spatial loss | Daily HMM temperature predictions | Daily |

### 4.2 Multi-Horizon Forecasting Constraints
- **AQI Multi-Horizon (1h, 3h, 6h, 24h):** Fully functional. For monitored zones, HMM predicts future hidden state distribution; for unmonitored zones, Spatial MRF infers posterior expectation across the 15-zone graph.
- **Rainfall Multi-Horizon (1h, 3h, 6h, 24h):** Functioning for monitored zones (4, 5, 10, 14). For the other 11 zones, returns `UNMONITORED_INSUFFICIENT_DATA` because MRF spatial coupling $\beta = 0.0$.
- **Temperature Sub-Daily Limitation (1h, 3h, 6h):** **Strictly unavailable**. Daily HMM steps are in days, not hours. 1h/3h/6h predictions are not fabricated. Only 24h (next-day $T_{\max}$) is produced.

### 4.3 Missing Value Handling in Existing Pipelines
- **HMM Fitting:** In `TemporalHMM.fit()`, NaNs are masked during Baum-Welch expectation-maximization.
- **HMM Forecasting:** Context windows (`context[-24:]`) carry forward the last valid observation (`context.append(context[-1])`) or inject standard `0.0` for unobserved rainfall steps.
- **Live Pipeline (`src.live.obs_preprocessor`):** Fetches current WeatherAPI data for zone centroids, computes live CPCB sub-index AQI, and feeds observations directly to HMM emission inference.

---

## 5. Data-Quality and Integrity Issues Identified

1. **Header Corruption in CAAQMS Files:**
   All 9 CAAQMS CSV files contain Unicode replacement bytes (`\ufffd`) in headers representing $\mu$ (micro, e.g., `PM2.5 (\ufffdg/m3)` instead of `PM2.5 (µg/m3)`). Handled by prefix matching in `clean_station_file()`.
2. **Artifact Header Row ('2025'):**
   In all raw CAAQMS CSVs, a rogue metadata/concatenation row exists where `Station ID == '2025'` or `Timestamp` is NaN. Must always be filtered before datetime parsing.
3. **Manali Village Rainfall Sensor Hardware Drift:**
   `manali-village-2024-25.csv` contains an erratic, continuous rainfall baseline drift even during dry periods. Explicitly invalidated to NaN in `clean_and_build_all.py` to preserve scientific integrity.
4. **IMD Date Format Ambiguity:**
   `chennai_daily_rainfall_1991_2023.csv` and `chennai_daily_temperature_1951_2024.csv` use `DD-MM-YYYY`. Standard ISO parsers fail on 60%+ of records unless `format='%d-%m-%Y'` is strictly specified.
5. **IMD Temperature Non-Numeric Sentinels and Outliers:**
   `chennai_daily_temperature_1951_2024.csv` contains string tokens `'-----'` for missing values (9 in Temp Max, 28 in Temp Min) and a physical anomaly on `2023-12-01` where `Temp Min = 92.0` $^\circ\text{C}$ (corrected to seasonal median 22.5 $^\circ\text{C}$).
6. **Date Range Truncation in Alandur Station:**
   `alandur-2024-25.csv` ends prematurely on `2025-12-02 16:30:00`, leaving December 2025 unmonitored for Zone 12.
7. **Complete Absence of Ground-Truth Temperature After June 2024:**
   The IMD temperature file stops at `2024-06-22`. No CAAQMS station reliably recorded ambient temperature after that date.

---

## 6. Prioritized Gap Matrix

| Priority Tier | Variable | Affected Zones | Nature of Deficit | Impact on System |
|---|---|---|---|---|
| **P1 (Critical)** | **Hourly Ambient Temperature (AT)** | **All 15 Zones** | Complete ground-truth absence from 23 June 2024 to present; 0% hourly data in 7 CAAQMS stations | Prevents training sub-daily temperature HMMs (1h, 3h, 6h); invalidates heatwave warning validation for 2025-2026 |
| **P1 (Critical)** | **Hourly Rainfall (RF)** | **11 Zones** (1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 15) | Zero hourly gauge data in 11 zones; historical IMD rain ended in Dec 2023 | 73% of Chennai zones cannot predict localized flash-flood or heavy-rain risks |
| **P2 (High)** | **Criteria Air Pollutants ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$)** | **7 Zones** (3, 6, 7, 8, 9, 11, 15) + **Zone 1** (Ennore) | Zero CAAQMS stations in 7 zones; ~90% missing in Zone 1 | Relies 100% on spatial graph interpolation without local emission calibration |
| **P2 (High)** | **Carbon Monoxide ($\text{CO}$) & Ozone ($\text{O}_3$)** | **Zone 10** (Arumbakkam) | Missing sensor hardware (0.0% valid) | Incomplete CPCB rolling 8-hr AQI sub-index calculation for Central Chennai |
| **P3 (Medium)** | **Daily Climatological Reference** | All 15 Zones | IMD Rainfall ends Dec 2023; IMD Temp ends June 2024 | Climatological baseline anomaly metrics cannot be updated for 2025-2026 |

---

## 7. Recommended Next Data-Collection Steps

### 7.1 Weather Gaps Suited for Open-Meteo Historical Weather API
*Target Period:* **1 January 2025 through 8 October 2026** (hourly resolution).  
*Source:* Open-Meteo Historical Weather API (`https://open-meteo.com/en/docs/historical-weather-api`), utilizing ECMWF ERA5 / ERA5-Land reanalysis.

1. **Hourly Ambient Temperature (`temperature_2m`):**
   - Retrieve hourly $T_{2m}$ ($^\circ\text{C}$) for all 15 Chennai zone centroids.
   - **Benefit:** Completely eliminates the 76.6% temperature gap in the master table from 23 June 2024 to 8 October 2026.
   - **Enables:** Training legitimate sub-daily (1h, 3h, 6h) temperature HMM models for all 15 zones.
2. **Hourly Precipitation (`precipitation`, `rain`):**
   - Retrieve hourly precipitation ($\text{mm}$) for all 15 zone centroids.
   - **Benefit:** Fills the 75.7% rainfall gap for the 11 currently unmonitored zones.
   - **Enables:** Training spatial and temporal precipitation models across the entire urban expanse.
3. **Supplementary Meteorological Regressors:**
   - Retrieve `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, and `wind_direction_10m`.
   - **Benefit:** Provides complete boundary layer inputs to improve pollutant dispersion modeling.

### 7.2 Pollutant Gaps Requiring Dedicated Air Quality Sources (Not Open-Meteo)
*Crucial Constraint:* **Open-Meteo does NOT provide measured ground-level criteria air pollutant observations.** Atmospheric reanalysis cannot substitute for CAAQMS station measurements.
- For criteria pollutants in the 7 unmonitored zones and Zone 1:
  - Acquire archival continuous monitoring data from **CPCB official archives** or **TNPCB CAAQMS portal** for additional operational stations (e.g., IIT Madras, Manali industrial monitors, or mobile sensor networks).
  - Do not use weather reanalysis as synthetic AQI observations.

### 7.3 Datasets That Are Sufficient and Must NOT Be Re-Downloaded
- **High-Completeness CAAQMS Records (2024-2025):**
  - Velachery (`velachery-2024-25.csv`): 96.4% $\text{PM}_{2.5}$ completeness.
  - Kodungaiyur (`kodungaiyur-2024-25.csv`): 92.5% $\text{PM}_{2.5}$ & 96.9% RF completeness.
  - Perungudi (`perungudi-2024-25.csv`): 92.1% $\text{PM}_{2.5}$ & 93.5% RF completeness.
  - Alandur (`alandur-2024-25.csv`): 92.1% $\text{PM}_{2.5}$ completeness (up to Dec 2, 2025).
  - Manali (`manali-2024-25.csv`): 82.4% $\text{PM}_{2.5}$ completeness.
- **Historical Reference Baselines:**
  - `chennai_daily_rainfall_1991_2023.csv` (1993-2023) is sufficient as a long-term climatological rainfall baseline.
  - `chennai_daily_temperature_1951_2024.csv` (1951-2024) is sufficient as a long-term climatological temperature baseline.

### 7.4 Data Needing Validation Before Retraining
- **Open-Meteo Calibration Check:**
  - Before merging Open-Meteo data into `master_table.csv`, run a cross-validation step comparing Open-Meteo historical reanalysis against:
    1. The 4 monitored CAAQMS rain gauges (Zones 4, 5, 10, 14) during overlapping 2024-2025 periods to determine local precipitation bias / scaling factors.
    2. IMD daily temperatures for January-June 2024 to verify $T_{\max}$ and $T_{\min}$ alignment.
- **Strict Distinction:**
  - Any reanalysis-derived inputs must be tagged with explicit provenance flags (e.g., `provenance: 'open_meteo_reanalysis'`) and never labeled as ground-truth station observations.

---

## 8. Summary Table of Coverage by Source

| Source / Network | Parameter Types | Spatial Coverage | Temporal Span | Resolution | Data Quality / Completeness | Recommended Action |
|---|---|---|---|---|---|---|
| **CAAQMS (TNPCB/CPCB)** | $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3, \text{NH}_3$ | 8 Zones monitored (7 unmonitored) | 2024-01-01 to 2025-12-31 | 15-min | High for 6 stations; severe missing in Zone 1 | Retain valid stations; seek supplementary CAAQMS stations for 7 unmonitored zones |
| **CAAQMS Weather Sensors** | Tipping-bucket RF, Ambient Temp | 4 Zones have Rain; 0 Zones have reliable Temp | 2024-01-01 to 2025-12-31 | 15-min | Rain: 82-96% in 4 zones; Temp: 91.2% missing overall | Retain 4 rain gauges; replace missing AT via Open-Meteo |
| **IMD Historical Temp** | Daily $T_{\max}, T_{\min}, T_{\text{mean}}$ | Regional Chennai baseline | 1951-01-01 to 2024-06-22 | Daily | Complete up to 2024-06-22; abrupt halt afterwards | Retain for climatology; supplement 2024-06-23 to 2026-10-08 via Open-Meteo |
| **IMD Historical Rain** | Daily rainfall | 45 local rain gauges | 1993-03-03 to 2023-12-26 | Daily | Complete up to 2023-12-26; zero 2024-2026 records | Retain for climatology; supplement 2025-2026 via Open-Meteo |
| **Open-Meteo (Proposed)** | Hourly $T_{2m}$, Precipitation, RH, Wind | All 15 GCC Zone centroids | 2025-01-01 to 2026-10-08 | Hourly | Continuous global reanalysis (ERA5 / ERA5-Land) | **Proceed with targeted fetch for weather parameters only** |

---
*Report generated and validated autonomously against UrbanPulse Chennai codebase.*
