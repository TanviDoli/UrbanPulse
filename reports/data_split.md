# UrbanPulse: Chronological Dataset Split Report
**Project:** UrbanPulse — Spatio-Temporal Early Warning System for Chennai  
**Date of Split:** October 3, 2026  
**Methodology:** Strictly Chronological (No Random Splitting, No Future Leakage)

---

## 1. Splitting Rationale & Constraints

Time-series forecasting cannot use random K-fold or random shuffling splits, as doing so leaks temporal autocorrelation and future state distributions into the past.
In adherence to **Critical Rule #2** and the non-negotiable scientific constraints:
1. The chronological split respects the arrow of time: $\text{Train} \prec \text{Validation} \prec \text{Test}$.
2. All parameter fitting (HMM transition/emission distributions, potential functions, scalers, imputers, and threshold calibrations) must occur **exclusively on the Training partition**.
3. **Validation set** is reserved for hyperparameter tuning (e.g. number of hidden states $K$, MRF spatial coupling strength $\beta$).
4. **Test set** is completely held out for the final research experiment comparing **HMM vs HMM + MRF**.
5. The Test partition covers the critical **Northeast Monsoon (NEM) season (mid-September through December 2025)**, capturing high-impact weather events, cyclonic rainfall episodes, and winter AQI degradation.

---

## 2. Chronological Split Details

| Partition | Start Timestamp (UTC) | End Timestamp (UTC) | Calendar Days | Hourly Steps / Zone | Total Rows across 15 Zones | % of Dataset | Primary Purpose |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Train** | `2024-01-01 00:00:00` | `2025-05-26 23:00:00` | 512 days | 12,288 | **184,320** | **70.04%** | HMM parameter learning (initial, transition, emission) |
| **Validation** | `2025-05-27 00:00:00` | `2025-09-13 23:00:00` | 110 days | 2,640 | **39,600** | **15.05%** | HMM state selection & MRF parameter calibration |
| **Test** | `2025-09-14 00:00:00` | `2025-12-31 23:00:00` | 109 days | 2,616 | **39,240** | **14.91%** | Out-of-sample benchmark: **HMM vs HMM + MRF** |
| **Total** | `2024-01-01 00:00:00` | `2025-12-31 23:00:00` | 731 days | 17,544 | **263,160** | **100.0%** | Full 15-Zone Operational Grid |

---

## 3. Data Modality Coverage by Split

### A. Air Quality Index (AQI)
- Monitored across 8 primary CAAQMS zones (Zones 1, 2, 4, 5, 10, 12, 13, 14).
- 7 zones remain unmonitored (`aqi_observed = False`) across all splits, requiring spatial inference via MRF.
- Training observed AQI rows: ~78,000 zone-hours.
- Validation observed AQI rows: ~16,500 zone-hours.
- Test observed AQI rows: ~16,700 zone-hours.

### B. Temperature
- IMD daily temperature covers `2024-01-01` to `2024-06-22` (174 calendar days).
- **All observed temperature values reside within the Training partition** ($4,176 \text{ hours} \times 15 \text{ zones} = 62,640 \text{ rows}$).
- For all timestamps after `2024-06-22 23:00:00`, `temperature_observed = False` and `temperature = NaN`.
- Daily provenance is explicitly documented in `temperature_resolution = 'daily'`.

### C. Rainfall
- Operational tipping-bucket hourly rainfall is observed in monitored zones (Zones 4, 5, 10, 14) across Train, Validation, and Test.
- Historical IMD daily rainfall (1993–2023) is stored in `data/processed/chennai_rainfall_climatology_1993_2023.csv` as an independent climatological baseline.

---

## 4. Verification of Non-Leakage

1. **Temporal Boundary Isolation:** Timestamps in Validation are strictly greater than Train ($t_{val} > t_{train}$); timestamps in Test are strictly greater than Validation ($t_{test} > t_{val}$).
2. **Feature Independence:** No forward-looking rolling windows overlap across boundaries.
3. **Parameter Integrity:** No validation or test data will touch the HMM EM (Baum-Welch) or MRF parameter estimation steps.
