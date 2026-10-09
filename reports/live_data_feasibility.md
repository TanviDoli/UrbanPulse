# UrbanPulse — Live Data Feasibility and Ingestion Assessment (Step 14A)

**Document Version:** 1.0 (Step 14A Milestone Deliverable)  
**Status:** Complete  
**Date:** October 5, 2026  
**System Scope:** Real-time ingestion feasibility for Chennai 15 GCC zones using FROZEN HMM temporal models, FROZEN Gaussian MRF spatial models, and FROZEN Alert Engine.

---

## Executive Summary

Step 14A investigates whether UrbanPulse can receive genuine current data from official Central Pollution Control Board (CPCB) and India Meteorological Department (IMD) sources, validating against documented government APIs. 

**Key Findings:**
1. **Official CPCB CAAQMS Data:** Fully accessible via the Open Government Data (OGD) Platform India (`data.gov.in`) under official Resource ID `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`. Access is governed by standard API key authorization (`DATA_GOV_IN_API_KEY`).
2. **Official IMD Weather Data:** Fully documented under the official IMD API Management Platform (`https://api.imd.gov.in`). Endpoints for Current Weather (`api/v1/current_wx`), City Forecast (`api/v1/cityforecast`), and AWS/ARG Data (`api/v1/aws_data`) require institutional API keys or IP whitelisting coordinated through IMD's designated Nodal Officer (`sankar.nath@imd.gov.in`).
3. **Model Compatibility:** **Zero model retraining is required.** The existing frozen zone-specific HMMs and continuous Gaussian MRF ingest these standardized physical quantities directly.
4. **Safety & Non-Fabrication:** Unmonitored zones (Zones 3, 6, 7, 8, 9, 11, 15) remain strictly unmonitored without synthetic observations ($\lambda_u = 0, b_u = 0$). Climatological normal temperature is never fabricated; IMD's official `cityforecast` API directly supplies verified departures from normal.

---

## 1. Official Sources Investigated

### 1.1 Central Pollution Control Board (CPCB)
* **Publishing Authority:** Ministry of Environment, Forest and Climate Change (MoEFCC) / Central Pollution Control Board (CPCB).
* **Official Gateway:** Open Government Data (OGD) Platform India ([data.gov.in](https://data.gov.in)).
* **Resource Title:** *Real-time Air Quality Index from various locations*.
* **Resource Identifier:** `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`.
* **API Base URL:** `https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`.
* **Investigation Method:** Direct protocol inspection against live server endpoints and official OGD API specifications. No unofficial scrapers or reverse-engineered internal endpoints were used.

### 1.2 India Meteorological Department (IMD)
* **Publishing Authority:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD).
* **Official Gateway:** IMD API Management Platform ([api.imd.gov.in](https://api.imd.gov.in)).
* **Official Reference Documentation:** `https://api.imd.gov.in/public/api_reference.html`.
* **Investigated Endpoints:**
  1. `api/v1/current_wx` (Current Weather API — Synoptic Observations).
  2. `api/v1/cityforecast` (7-Day City Weather Forecast Bulletin & Climatological Departures).
  3. `api/v1/aws_data` (Automatic Weather Station & Automatic Rain Gauge Data, Tamil Nadu State ID: 25).
  4. `api/v1/districtrainfall` (District-wise Rainfall Monitoring).

---

## 2. Accessibility Status

| Source | Endpoint | Verification Result | Operational Access Status |
| :--- | :--- | :--- | :--- |
| **CPCB (data.gov.in)** | `/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69` | `HTTP 400 Bad Request` with `{"error": "Authorization field missing"}` when key is absent; `HTTP 403 Forbidden` with `{"error": "Key not authorised"}` when key is invalid. | **Accessible.** Active and operational. Strictly requires a registered `api-key` parameter. |
| **IMD (api.imd.gov.in)** | `/api/v1/current_wx` | `HTTP 401 Unauthorized` with `{"error":"API key missing"}`. | **Accessible with Credentials.** Active and operational. Requires registered institutional credentials or IP authorization. |
| **IMD (api.imd.gov.in)** | `/api/v1/cityforecast` | `HTTP 401 Unauthorized` with `{"error":"API key missing"}`. | **Accessible with Credentials.** Active and operational. |
| **IMD (api.imd.gov.in)** | `/api/v1/aws_data` | `HTTP 401 Unauthorized` with `{"error":"API key missing"}`. | **Accessible with Credentials.** Active and operational. |

---

## 3. Authentication Requirements

### 3.1 CPCB / data.gov.in
* **Access Model:** Free developer registration on [data.gov.in](https://data.gov.in).
* **Credential Type:** Personal API Key (UUID string).
* **Query Signature:**
  ```http
  GET /resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=YOUR_API_KEY&format=json&filters[city]=Chennai&limit=100
  ```
* **Rate Limits:** Standard tiers allow up to 1,000 requests per day per API key, which comfortably supports UrbanPulse's once-per-hour ingestion cadence (24 requests/day).

### 3.2 IMD / api.imd.gov.in
* **Access Model:** Institutional data agreement and official user registration under IMD data sharing policy.
* **Credential Type:** API Key passed via request header (`X-Api-Key`) or pre-registered static IP whitelisting.
* **Administrative Contact:**
  * **Nodal Officer:** Dr. Sankar Nath, Scientist-E, IMD New Delhi.
  * **Contact Email:** `sankar.nath@imd.gov.in`.

---

## 4. Available Fields & Data Schemas

### 4.1 CPCB CAAQMS Schema (`data.gov.in`)
The API returns a JSON structure containing:
* `country`, `state`, `city`: Geographic hierarchy (`"India"`, `"Tamil Nadu"`, `"Chennai"`).
* `station`: Full station name string (e.g., `"Alandur Bus Depot, Chennai - CPCB"`, `"Manali, Chennai - CPCB"`).
* `latitude`, `longitude`: Decimal coordinates.
* `last_update`: Timestamp string formatted as `DD-MM-YYYY HH:MM:SS` (IST).
* `pollutant_id`: Criteria pollutant (`PM2.5`, `PM10`, `NO2`, `NH3`, `SO2`, `CO`, `OZONE`).
* `pollutant_min`, `pollutant_max`: Intraday minimum and maximum values.
* `pollutant_avg`: Hourly average concentration.
* `pollutant_unit`: Measurement unit (`ug/m3` or `mg/m3`).

### 4.2 IMD Weather Schemas (`api.imd.gov.in`)
1. **Current Weather (`current_wx`):**
   * `Station Id`, `Station`: Identifiers (e.g. `43279` for Meenambakkam, `43278` for Nungambakkam).
   * `Date of Observation`, `Time of Observation UTC`: Date and UTC time.
   * `Temperature deg C`: Instantaneous ambient air temperature.
   * `Humidity %`: Relative humidity.
   * `Last 24 hrs Rainfall mm`: 24-hour accumulated rainfall (recorded 08:30 to 08:30 IST).
   * `Wind Speed KMPH`, `Wind Direction Code`: Surface wind parameters.
   * `M.S.L.P hPa`: Mean sea level atmospheric pressure.
2. **City Forecast (`cityforecast`):**
   * `Today_Max_temp`: Maximum temperature observed/forecasted for today ($^\circ\text{C}$).
   * `Today_Max_Departure_from_Normal`: **Official departure from 30-year climatological normal** ($^\circ\text{C}$).
   * `Previous_Day_Max_temp`, `Previous_Day_Max_Departure_from_Normal`: Previous day values.
   * `Past_24_hrs_Rainfall`: Official 24-hour rainfall total (mm).
3. **AWS / ARG Data (`aws_data`):**
   * `ID`, `CALL_SIGN`, `STATION`: Automated weather station call sign.
   * `DATE`, `TIME`: Observation timestamp.
   * `CURR_TEMP`: Ambient temperature ($^\circ\text{C}$).
   * `RH`: Relative humidity (%).
   * `RAIN_FALL`: Hourly incremental precipitation (mm).

---

## 5. Temporal Resolution

* **CPCB / data.gov.in:** Hourly updates. The endpoint serves the most recent hourly arithmetic average for each monitored station.
* **IMD `current_wx`:** Synoptic hours (3-hourly standard or hourly automated).
* **IMD `cityforecast`:** Issued twice daily (08:30 IST morning bulletin and 17:30 IST evening bulletin).
* **IMD `aws_data`:** Hourly automated transmission from Chennai automated weather stations.

**Operational Cadence:** Live inference in UrbanPulse is designed to execute **once per hour**, reading the latest available hourly updates.

---

## 6. Physical Units

| Parameter | Reported API Unit | UrbanPulse Model Unit | Conversion Required? |
| :--- | :---: | :---: | :---: |
| **PM2.5, PM10, NO2, SO2, NH3, Ozone** | $\mu\text{g}/\text{m}^3$ | $\mu\text{g}/\text{m}^3$ | None (Direct 1:1) |
| **Carbon Monoxide (CO)** | $\text{mg}/\text{m}^3$ | $\text{mg}/\text{m}^3$ | None (Direct 1:1) |
| **Calculated NAQI** | Unitless Index (0–500) | Unitless Index (0–500) | None (CPCB Breakpoints) |
| **Temperature ($T_{\max}$, $T_{\text{ambient}}$)** | $^\circ\text{C}$ | $^\circ\text{C}$ | None (Direct 1:1) |
| **Departure from Normal** | $^\circ\text{C}$ | $^\circ\text{C}$ | None (Direct 1:1) |
| **Rainfall (Hourly / 24h)** | $\text{mm}$ | $\text{mm}$ | None (Direct 1:1) |

---

## 7. Spatial and Station Coverage Across Chennai

### 7.1 AQI Monitoring Coverage (CPCB)
CPCB stations map directly to the 15 GCC zones using the verified point-in-polygon mapping:

| Station Name | GCC Zone ID | GCC Zone Name | Ingestion Status |
| :--- | :---: | :--- | :--- |
| **Gandhi Nagar / Ennore** | Zone 1 | Thiruvottiyur | Monitored (sensor sparse) |
| **Manali (CPCB)** | Zone 2 | Manali | Monitored (Primary) |
| **Manali Village (TNPCB)** | Zone 2 | Manali | Monitored (Secondary) |
| *None* | **Zone 3** | **Madhavaram** | **STRICTLY UNMONITORED** |
| **Kodungaiyur** | Zone 4 | Tondiarpet | Monitored |
| **Royapuram** | Zone 5 | Royapuram | Monitored |
| *None* | **Zone 6** | **Thiru-Vi-Ka Nagar** | **STRICTLY UNMONITORED** |
| *None* | **Zone 7** | **Ambattur** | **STRICTLY UNMONITORED** |
| *None* | **Zone 8** | **Anna Nagar** | **STRICTLY UNMONITORED** |
| *None* | **Zone 9** | **Teynampet** | **STRICTLY UNMONITORED** |
| **Arumbakkam** | Zone 10 | Kodambakkam | Monitored |
| *None* | **Zone 11** | **Valasaravakkam** | **STRICTLY UNMONITORED** |
| **Alandur Bus Depot** | Zone 12 | Alandur | Monitored |
| **Velachery Res. Area** | Zone 13 | Adyar | Monitored |
| **Perungudi** | Zone 14 | Perungudi | Monitored |
| *None* | **Zone 15** | **Sholinganallur** | **STRICTLY UNMONITORED** |

**Strict Rule Enforced:** Zones 3, 6, 7, 8, 9, 11, and 15 are marked with `is_monitored = False` and `observed_aqi = NaN`. No synthetic, regional mean, or forward-filled observations are ever injected.

### 7.2 Weather & Meteorological Coverage (IMD)
* Synoptic observatories: Chennai Meenambakkam (`43279`) and Chennai Nungambakkam (`43278`).
* AWS network: Multiple automated rain gauges across the district.
* **Important Scientific Truth:** IMD temperature and rainfall data represent district/regional meteorological monitoring, **NOT 15 independent zone-level sensors**. The pipeline preserves this exact distinction.

---

## 8. Missingness, Limitations & Preprocessing Compatibility

### 8.1 15-Minute to Hourly Preprocessing Compatibility
* **CPCB Source Data:** Historical training data was 15-minute resolution; the live API provides hourly aggregates (`pollutant_avg`).
* **Rolling Buffer Architecture:** Official CPCB sub-index formulas require 24-hour rolling averages for PM2.5, PM10, NO2, SO2, NH3 (min 16 hours), and 8-hour rolling averages for CO and Ozone (min 6 hours).
* **Solution in `src/live/cpcb_client.py`:** A 24-hour in-memory rolling buffer stores incoming hourly observations per station, continuously evaluating the rolling averages and official CPCB breakpoints.

### 8.2 Rainfall 24-Hour Accumulation Requirement
* **IMD Heavy Rainfall Standard:** $\ge 64.5\text{ mm}$ accumulated in 24 hours.
* **Strict Constraint:** Hourly rainfall (e.g., $15\text{ mm/h}$) is **NEVER** compared directly against $64.5\text{ mm}$.
* **Solution in `src/live/imd_client.py`:** Hourly AWS increments are accumulated across a rolling 24-hour window before passing to the heavy rainfall alert evaluator.

### 8.3 Temperature Climatological Normal Requirement
* **IMD Coastal Heatwave Criterion:**
  $$T_{\max} \ge 37.0^\circ\text{C} \quad \text{AND} \quad \text{Departure from Normal} \ge +4.5^\circ\text{C}$$
* **Strict Constraint:** $37.0^\circ\text{C}$ is never used as a standalone trigger. Climatological normal $T_{\max}$ must never be fabricated.
* **Key Finding:** In the official IMD `cityforecast` API (API #1), IMD explicitly reports `Today_Max_Departure_from_Normal`. This means that with official IMD API credentials, **genuine official departures are directly available from IMD**, completely resolving the climatology limitation without any fabrication. If the departure field is ever missing, the alert engine defaults safely to `UNAVAILABLE_MISSING_CLIMATOLOGY`.

---

## 9. Can Live Data Feed UrbanPulse?

**YES.** The data streams directly into the existing operational pipeline:

```text
Official CPCB (data.gov.in) & IMD (api.imd.gov.in)
                       │
                       ▼
         Live Ingestion & Schema Validation
             (src/live/cpcb_client.py)
             (src/live/imd_client.py)
                       │
                       ▼
       Station-to-Zone Spatial Mapping
    (7 unmonitored zones strictly flagged)
                       │
                       ▼
        FROZEN Zone-Specific HMM Models
   (Temporal filtering & probabilistic forecast)
                       │
                       ▼
           FROZEN Spatial Gaussian MRF
    (Exact solve: Q = Lambda + beta*L_w, b = Lambda*yhat)
                       │
                       ▼
          FROZEN Alert Decision Engine
   (Official thresholds: AQI > 200, Rain 24h >= 64.5mm,
    Heatwave: Tmax >= 37C & Dep >= 4.5C; P cutoffs: 20%/60%)
                       │
                       ▼
          Streamlit Monitoring Dashboard
```

---

## 10. Required Credentials and Access Setup

To transition from feasibility verification to live production execution (Step 14B):

1. **CPCB Access:**
   * Variable Name: `DATA_GOV_IN_API_KEY`.
   * Action Required: User creates a free account at [https://data.gov.in](https://data.gov.in) and copies the personal API key from the user dashboard.
2. **IMD Access:**
   * Variable Name: `IMD_API_KEY`.
   * Action Required: Request institutional API key from IMD Nodal Officer (`sankar.nath@imd.gov.in`) or configure server IP whitelisting.

---

## 11. Is Model Retraining Required?

**NO. ABSOLUTELY NOT.**
* All 26 zone-specific HMM models and all Gaussian MRF models operate on standard physical units (continuous AQI, hourly rainfall in mm, daily maximum temperature in $^\circ\text{C}$).
* The live pipeline outputs vectors of identical mathematical format to those used during research training and validation.
* Frozen parameters ($\beta_{\text{AQI}} = 0.00010$, $\beta_{\text{Rainfall}} = 0.0$, $\beta_{\text{Temperature}} = 0.0$) and model weights remain 100% untouched.

---

## 12. Implemented Code and Artifacts (Step 14A)

All Step 14A code and schema specifications have been created and verified:

1. **Schemas and Formats:**
   * `data/live/cpcb_live_schema.json`: Formal JSON schema for data.gov.in CPCB records.
   * `data/live/sample_cpcb_response.json`: Sample payload matching official CPCB fields and Chennai stations.
   * `data/live/imd_live_schema.json`: Formal JSON schema for IMD endpoints (`current_wx`, `cityforecast`, `aws_data`).
   * `data/live/sample_imd_response.json`: Sample payload matching official IMD weather fields.
2. **Live Ingestion Modules:**
   * `src/live/schemas.py`: Data classes, timestamp validators, unit validators, and physical bounds checks.
   * `src/live/cpcb_client.py`: Official CPCB / data.gov.in client with 24h rolling buffer and CPCB NAQI calculation.
   * `src/live/imd_client.py`: Official IMD client with genuine departure parsing and 24h rainfall accumulation.
   * `src/live/pipeline.py`: Orchestrator connecting live observations to frozen HMM, MRF, and Alert Engine.
3. **Automated Test Suite:**
   * `tests/test_live_ingestion.py`: 9 comprehensive test cases covering timestamp validity, units, station mapping, missing value handling, non-fabrication of unmonitored zones, 24h rainfall accumulation, temperature normal requirement, and pipeline integration.
   * **Full Test Results:** All 43 tests pass (34 historical tests + 9 live ingestion tests).

---

## 13. Recommended Next Implementation Steps (For Step 14B)

When approved to proceed to Step 14B:
1. **Credentials Ingestion:** Store production `DATA_GOV_IN_API_KEY` and `IMD_API_KEY` in environment variables or `.env`.
2. **Hourly Cron / Daemon Runner:** Configure a lightweight background worker to invoke `LiveIngestionPipeline.run_aqi_hourly_cycle` once per hour at minute 05 (allowing CPCB stations to aggregate the previous hour).
3. **Live Data Persistence:** Cache the latest hourly forecast and alert state in `data/live/latest_hourly_forecast.json` and `data/live/latest_hourly_alerts.json`.
4. **Dashboard Real-Time Toggle:** Update Streamlit dashboard to allow users to switch cleanly between:
   * **"Historical Research (Test Period Sep–Dec 2025)"** (displays frozen research results).
   * **"Live Real-Time Operational Monitoring"** (displays live hourly forecasts and alerts).

---

## 14. Verification of Frozen Artifacts

All frozen research artifacts have been inspected and verified unchanged:
* `models/hmm/*` (All 26 zone-specific HMM joblib models remain untouched).
* `models/hmm_mrf/*` (All MRF joblib models remain untouched).
* `data/processed/*` (`train.csv`, `validate.csv`, `test.csv`, `master_table.csv`, `zone_adjacency.csv`, `station_zone_mapping.csv` untouched).
* `src/alerts/alert_engine.py` (Untouched).
* `dashboard/app.py` (Untouched).
* `results/metrics/*`, `results/predictions/*`, `results/figures/*` (Untouched).
* Complete pytest test suite passes with 0 failures (43 passed).
