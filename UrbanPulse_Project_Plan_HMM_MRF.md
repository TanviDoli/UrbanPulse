# UrbanPulse — Detailed Project Plan
## Spatio-Temporal Early Warning System for Chennai using HMM + MRF

---

## 1. Project Title

**UrbanPulse: Spatio-Temporal Early Warning System for Chennai — AQI, Rainfall & Temperature Forecasting using HMM + MRF**

---

## 2. Project Objective

UrbanPulse is a zone-level early-warning system for Chennai that forecasts:

- Air Quality Index (AQI)
- Rainfall
- Temperature

for Chennai's **15 administrative zones**, using a combination of:

1. **Hidden Markov Model (HMM)** for temporal forecasting
2. **Markov Random Field (MRF)** for spatial refinement using neighboring zones

The system should provide forecasts at:

- 1 hour
- 3 hours
- 6 hours
- 24 hours

where the available data resolution supports the horizon.

The final system should also estimate the probability that a forecast will cross a predefined dangerous threshold and generate an early warning.

### Core research comparison

The central experiment is:

> **HMM alone vs HMM + MRF**

This measures whether incorporating spatial relationships between neighboring Chennai zones improves forecasting.

There will be **no persistence baseline and no additional forecasting model** in the final architecture.

---

# 3. Final Architecture

```text
Raw Data
   │
   ├── Chennai 15-zone boundaries
   ├── Air-quality station data
   ├── IMD rainfall data
   └── IMD temperature data
          │
          ▼
   Data Audit & Cleaning
          │
          ▼
   AQI Calculation
   Station → Zone Mapping
   Common Time Resolution
          │
          ▼
      Master Dataset
          │
          ├───────────────┐
          │               │
          ▼               ▼
     Temporal Data    Zone Graph
          │               │
          ▼               │
        HMM               │
          │               │
          │               ▼
          └──────────►  MRF
                         │
                         ▼
                   HMM + MRF
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Forecasts              Threshold
       1h/3h/6h/24h            Probability
              │                     │
              └──────────┬──────────┘
                         ▼
                    Early Alerts
                         │
                         ▼
                  Streamlit Dashboard
```

---

# 4. Datasets

## 4.1 Chennai Zone Boundaries

File:

```text
chennai_zones_2025.geojson
```

Source:

Greater Chennai Corporation (GCC), 2025.

Purpose:

- identify the 15 Chennai zones
- obtain polygon geometry
- construct spatial adjacency
- map monitoring stations to zones
- display results on the dashboard

Expected output:

```text
data/processed/zone_adjacency.csv
```

---

## 4.2 Air Quality Data

Nine station-level CSV files are available for 2024–25:

1. Alandur Bus Depot
2. Arumbakkam
3. Gandhi Nagar / Ennore
4. Kodungaiyur
5. Manali Village
6. Manali
7. Perungudi
8. Royapuram
9. Velachery Residential Area

Source:

OpenCity Chennai Hourly / 15-minute Air Quality Reports.

Important:

The files contain **raw pollutant concentrations**, not a ready-made AQI column.

The project must calculate AQI using the appropriate official CPCB methodology rather than treating PM2.5 or PM10 directly as AQI.

---

## 4.3 Rainfall

File:

```text
chennai_daily_rainfall_1991_2023.csv
```

Source:

IMD data distributed through OpenCity.

Frequency:

Daily.

Important:

Do **not** fabricate hourly rainfall values from daily rainfall observations.

---

## 4.4 Temperature

File:

```text
chennai_daily_temperature_1951_2024.csv
```

Source:

IMD data distributed through OpenCity.

Frequency:

Daily.

Important:

The project must preserve the fact that this is daily data. It must not falsely represent daily observations as independently observed hourly measurements.

---

# 5. Known Data Constraints

Before modeling, the project must explicitly document:

1. AQI station data covers a much shorter period than the historical rainfall and temperature datasets.
2. There are 15 Chennai zones but only 9 air-quality monitoring stations.
3. Not every zone necessarily has a direct monitoring station.
4. AQI observations are higher frequency than the rainfall and temperature datasets.
5. Rainfall and temperature are daily datasets.
6. Missing observations may exist in station-level AQI data.
7. The final common temporal resolution must be selected only after auditing the actual timestamps and coverage.

No missing observation should be fabricated.

---

# 6. Phase 1 — Data Audit

Before writing the forecasting models, inspect every raw dataset.

For each file, report:

- number of rows
- number of columns
- column names
- data types
- date range
- timestamp frequency
- missing-value counts
- duplicate rows
- invalid numeric values
- station names
- units
- suspicious values/outliers

Create:

```text
reports/data_audit.md
```

The audit must be based on the actual files.

Do not assume that every CSV has an identical schema until this is verified.

---

# 7. Phase 2 — AQI Calculation

The air-quality files contain pollutant concentrations.

Calculate AQI using the applicable official CPCB AQI methodology.

The implementation must:

1. identify the pollutants required by the methodology
2. apply the required averaging period for each pollutant
3. calculate pollutant sub-indices
4. determine the overall AQI according to the official method
5. retain the pollutant values used for the calculation
6. document assumptions and unavailable pollutants

Do not calculate:

```text
AQI = average(PM2.5, PM10, ...)
```

Do not treat one raw 15-minute pollutant observation as a complete official AQI observation if the methodology requires a longer averaging period.

The final AQI pipeline must be reproducible.

---

# 8. Phase 3 — Station → Zone Mapping

There are 9 monitoring stations and 15 zones.

Create:

```text
data/processed/station_zone_mapping.csv
```

Required fields:

```text
station_name
zone_id
zone_label
mapping_method
mapping_confidence
notes
```

Mapping should be based on the actual station location and the 2025 GCC zone polygons.

Do not guess a station's zone without documenting the basis.

Zones without direct monitoring stations must remain explicitly identifiable.

---

# 9. Phase 4 — Common Temporal Resolution

The project needs a defensible common temporal resolution.

First inspect:

- AQI timestamp frequency
- rainfall date format
- temperature date format
- gaps
- coverage

The preferred operational resolution may be **hourly** if the data audit shows this is practical.

However, the final resolution must be chosen after the audit.

### Important rule

Do not create fake hourly rainfall or temperature observations.

If daily weather data is aligned with an hourly modeling grid, preserve its provenance and explicitly document the representation.

---

# 10. Phase 5 — Master Dataset

Create a clean master dataset:

```text
data/processed/master_table.csv
```

Recommended conceptual schema:

```text
timestamp
zone_id
zone_label
AQI
rainfall
temperature
aqi_observed
rainfall_observed
temperature_observed
station_name
```

Additional fields may be added if justified by the actual data.

The dataset must not contain fabricated measurements.

---

# 11. Phase 6 — Spatial Graph

The 15 zones form a spatial graph.

### Nodes

Each node represents one Chennai zone.

```text
Node = Chennai zone
```

### Edges

An edge connects two zones that are spatial neighbors.

Construct adjacency from the actual zone polygons.

Prefer shared-boundary relationships rather than arbitrary geographic distance.

Create:

```text
data/processed/zone_adjacency.csv
```

Suggested schema:

```text
zone_id_1
zone_id_2
shared_boundary_length
adjacent
```

The exact edge-weight definition must be documented.

---

# 12. Phase 7 — Chronological Data Split

Never use a random train/test split.

Use chronological splitting so future observations cannot influence past predictions.

Recommended initial structure:

```text
Training     ≈ 70%
Validation   ≈ 15%
Testing      ≈ 15%
```

The exact dates must be determined from the actual usable modeling period after preprocessing.

Create:

```text
data/processed/train.csv
data/processed/validate.csv
data/processed/test.csv
```

The split dates must be recorded in:

```text
reports/data_split.md
```

### Leakage rule

No information from validation or test periods may be used to fit:

- HMM parameters
- MRF parameters
- scalers
- imputation parameters
- feature transformations
- threshold calibration

---

# 13. Final Model 1 — Hidden Markov Model

The HMM represents the temporal component.

For each variable and applicable zone:

```text
Observed measurements
        ↓
      HMM
        ↓
Latent temporal state
        ↓
Future forecast distribution
```

The hidden state represents an underlying temporal regime such as:

- low
- normal
- elevated
- extreme

The exact number of states must be selected using the training data and documented.

The model should learn:

- initial state probabilities
- state-transition probabilities
- emission distributions

The implementation must be a genuine HMM.

It must not simply be a moving average with HMM terminology.

---

# 14. Forecast Horizons

Evaluate the HMM at:

- 1 hour
- 3 hours
- 6 hours
- 24 hours

Only report horizons that are supported by the actual temporal resolution and data coverage.

For each prediction, retain:

```text
timestamp
zone_id
variable
forecast_horizon
prediction
observed_value
```

---

# 15. Final Model 2 — Markov Random Field

The MRF represents spatial dependence among neighboring zones.

The MRF must operate on the Chennai zone graph.

Conceptually:

```text
Zone A ─── Zone B ─── Zone C
   │         │
Zone D ─── Zone E
```

Neighboring zones can influence the estimated state of a target zone.

The MRF should use a documented probabilistic formulation, such as an energy/potential formulation:

```text
E(X) =
    unary terms
  + pairwise spatial terms
```

where:

- unary terms come from the HMM-derived state/forecast evidence
- pairwise terms represent spatial consistency between neighboring zones

The MRF must be a real probabilistic graphical model.

### Forbidden

Do not implement the MRF as:

```text
prediction = own_prediction + 10%
```

or:

```text
prediction = average(neighbor_predictions)
```

or any arbitrary weighted neighbor average presented as an MRF.

---

# 16. Final Combined Model — HMM + MRF

The final model combines:

```text
Temporal information
        │
        ▼
       HMM
        │
        ▼
HMM state / forecast evidence
        │
        ▼
       MRF
        │
        ▼
Spatially refined forecast
```

The exact mathematical formulation must be documented in:

```text
reports/model_methodology.md
```

The combined model must use only information available at prediction time.

---

# 17. Main Research Experiment

The key experiment is:

## HMM vs HMM + MRF

For every supported:

- variable
- zone
- forecast horizon

calculate evaluation metrics for:

### Model A

```text
HMM
```

### Model B

```text
HMM + MRF
```

The comparison should answer:

> Does spatial information from neighboring Chennai zones improve temporal forecasting?

Do not claim that MRF improves performance unless the held-out test results demonstrate it.

---

# 18. Evaluation Metrics

For continuous forecasts calculate:

### MAE

```text
MAE = mean(|y - ŷ|)
```

### RMSE

```text
RMSE = sqrt(mean((y - ŷ)^2))
```

Report metrics separately for:

- AQI
- rainfall
- temperature
- 1h
- 3h
- 6h
- 24h
- HMM
- HMM + MRF

Do not combine variables with different units into one unexplained score.

Create:

```text
results/metrics/
```

---

# 19. Threshold / Early-Warning Layer

After forecasting, convert forecast distributions into threshold-crossing probabilities where statistically justified.

The alert layer is separate from the forecasting model.

## AQI

Use the applicable CPCB AQI category/threshold definition.

The project specifically targets an early warning around the **Poor AQI category (AQI > 200)**.

The exact official definition and source must be recorded in the project documentation.

## Rainfall

Use the applicable IMD rainfall classification.

The project should use the official **heavy rainfall threshold of 64.5 mm/day** where the daily rainfall definition is applicable.

Do not invent an hourly equivalent.

## Temperature

Use the applicable official IMD heatwave definition for the relevant Chennai/Tamil Nadu/coastal context.

The exact operational threshold must be sourced, documented, and implemented consistently.

Do not invent a temperature danger threshold.

---

# 20. Probability of Threshold Crossing

The system should estimate:

```text
P(future value > threshold | current observations)
```

or the appropriate two-sided/interval event for the chosen threshold definition.

Do not hard-code arbitrary probabilities.

Example of the desired output:

```text
Zone: X
Variable: AQI
Forecast horizon: 3h
Predicted AQI: ...
Threshold: ...
Probability of crossing: ...
Alert: ...
```

The probability must come from the model's predictive distribution or a documented statistically valid calibration procedure.

---

# 21. Alert Levels

The dashboard may display states such as:

```text
NORMAL
WATCH
WARNING
```

The exact probability boundaries must be determined from the model/calibration methodology and documented.

Do not choose arbitrary percentages merely to make the dashboard look convincing.

---

# 22. Dashboard

Build the final dashboard using Streamlit.

The dashboard should contain:

## A. Chennai Zone Map

Display the 15 zones using the GCC GeoJSON.

For each zone show:

- current/latest status
- forecast value
- threshold
- threshold-crossing probability
- alert status

## B. Zone Selection

Allow the user to select a zone.

## C. Forecast Panel

Show:

- AQI forecast
- rainfall forecast
- temperature forecast
- 1h / 3h / 6h / 24h horizons where supported

## D. HMM vs HMM + MRF

Display forecast comparison.

## E. Historical Trends

Show recent observed values and forecast values.

## F. Alert Panel

Display active early warnings.

---

# 23. Recommended Folder Structure

```text
UrbanPulse/
│
├── data/
│   ├── raw/
│   │   ├── chennai_zones_2025.geojson
│   │   ├── air_quality/
│   │   ├── chennai_daily_rainfall_1991_2023.csv
│   │   └── chennai_daily_temperature_1951_2024.csv
│   │
│   └── processed/
│       ├── master_table.csv
│       ├── train.csv
│       ├── validate.csv
│       ├── test.csv
│       ├── station_zone_mapping.csv
│       └── zone_adjacency.csv
│
├── src/
│   ├── preprocessing/
│   ├── graph/
│   ├── models/
│   │   ├── hmm_model.py
│   │   ├── mrf_model.py
│   │   └── alert_engine.py
│   ├── forecasting/
│   └── evaluation/
│
├── models/
│   ├── hmm/
│   └── hmm_mrf/
│
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── data_audit.md
│   ├── data_split.md
│   └── model_methodology.md
│
├── configs/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

# 24. Suggested Technical Stack

Python:

- pandas
- numpy
- scipy
- scikit-learn
- hmmlearn or an equivalent validated HMM implementation
- geopandas
- shapely
- networkx
- matplotlib
- seaborn
- plotly
- streamlit

Additional probabilistic-modeling libraries may be used if required for a genuine MRF implementation.

Do not add unnecessary machine-learning models.

---

# 25. Reproducibility Requirements

The project must record:

- dataset filenames
- data sources
- data coverage
- preprocessing rules
- AQI methodology
- station-zone mapping
- graph construction
- temporal resolution
- train/validation/test dates
- HMM state count
- HMM initialization
- MRF formulation
- MRF parameters
- alert thresholds
- probability calibration method
- random seeds where applicable

---

# 26. Required Figures

At minimum:

1. Chennai 15-zone map
2. Zone adjacency graph
3. Data coverage timeline
4. Example AQI time series
5. Example rainfall time series
6. Example temperature time series
7. HMM forecast
8. HMM + MRF forecast
9. HMM vs HMM + MRF metric comparison
10. Threshold-crossing probability visualization
11. Example early-warning map

---

# 27. Required Tables

At minimum:

1. Dataset summary
2. Data quality summary
3. Station → zone mapping
4. Zone adjacency
5. Train/validation/test date ranges
6. HMM configuration
7. MRF configuration
8. HMM evaluation metrics
9. HMM + MRF evaluation metrics
10. Alert thresholds and official sources

---

# 28. Testing Requirements

Create tests for:

- timestamp parsing
- duplicate detection
- missing-value handling
- AQI calculation
- station-zone mapping
- zone adjacency
- chronological splitting
- HMM fitting
- MRF fitting/inference
- forecast horizon generation
- threshold detection
- probability calculation
- dashboard data loading

---

# 29. Hard Rules

These rules must never be violated.

### Rule 1
Do not fabricate observations.

### Rule 2
Do not fabricate missing hourly rainfall or temperature measurements.

### Rule 3
Do not use future data during training.

### Rule 4
Do not randomly split time-series data.

### Rule 5
Do not claim the MRF improves forecasting before evaluating the held-out test set.

### Rule 6
Do not call a neighbor average an MRF.

### Rule 7
Do not use arbitrary alert thresholds.

### Rule 8
Do not use arbitrary probability percentages.

### Rule 9
Do not hide the fact that there are 9 AQI monitoring stations and 15 zones.

### Rule 10
Document all assumptions.

### Rule 11
Use HMM + MRF as the final architecture.

### Rule 12
The only model comparison required is:

```text
HMM
vs
HMM + MRF
```

Do not add persistence, LSTM, XGBoost, random forest, GRU, CNN, Transformer, or other forecasting models unless explicitly requested later.

---

# 30. Final Deliverables

The completed project should contain:

### Data

```text
master_table.csv
train.csv
validate.csv
test.csv
station_zone_mapping.csv
zone_adjacency.csv
```

### Models

```text
HMM
HMM + MRF
```

### Evaluation

```text
MAE
RMSE
HMM vs HMM + MRF
```

### Early Warning

```text
threshold
forecast probability
alert status
```

### Dashboard

```text
15-zone Chennai map
forecast panels
historical trends
HMM vs HMM + MRF
early-warning alerts
```

### Documentation

```text
README.md
data_audit.md
data_split.md
model_methodology.md
```

---

# 31. Implementation Order

Implement in this exact order:

```text
1. Inspect all raw files
2. Produce data audit
3. Clean and standardize AQI station files
4. Implement official AQI calculation
5. Map stations to zones
6. Determine common operational resolution
7. Merge usable datasets
8. Construct 15-zone adjacency graph
9. Create chronological train/validation/test datasets
10. Implement HMM
11. Validate HMM
12. Implement genuine MRF
13. Integrate HMM + MRF
14. Generate forecasts
15. Evaluate HMM
16. Evaluate HMM + MRF
17. Compare results
18. Implement threshold probabilities
19. Implement early-warning engine
20. Build Streamlit dashboard
21. Add tests
22. Complete documentation
```

Do not skip the data audit.

Do not begin by training the model.

---

# 32. Definition of a Successful Project

UrbanPulse is considered complete when:

1. All raw datasets have been audited.
2. The 15 Chennai zones are correctly represented.
3. AQI is calculated using a documented official methodology.
4. Station-to-zone mapping is documented.
5. The spatial adjacency graph is generated from zone geometry.
6. The temporal resolution is justified from the actual data.
7. Train/validation/test periods are chronological.
8. HMM forecasting works on held-out data.
9. MRF performs genuine spatial probabilistic refinement.
10. HMM + MRF produces valid forecasts.
11. HMM and HMM + MRF are evaluated on the same held-out test periods.
12. Threshold-crossing probabilities are statistically justified.
13. Early warnings are generated from those probabilities.
14. The dashboard displays the 15-zone spatial results.
15. No fabricated data or unsupported claims are used.

---

# 33. Initial Antigravity Prompt

Copy the following prompt into Antigravity after placing all raw datasets inside the project.

You are the lead ML/data-engineering developer for my project **UrbanPulse**.

I am building:

**UrbanPulse: Spatio-Temporal Early Warning System for Chennai — AQI, Rainfall & Temperature Forecasting using HMM + MRF**

The FINAL architecture is ONLY:

**HMM + MRF**

The research comparison is ONLY:

**HMM vs HMM + MRF**

Do NOT add persistence, LSTM, GRU, XGBoost, Random Forest, CNN, Transformer, or any other forecasting model unless I explicitly ask later.

## PROJECT GOAL

UrbanPulse should forecast, for Chennai's 15 administrative zones:

- AQI
- rainfall
- temperature

at supported horizons:

- 1 hour
- 3 hours
- 6 hours
- 24 hours

The HMM models temporal behavior.

The MRF models spatial dependence between neighboring Chennai zones.

The final system should also estimate threshold-crossing probability and produce an early-warning alert.

---

# AVAILABLE RAW DATA

Place/expect these files under `data/raw/`:

### Zone boundaries

`chennai_zones_2025.geojson`

Official Greater Chennai Corporation 2025 zone polygons.

### AQI station data

Nine 2024–25 station CSVs:

- gandhi-nagar-2024-25.csv
- arumbakkam-2024-25.csv
- manali-village-2024-25.csv
- alandur-2024-25.csv
- manali-2024-25.csv
- velachery-2024-25.csv
- perungudi-2024-25.csv
- royapuram-2024-25.csv
- kodungaiyur-2024-25.csv

These contain raw pollutant concentrations. They do NOT necessarily contain a ready-made AQI value.

### Rainfall

`chennai_daily_rainfall_1991_2023.csv`

Daily IMD rainfall data.

### Temperature

`chennai_daily_temperature_1951_2024.csv`

Daily IMD temperature data.

---

# CRITICAL RULES

## 1. AUDIT FIRST

Do NOT immediately train models.

First inspect every raw file and produce:

`reports/data_audit.md`

For each file report:

- rows
- columns
- schema
- data types
- date range
- timestamp/date frequency
- missing values
- duplicates
- invalid values
- station names
- units
- suspicious values/outliers

Do not assume schemas are identical until verified.

## 2. DO NOT FABRICATE DATA

Never invent:

- observations
- timestamps
- rainfall
- temperature
- AQI
- missing sensor values

Do not convert daily rainfall into fake hourly rainfall.

Do not pretend daily temperature measurements are independently observed hourly values.

## 3. AQI MUST BE CALCULATED CORRECTLY

The AQI station files contain pollutant concentrations.

Implement the applicable official CPCB AQI methodology.

Do NOT do:

`AQI = average(all pollutant values)`

Do NOT treat one raw 15-minute row as an official AQI observation when the methodology requires averaging periods.

Document the exact methodology and assumptions.

## 4. 15 ZONES, 9 AQI STATIONS

There are 15 Chennai zones but only 9 monitoring stations.

Create:

`data/processed/station_zone_mapping.csv`

with:

- station_name
- zone_id
- zone_label
- mapping_method
- mapping_confidence
- notes

Use the actual station location and GCC polygons where possible.

Do not silently assign stations to zones by guesswork.

Zones without direct stations must remain identifiable.

## 5. TEMPORAL RESOLUTION

Inspect the real data before selecting the common operational resolution.

Hourly may be appropriate if the audit supports it.

Do not fabricate hourly rainfall or temperature.

Document how daily weather data is aligned with the operational timeline.

## 6. CHRONOLOGICAL SPLIT

Create:

- `data/processed/train.csv`
- `data/processed/validate.csv`
- `data/processed/test.csv`

Use chronological splitting.

Do NOT randomly split the time series.

Do NOT use future information during preprocessing/model fitting.

Record split dates in:

`reports/data_split.md`

---

# SPATIAL GRAPH

Use:

`chennai_zones_2025.geojson`

to create the 15-zone spatial graph.

Two zones should be connected when they are actual spatial neighbors.

Prefer shared-boundary relationships.

Create:

`data/processed/zone_adjacency.csv`

with:

- zone_id_1
- zone_id_2
- shared_boundary_length
- adjacent

Document the graph construction.

---

# MODEL 1: HMM

Implement a genuine Hidden Markov Model.

The HMM represents temporal dynamics.

It should learn from training data:

- initial state probabilities
- transition probabilities
- emission distributions

Use a documented number of hidden states.

The hidden state represents a latent temporal regime.

Evaluate supported horizons:

- 1h
- 3h
- 6h
- 24h

Save predictions in:

`results/predictions/`

and model artifacts under:

`models/hmm/`

---

# MODEL 2: MRF

Implement a genuine Markov Random Field over the 15-zone graph.

The MRF must incorporate:

- HMM-derived evidence/unary terms
- spatial pairwise relationships between neighboring zones

A valid conceptual formulation is:

`Energy = unary terms + pairwise spatial terms`

The exact mathematical formulation must be documented in:

`reports/model_methodology.md`

DO NOT implement MRF as:

- neighbor average
- arbitrary weighted average
- adding 10%
- manually adjusting forecasts
- heuristic smoothing

If a library is used, explain exactly how the MRF corresponds to the mathematical model.

---

# FINAL MODEL: HMM + MRF

The combined pipeline should be:

Raw observations
→ temporal preprocessing
→ HMM
→ HMM state/forecast evidence
→ MRF spatial inference
→ final spatially refined forecast

The combined model must use only information available at prediction time.

Save artifacts under:

`models/hmm_mrf/`

---

# MAIN EXPERIMENT

Evaluate:

### Model A
HMM

### Model B
HMM + MRF

Use the SAME held-out test periods.

For every supported:

- variable
- zone
- horizon

calculate:

- MAE
- RMSE

Do not claim MRF improves performance unless the actual test results demonstrate it.

---

# EARLY WARNING

After forecasting, calculate threshold-crossing probabilities from valid predictive distributions or a documented calibration method.

Do NOT hard-code fake probabilities.

Targets include:

### AQI
Use the applicable CPCB definition for the Poor category / AQI > 200.

### Rainfall
Use the applicable IMD rainfall classification, including 64.5 mm/day for heavy rainfall where the daily definition applies.

Do NOT invent an hourly equivalent.

### Temperature
Use the applicable official IMD heatwave definition for the relevant Chennai/Tamil Nadu/coastal context.

Do NOT invent a temperature danger threshold.

Document every threshold and source.

---

# PROJECT STRUCTURE

Create/maintain:

```text
UrbanPulse/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── preprocessing/
│   ├── graph/
│   ├── models/
│   │   ├── hmm_model.py
│   │   ├── mrf_model.py
│   │   └── alert_engine.py
│   ├── forecasting/
│   └── evaluation/
├── models/
│   ├── hmm/
│   └── hmm_mrf/
├── results/
│   ├── predictions/
│   ├── metrics/
│   └── figures/
├── dashboard/
│   └── app.py
├── reports/
├── configs/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

---

# IMPLEMENTATION ORDER

Follow this exact order:

1. Audit all raw data.
2. Create `reports/data_audit.md`.
3. Clean/standardize AQI station files.
4. Implement official AQI calculation.
5. Map stations to zones.
6. Determine common operational resolution.
7. Create the master dataset.
8. Build the 15-zone adjacency graph.
9. Create chronological train/validation/test datasets.
10. Implement HMM.
11. Validate HMM.
12. Implement genuine MRF.
13. Integrate HMM + MRF.
14. Generate forecasts.
15. Evaluate HMM.
16. Evaluate HMM + MRF.
17. Compare them.
18. Implement threshold probabilities.
19. Implement early-warning engine.
20. Build Streamlit dashboard.
21. Add tests.
22. Complete documentation.

Do not skip ahead to model training before the data audit.

---

# IMPORTANT DEVELOPMENT BEHAVIOR

When you encounter an ambiguity:

1. Inspect the actual data.
2. Check the project documentation.
3. Choose the most defensible approach.
4. Document the assumption.
5. Do not silently fabricate a solution.

If a methodological decision could materially affect the scientific validity of the project, stop and explain the issue before implementing it.

Keep the implementation reproducible and reasonably simple.

Do not create unnecessary files or models.

The final project must be defensible in an academic presentation.

Start by auditing the raw files and report exactly what you find before implementing the HMM or MRF.
