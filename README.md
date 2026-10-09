# UrbanPulse Chennai

Zone-level air-quality, rainfall and temperature early-warning prototype for the 15 Greater
Chennai Corporation (GCC) zones, built on a **Hidden Markov Model (temporal) + Gaussian Markov
Random Field (spatial)** architecture. The research comparison is HMM vs HMM + MRF
(see `reports/final_evaluation_report.md`).

Start with **`reports/final_data_audit.md`**: it lists what data exists, where each number comes
from, and what the system can and cannot predict.

## What the live system reports

| Variable | Live input (provenance) | Zones | Horizons | Model |
|---|---|---|---|---|
| AQI (CPCB NAQI method) | OpenAQ hourly CPCB/TNPCB station data (**measured**) | 7 zones with an AQI HMM; MRF spatial *estimates* for neighbours | 1 h, 3 h, 6 h, 24 h | frozen AQI HMMs + AQI MRF |
| Rainfall | Open-Meteo precipitation (**reanalysis**) | current + last 24 h, all 15 | **no forecast** (disabled: gauge-trained HMMs, input failed validation) | — |
| Temperature | Open-Meteo 2 m temperature → daily means (**reanalysis**) | all 15 | next-day daily mean only (low skill) | frozen daily-mean HMM |

Missing, stale or unsupported items are shown as *unavailable*, never as zero or as "low risk".
The IMD heatwave criterion is **not** evaluated: the temperature model predicts daily mean, not Tmax.

## Layout

```
src/collectors/      data collection: OpenAQ client, Open-Meteo loader/client, live collector
src/preprocessing/   cleaning, CPCB AQI (cpcb_aqi.py), live input preparation (live_inputs.py)
src/forecasting/     live_forecaster.py (frozen HMM + MRF inference)
src/models/          HMM / MRF classes and the original training scripts
src/alerts/          threshold-probability alert engine and alert records
src/live/            live_pipeline.py orchestrator (collect -> preprocess -> infer -> write JSON)
src/audit/           reproducible audits (OpenAQ, Open-Meteo, combined)
dashboard/           Streamlit app (reads data/live/current_predictions.json only)
models/              frozen artifacts (not retrained)
reports/             audits, methodology, evaluation
```

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in the project root (it is git-ignored; never commit it):

```
OPENAQ_API_KEY=...
```

Open-Meteo needs no key.

## Commands

```bash
# Audits (Phase B/C/D). The OpenAQ audit downloads ~2,000 sensor-months and is rate limited (~40–60 min).
python -m src.audit.open_meteo_audit
python -m src.audit.openaq_audit
python -m src.audit.combined_audit          # also writes reports/unit_validation.csv used by the pipeline

# Live pipeline
python -m src.live.live_pipeline --once              # collect + infer once
python -m src.live.live_pipeline --once --no-fetch   # re-run inference on the last collected data
python -m src.live.live_pipeline                     # hourly scheduler

# Dashboard
streamlit run dashboard/app.py

# Tests
python -m pytest
```

## Data sources

- `data/raw/chennai_zones_2025.geojson` — GCC 2025 zone polygons.
- `data/raw/AQI/*.csv` — 9 CAAQMS station files, 2024-01-01 to 2025-12-31 (timestamps are IST
  despite a `+0000` label — see the final audit).
- `data/raw/chennai_daily_temperature_1951_2024.csv`, `data/raw/chennai_daily_rainfall_1991_2023.csv` — IMD daily records.
- `data/external/open_meteo_chennai_hourly_2025_2026.csv` — Open-Meteo hourly reanalysis,
  2025-01-01 to 2026-10-07 23:00 UTC, 15 zone centroids (8 distinct grid cells).
- OpenAQ v3 API — hourly CPCB/TNPCB station data used for the live AQI and audited for 2025-01-01..2026-10-08.
