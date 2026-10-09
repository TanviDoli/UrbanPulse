"""
Official CPCB National Air Quality Index (NAQI) Calculation Engine.
Strictly implements the Central Pollution Control Board (CPCB) methodology:
- Piecewise linear sub-index interpolation for each criteria pollutant.
- 24-hour rolling averages for PM2.5, PM10, NO2, SO2, NH3 (min 16 hours / 66.7% data).
- 8-hour rolling averages for CO, Ozone (min 6 hours / 75% data).
- Minimum criteria: At least 3 pollutants monitored, of which at least one MUST be PM2.5 or PM10.
- Overall AQI = max(sub-indices) when minimum criteria is satisfied.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# CPCB Breakpoints: (B_lo, B_hi, I_lo, I_hi)
CPCB_BREAKPOINTS = {
    'PM2.5': [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ],
    'PM10': [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 600.0, 401, 500),
    ],
    'NO2': [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, 800.0, 401, 500),
    ],
    'SO2': [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, 2400.0, 401, 500),
    ],
    'CO': [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
    'Ozone': [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, 1000.0, 401, 500),
    ],
    'NH3': [
        (0.0, 200.0, 0, 50),
        (200.0, 400.0, 51, 100),
        (400.0, 800.0, 101, 200),
        (800.0, 1200.0, 201, 300),
        (1200.0, 1800.0, 301, 400),
        (1800.0, 2400.0, 401, 500),
    ]
}

def calculate_sub_index(pollutant: str, concentration: float) -> Optional[float]:
    """Calculate single pollutant sub-index via CPCB piecewise linear formula."""
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    
    if pollutant not in CPCB_BREAKPOINTS:
        return np.nan
    
    table = CPCB_BREAKPOINTS[pollutant]
    
    for b_lo, b_hi, i_lo, i_hi in table:
        if b_lo <= concentration <= b_hi:
            sub = i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (concentration - b_lo)
            return float(sub)
    
    # If concentration exceeds the highest breakpoint, linearly extrapolate from top bracket
    b_lo, b_hi, i_lo, i_hi = table[-1]
    if concentration > b_hi:
        sub = i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (concentration - b_lo)
        return float(sub)
        
    return np.nan

def compute_cpcb_aqi_row(sub_indices: Dict[str, float]) -> Tuple[Optional[float], Optional[str], int]:
    """
    Computes overall AQI from a dictionary of pollutant sub-indices.
    Enforces CPCB rule:
    1. At least 3 sub-indices must be present.
    2. At least one must be PM2.5 or PM10.
    Returns: (aqi_value, dominant_pollutant, valid_subindex_count)
    """
    valid = {k: v for k, v in sub_indices.items() if not pd.isna(v)}
    count = len(valid)
    
    has_pm = ('PM2.5' in valid) or ('PM10' in valid)
    
    if count >= 3 and has_pm:
        dominant = max(valid.keys(), key=lambda k: valid[k])
        aqi_val = valid[dominant]
        return aqi_val, dominant, count
    else:
        return np.nan, None, count

def aqi_category(aqi: float) -> str:
    """Classify AQI according to CPCB NAQI categories."""
    if pd.isna(aqi):
        return 'Unavailable'
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'


POLLUTANTS_24H = ['PM2.5', 'PM10', 'NO2', 'SO2', 'NH3']
POLLUTANTS_8H = ['CO', 'Ozone']


def aqi_from_hourly_concentrations(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly CPCB AQI from a gap-preserving HOURLY concentration frame
    (index = consecutive hourly timestamps, columns subset of POLLUTANTS_24H + POLLUTANTS_8H,
    units µg/m³ except CO in mg/m³).

    Identical rules to src/preprocessing/clean_and_build_all.compute_hourly_station_aqi_and_weather,
    which produced the AQI series the HMMs were trained on:
      - 24 h rolling mean (min 16 valid hours) for PM2.5, PM10, NO2, SO2, NH3
      - 8 h rolling mean (min 6 valid hours) for CO, Ozone
      - sub-index via CPCB breakpoints; AQI = max sub-index when >= 3 sub-indices incl. PM2.5 or PM10
    Missing hours must be present as NaN rows (the caller reindexes) so windows are time-correct.
    """
    hourly = hourly.copy()
    for p in POLLUTANTS_24H + POLLUTANTS_8H:
        if p not in hourly.columns:
            hourly[p] = np.nan
    r24 = hourly[POLLUTANTS_24H].rolling(window=24, min_periods=16).mean()
    r8 = hourly[POLLUTANTS_8H].rolling(window=8, min_periods=6).mean()
    rolled = pd.concat([r24, r8], axis=1)
    out = []
    for ts, row in rolled.iterrows():
        subs = {p: (calculate_sub_index(p, row[p]) if pd.notna(row[p]) else np.nan) for p in rolled.columns}
        aqi, dom, cnt = compute_cpcb_aqi_row(subs)
        out.append({'timestamp': ts, 'AQI': round(aqi, 2) if pd.notna(aqi) else np.nan,
                    'dominant_pollutant': dom, 'n_subindices': cnt,
                    'pollutants_used': ','.join(k for k, v in subs.items() if pd.notna(v))})
    return pd.DataFrame(out).set_index('timestamp')
