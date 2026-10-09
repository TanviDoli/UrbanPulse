"""
Data schemas and strict validators for UrbanPulse Live Ingestion (Step 14A).
Ensures data integrity, explicit non-fabrication of unmonitored zones,
proper units, timestamp validity, and physical bounds checks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import datetime
import numpy as np
import pandas as pd

# Expected official units
EXPECTED_UNITS = {
    'PM2.5': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'PM10': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'NO2': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'NH3': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'SO2': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'CO': ['mg/m3', 'mg/m³', 'mg/m^3'],
    'OZONE': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'Ozone': ['ug/m3', 'µg/m³', 'ug/m^3'],
    'temperature': ['deg C', '°C', 'C'],
    'rainfall': ['mm']
}

# Physical bounds for validation
PHYSICAL_BOUNDS = {
    'PM2.5': (0.0, 1000.0),
    'PM10': (0.0, 1500.0),
    'NO2': (0.0, 1000.0),
    'NH3': (0.0, 2500.0),
    'SO2': (0.0, 2500.0),
    'CO': (0.0, 100.0),
    'OZONE': (0.0, 1200.0),
    'Ozone': (0.0, 1200.0),
    'temperature': (-5.0, 60.0),
    'humidity': (0.0, 100.0),
    'rainfall': (0.0, 500.0)
}

# The 7 GCC zones that are explicitly unmonitored for AQI
UNMONITORED_AQI_ZONES = {3, 6, 7, 8, 9, 11, 15}

# The 11 GCC zones that do not have dedicated tipping-bucket rain gauges
UNMONITORED_RAINFALL_ZONES = {1, 2, 3, 6, 7, 8, 9, 11, 12, 13, 15}


def validate_timestamp(ts_input: Any) -> pd.Timestamp:
    """
    Validates and normalizes timestamp to UTC pd.Timestamp.
    Rejects malformed strings, isolated year numbers (e.g. '2025' corruption artifact),
    and non-parseable inputs.
    """
    if ts_input is None or pd.isna(ts_input):
        raise ValueError("Timestamp cannot be null or NaN.")
    
    s = str(ts_input).strip()
    if s == '2025' or len(s) < 8:
        raise ValueError(f"Corrupt or incomplete timestamp rejected: '{ts_input}'")
        
    try:
        ts = pd.to_datetime(s, dayfirst=True)
        if ts.tzinfo is None:
            # Assume local IST (+05:30) if unlocalized and convert to UTC, or localize as UTC
            # Standard CPCB last_update is given in IST without timezone suffix
            ts = ts.tz_localize('Asia/Kolkata').tz_convert('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return ts
    except Exception as e:
        raise ValueError(f"Invalid timestamp format '{ts_input}': {e}")


def validate_pollutant_unit(pollutant: str, unit: Optional[str]) -> bool:
    """Validates that the reported unit matches official standards."""
    if not unit:
        return True # Default to official standard if missing in header
    clean_unit = str(unit).strip().lower().replace(' ', '')
    expected = [u.lower().replace(' ', '') for u in EXPECTED_UNITS.get(pollutant, [])]
    return clean_unit in expected or len(expected) == 0


def validate_physical_bounds(param: str, value: Optional[float]) -> Optional[float]:
    """
    Validates that a physical measurement is within realistic atmospheric bounds.
    Returns float or np.nan if invalid/out-of-bounds.
    """
    if value is None or pd.isna(value):
        return np.nan
    val = float(value)
    if param in PHYSICAL_BOUNDS:
        min_v, max_v = PHYSICAL_BOUNDS[param]
        if val < min_v or val > max_v:
            return np.nan
    return val


@dataclass
class LivePollutantObservation:
    """Single pollutant measurement from official CAAQMS station."""
    station_name: str
    pollutant_id: str
    value: float
    timestamp: pd.Timestamp
    unit: str = "ug/m3"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class LiveStationAQI:
    """Aggregated AQI calculation for a specific monitoring station / zone."""
    station_name: str
    zone_id: int
    zone_label: str
    timestamp: pd.Timestamp
    sub_indices: Dict[str, float] = field(default_factory=dict)
    aqi_value: Optional[float] = None
    dominant_pollutant: Optional[str] = None
    is_monitored: bool = True
    valid_subindex_count: int = 0


@dataclass
class LiveWeatherObservation:
    """Official meteorological observation from IMD."""
    station_id: str
    station_name: str
    timestamp: pd.Timestamp
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall_hourly: Optional[float] = None
    rainfall_24h_accum: Optional[float] = None
    max_temperature: Optional[float] = None
    departure_from_normal: Optional[float] = None
    normal_tmax: Optional[float] = None
    is_monitored: bool = True
