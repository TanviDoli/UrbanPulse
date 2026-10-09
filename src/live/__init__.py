"""
UrbanPulse Live Data Ingestion Module.
Provides:
  - Original CPCB (data.gov.in) and IMD clients (Step 14A — preserved unchanged).
  - Phase 2: WeatherAPI current-data client, observation preprocessor, and live pipeline.

Note: src.live.live_pipeline is NOT imported here to avoid circular import
when running `python -m src.live.live_pipeline`. Import it directly when needed.
"""

from src.live.schemas import (
    LivePollutantObservation,
    LiveStationAQI,
    LiveWeatherObservation,
    validate_timestamp,
    validate_pollutant_unit,
    validate_physical_bounds,
)
from src.live.cpcb_client import CPCBClient
from src.live.imd_client import IMDClient
from src.live.pipeline import LiveIngestionPipeline
# Phase 2 additions
from src.live.weatherapi_client import WeatherAPIClient, ZONE_COORDINATES
from src.live.obs_preprocessor import preprocess_weatherapi_observations

__all__ = [
    # Original
    'LivePollutantObservation',
    'LiveStationAQI',
    'LiveWeatherObservation',
    'validate_timestamp',
    'validate_pollutant_unit',
    'validate_physical_bounds',
    'CPCBClient',
    'IMDClient',
    'LiveIngestionPipeline',
    # Phase 2
    'WeatherAPIClient',
    'ZONE_COORDINATES',
    'preprocess_weatherapi_observations',
]
