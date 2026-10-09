"""
Official IMD Live Ingestion Client for UrbanPulse (Step 14A).
Interacts with official India Meteorological Department API (api.imd.gov.in):
- Current Weather (api/v1/current_wx)
- City Weather Forecast (api/v1/cityforecast)
- Automatic Weather Station / Automatic Rain Gauge (api/v1/aws_data)

Strict Non-Negotiable Rules:
- Normal Tmax is NEVER fabricated. If missing from official feed, departure remains None.
- 24-hour accumulated rainfall is strictly computed/preserved for heavy rainfall alerts (>= 64.5 mm).
- Hourly rainfall increments are NEVER compared directly against 64.5 mm.
- Unmonitored zones for rainfall are marked with is_monitored=False without synthetic observations.
"""

import os
import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.live.schemas import (
    LiveWeatherObservation,
    validate_timestamp,
    validate_physical_bounds,
    UNMONITORED_RAINFALL_ZONES
)

IMD_BASE_URL = "https://api.imd.gov.in/api/v1"


class IMDClient:
    """Client for fetching and parsing official IMD live weather data."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("IMD_API_KEY")
        # Rolling 24-hour rainfall buffer by station: {station_id: pd.Series}
        self.rainfall_buffer_24h: Dict[str, pd.Series] = {}

    def fetch_current_weather(self, station_id: str = "43279") -> Dict[str, Any]:
        """
        Fetches current synoptic weather from api.imd.gov.in/api/v1/current_wx.
        Requires official IMD_API_KEY and registration.
        """
        if not self.api_key:
            raise PermissionError(
                "IMD_API_KEY is not configured. "
                "Official registration with IMD (Nodal Officer, Dr. Sankar Nath, sankar.nath@imd.gov.in) "
                "or IP whitelisting is required."
            )

        url = f"{IMD_BASE_URL}/current_wx?id={station_id}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "UrbanPulse-Weather-LiveClient/1.0",
                "X-Api-Key": self.api_key
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as res:
                return json.loads(res.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            raise ConnectionError(f"IMD API error ({e.code} {e.reason}): {err_body}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to IMD API: {e}")

    def fetch_city_forecast(self, station_id: str = "43279") -> Dict[str, Any]:
        """
        Fetches 7-day city weather bulletin containing official climatological departures.
        URL: api.imd.gov.in/api/v1/cityforecast
        """
        if not self.api_key:
            raise PermissionError("IMD_API_KEY is not configured.")

        url = f"{IMD_BASE_URL}/cityforecast?id={station_id}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "UrbanPulse-Weather-LiveClient/1.0",
                "X-Api-Key": self.api_key
            }
        )
        with urllib.request.urlopen(req, timeout=15) as res:
            return json.loads(res.read().decode('utf-8'))

    def parse_current_weather(self, payload: Dict[str, Any]) -> List[LiveWeatherObservation]:
        """Parses current_wx records with physical bounds validation."""
        records = payload.get('data', [])
        observations = []

        for rec in records:
            st_id = str(rec.get('Station Id', ''))
            st_name = str(rec.get('Station', ''))
            date_str = str(rec.get('Date of Observation', ''))
            time_str = str(rec.get('Time of Observation UTC', '00:00:00'))

            try:
                full_ts_str = f"{date_str} {time_str}"
                ts = pd.to_datetime(full_ts_str, utc=True)
            except Exception:
                continue

            temp = validate_physical_bounds('temperature', rec.get('Temperature deg C'))
            rh = validate_physical_bounds('humidity', rec.get('Humidity %'))
            rf_24h = validate_physical_bounds('rainfall', rec.get('Last 24 hrs Rainfall mm'))

            obs = LiveWeatherObservation(
                station_id=st_id,
                station_name=st_name,
                timestamp=ts,
                temperature=temp,
                humidity=rh,
                rainfall_24h_accum=rf_24h,
                is_monitored=True
            )
            observations.append(obs)

        return observations

    def parse_city_forecast(self, payload: Dict[str, Any]) -> List[LiveWeatherObservation]:
        """
        Parses official city forecast bulletin for maximum temperature and genuine departure.
        Strict Non-Negotiable Rule:
        - If Today_Max_Departure_from_Normal is provided by IMD, extracts it.
        - If missing or unparseable, departure remains None. NEVER fabricated.
        """
        records = payload.get('data', [])
        observations = []

        for rec in records:
            st_code = str(rec.get('Station_Code', ''))
            st_name = str(rec.get('Station_Name', ''))
            date_str = str(rec.get('Date', ''))

            try:
                ts = pd.to_datetime(date_str, utc=True)
            except Exception:
                continue

            max_t = validate_physical_bounds('temperature', rec.get('Today_Max_temp'))
            raw_dep = rec.get('Today_Max_Departure_from_Normal')

            dep = None
            if raw_dep is not None and pd.notna(raw_dep):
                try:
                    dep_str = str(raw_dep).replace('+', '').strip()
                    dep = float(dep_str)
                except ValueError:
                    dep = None

            # Normal Tmax can be derived only if both max_t and dep are genuinely given:
            # dep = max_t - normal_tmax => normal_tmax = max_t - dep
            norm_tmax = None
            if max_t is not None and pd.notna(max_t) and dep is not None:
                norm_tmax = float(max_t - dep)

            obs = LiveWeatherObservation(
                station_id=st_code,
                station_name=st_name,
                timestamp=ts,
                max_temperature=max_t,
                departure_from_normal=dep,
                normal_tmax=norm_tmax,
                rainfall_24h_accum=validate_physical_bounds('rainfall', rec.get('Past_24_hrs_Rainfall')),
                is_monitored=True
            )
            observations.append(obs)

        return observations

    def parse_aws_data(self, payload: Dict[str, Any]) -> List[LiveWeatherObservation]:
        """Parses AWS/ARG automatic weather station records."""
        records = payload.get('data', [])
        observations = []

        for rec in records:
            st_id = str(rec.get('ID', rec.get('CALL_SIGN', '')))
            st_name = str(rec.get('STATION', ''))
            d_str = str(rec.get('DATE', ''))
            t_str = str(rec.get('TIME', '00:00:00'))

            try:
                ts = pd.to_datetime(f"{d_str} {t_str}", utc=True)
            except Exception:
                continue

            temp = validate_physical_bounds('temperature', rec.get('CURR_TEMP'))
            rh = validate_physical_bounds('humidity', rec.get('RH'))
            rf = validate_physical_bounds('rainfall', rec.get('RAIN_FALL'))

            obs = LiveWeatherObservation(
                station_id=st_id,
                station_name=st_name,
                timestamp=ts,
                temperature=temp,
                humidity=rh,
                rainfall_hourly=rf,
                is_monitored=True
            )
            observations.append(obs)

        return observations

    def compute_24h_rainfall_accumulation(
        self,
        station_id: str,
        hourly_rainfall: float,
        timestamp: pd.Timestamp
    ) -> float:
        """
        Maintains rolling 24-hour buffer to calculate exact accumulated rainfall.
        Ensures that 24h accumulation is computed correctly before evaluating
        heavy rainfall alert threshold (64.5 mm).
        """
        if station_id not in self.rainfall_buffer_24h:
            self.rainfall_buffer_24h[station_id] = pd.Series(dtype=float)

        self.rainfall_buffer_24h[station_id].loc[timestamp] = hourly_rainfall

        # Retain only past 24 hours
        cutoff = timestamp - pd.Timedelta(hours=24)
        s = self.rainfall_buffer_24h[station_id]
        s = s[s.index >= cutoff]
        self.rainfall_buffer_24h[station_id] = s

        return float(s.sum())
