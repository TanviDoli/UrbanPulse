"""
Official CPCB / data.gov.in Live Ingestion Client for UrbanPulse (Step 14A).
Connects to official OGD Platform resource: 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.
Enforces station-zone mapping, strict non-fabrication for unmonitored zones,
rolling window calculations for CPCB AQI, and input validation.
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
    LivePollutantObservation,
    LiveStationAQI,
    validate_timestamp,
    validate_pollutant_unit,
    validate_physical_bounds,
    UNMONITORED_AQI_ZONES
)
from src.preprocessing.cpcb_aqi import calculate_sub_index, compute_cpcb_aqi_row

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DATA_GOV_IN_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
DATA_GOV_IN_BASE_URL = f"https://api.data.gov.in/resource/{DATA_GOV_IN_RESOURCE_ID}"


class CPCBClient:
    """Client for fetching and processing live CPCB CAAQMS air quality data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        mapping_csv: str = "data/processed/station_zone_mapping.csv"
    ):
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY")
        self.mapping_csv = mapping_csv
        self.station_mapping_df = self._load_station_mapping()
        # Rolling historical buffer: {station_name: pd.DataFrame with index=timestamp, cols=pollutants}
        self.rolling_buffer: Dict[str, pd.DataFrame] = {}

    def _load_station_mapping(self) -> pd.DataFrame:
        """Loads verified station-to-zone mapping from processed CSV."""
        if os.path.exists(self.mapping_csv):
            df = pd.read_csv(self.mapping_csv)
            return df
        raise FileNotFoundError(f"Required mapping file not found: {self.mapping_csv}")

    def map_station_to_zone(self, station_name: str) -> Optional[Tuple[int, str]]:
        """
        Maps a station name from data.gov.in to (zone_id, zone_label).
        Returns None if station is not within GCC boundaries.
        """
        st_clean = station_name.strip().lower()
        for _, row in self.station_mapping_df.iterrows():
            official_name = str(row['station_name']).strip().lower()
            if official_name != 'none' and official_name != 'nan':
                # Check for direct or substring match
                prefix = official_name.split(',')[0].strip()
                if prefix in st_clean or st_clean in official_name:
                    return int(row['zone_id']), str(row['zone_label'])
        return None

    def fetch_live_data(self, city: str = "Chennai", limit: int = 100) -> Dict[str, Any]:
        """
        Fetches live real-time AQI records from data.gov.in API.
        Requires valid DATA_GOV_IN_API_KEY.
        """
        if not self.api_key:
            raise PermissionError(
                "DATA_GOV_IN_API_KEY is not configured. "
                "Registration on https://data.gov.in is required to obtain an official API key."
            )

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "filters[city]": city
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{DATA_GOV_IN_BASE_URL}?{query_string}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "UrbanPulse-AirQuality-LiveClient/1.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode('utf-8'))
                return payload
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='ignore')
            raise ConnectionError(f"data.gov.in API error ({e.code} {e.reason}): {err_body}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to data.gov.in: {e}")

    def parse_records(self, payload: Dict[str, Any]) -> List[LivePollutantObservation]:
        """
        Parses and strictly validates raw records from data.gov.in JSON response.
        Filters corrupt rows, checks physical bounds and units.
        """
        records = payload.get('records', [])
        observations = []

        for rec in records:
            station = rec.get('station', '')
            pollutant_id = rec.get('pollutant_id', '')
            unit = rec.get('pollutant_unit', '')
            raw_ts = rec.get('last_update')
            raw_val = rec.get('pollutant_avg')

            if not station or not pollutant_id:
                continue

            # Standardize pollutant names
            if pollutant_id.upper() == 'OZONE':
                norm_pollutant = 'Ozone'
            else:
                norm_pollutant = pollutant_id.strip()

            # Validate timestamp
            try:
                ts = validate_timestamp(raw_ts)
            except ValueError:
                continue

            # Validate unit
            if not validate_pollutant_unit(norm_pollutant, unit):
                continue

            # Validate physical bounds
            val = validate_physical_bounds(norm_pollutant, raw_val)
            if pd.isna(val) or val is None:
                continue

            lat = float(rec['latitude']) if 'latitude' in rec and pd.notna(rec['latitude']) else None
            lon = float(rec['longitude']) if 'longitude' in rec and pd.notna(rec['longitude']) else None

            obs = LivePollutantObservation(
                station_name=station.strip(),
                pollutant_id=norm_pollutant,
                value=float(val),
                timestamp=ts,
                unit=unit,
                latitude=lat,
                longitude=lon
            )
            observations.append(obs)

        return observations

    def update_buffer_and_compute_aqi(
        self,
        observations: List[LivePollutantObservation]
    ) -> Dict[int, LiveStationAQI]:
        """
        Updates rolling historical buffer and computes CPCB AQI per zone.
        Strict Non-Negotiable Rules:
        - Explicit unmonitored zones (Zones 3, 6, 7, 8, 9, 11, 15) are NEVER fabricated.
        - CPCB minimum criteria: >= 3 pollutants including at least one of PM2.5 or PM10.
        Returns:
            Dict mapping zone_id (1..15) to LiveStationAQI object.
        """
        # Determine latest observation timestamp
        if not observations:
            latest_ts = pd.Timestamp.now(tz='UTC').floor('1h')
        else:
            latest_ts = max(obs.timestamp for obs in observations)

        # Ingest observations into rolling station tables
        for obs in observations:
            st = obs.station_name
            if st not in self.rolling_buffer:
                self.rolling_buffer[st] = pd.DataFrame(
                    columns=['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']
                )

            # Insert or update timestamp row
            ts = obs.timestamp
            p = obs.pollutant_id
            if p in self.rolling_buffer[st].columns:
                self.rolling_buffer[st].loc[ts, p] = obs.value

            # Retain only past 24 hours of data
            cutoff = latest_ts - pd.Timedelta(hours=24)
            self.rolling_buffer[st] = self.rolling_buffer[st][self.rolling_buffer[st].index >= cutoff]

        # Initialize results for all 15 GCC zones
        results: Dict[int, LiveStationAQI] = {}

        # 1. Populate unmonitored zones strictly with is_monitored=False and NaN
        for zone_id in range(1, 16):
            z_match = self.station_mapping_df[self.station_mapping_df['zone_id'] == zone_id]
            z_label = z_match['zone_label'].iloc[0] if not z_match.empty else f"Zone {zone_id}"

            if zone_id in UNMONITORED_AQI_ZONES:
                results[zone_id] = LiveStationAQI(
                    station_name="None (Unmonitored)",
                    zone_id=zone_id,
                    zone_label=z_label,
                    timestamp=latest_ts,
                    sub_indices={},
                    aqi_value=np.nan,
                    dominant_pollutant=None,
                    is_monitored=False,
                    valid_subindex_count=0
                )

        # 2. Process monitored stations and compute CPCB AQI
        for st, df_buf in self.rolling_buffer.items():
            mapped = self.map_station_to_zone(st)
            if not mapped:
                continue

            zone_id, zone_label = mapped

            # Skip if zone is already marked unmonitored
            if zone_id in UNMONITORED_AQI_ZONES:
                continue

            # Compute rolling averages:
            # 24h rolling mean for PM2.5, PM10, NO2, SO2, NH3
            # 8h rolling mean for CO, Ozone
            sub_dict = {}
            for p in ['PM2.5', 'PM10', 'NO2', 'SO2', 'NH3']:
                if p in df_buf.columns and df_buf[p].dropna().shape[0] > 0:
                    conc_mean = float(df_buf[p].dropna().mean())
                    sub = calculate_sub_index(p, conc_mean)
                    if pd.notna(sub):
                        sub_dict[p] = sub

            for p in ['CO', 'Ozone']:
                if p in df_buf.columns:
                    # Past 8 hours only
                    cutoff_8h = latest_ts - pd.Timedelta(hours=8)
                    s_8h = df_buf.loc[df_buf.index >= cutoff_8h, p].dropna()
                    if s_8h.shape[0] > 0:
                        conc_mean = float(s_8h.mean())
                        sub = calculate_sub_index(p, conc_mean)
                        if pd.notna(sub):
                            sub_dict[p] = sub

            aqi_val, dominant_p, count = compute_cpcb_aqi_row(sub_dict)

            results[zone_id] = LiveStationAQI(
                station_name=st,
                zone_id=zone_id,
                zone_label=zone_label,
                timestamp=latest_ts,
                sub_indices=sub_dict,
                aqi_value=float(aqi_val) if pd.notna(aqi_val) else np.nan,
                dominant_pollutant=dominant_p,
                is_monitored=True,
                valid_subindex_count=count
            )

        # Fill any remaining monitored zones where no station reported with is_monitored=True but aqi=NaN
        for zone_id in range(1, 16):
            if zone_id not in results:
                z_match = self.station_mapping_df[self.station_mapping_df['zone_id'] == zone_id]
                z_label = z_match['zone_label'].iloc[0] if not z_match.empty else f"Zone {zone_id}"
                results[zone_id] = LiveStationAQI(
                    station_name="Telemetry Missing",
                    zone_id=zone_id,
                    zone_label=z_label,
                    timestamp=latest_ts,
                    sub_indices={},
                    aqi_value=np.nan,
                    dominant_pollutant=None,
                    is_monitored=True,
                    valid_subindex_count=0
                )

        return results
