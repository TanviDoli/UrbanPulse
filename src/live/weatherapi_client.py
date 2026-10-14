"""
WeatherAPI Live Current-Data Client for UrbanPulse.

ROLE: OBSERVATION SOURCE ONLY.
- Fetches CURRENT weather + air-quality readings for each of the 15 GCC Chennai zones
  using the zone centroids computed from the official GCC 2025 GeoJSON.
- Only the /current.json endpoint is used — the /forecast.json endpoint is NEVER called.
  WeatherAPI's own future forecasts are NEVER used as UrbanPulse prediction output.
- The raw pollutant concentrations (CO, NO2, O3, SO2, PM2.5, PM10) are preserved and
  passed to the existing UrbanPulse CPCB AQI pipeline.
- WeatherAPI's us-epa-index / gb-defra-index are stored as metadata and are
  NEVER labelled as CPCB AQI.

STRICT SAFETY:
- API key is read exclusively from the WEATHERAPI_KEY environment variable.
- The key is never printed, logged, committed, or hardcoded.
"""

import os
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional, Any, Tuple
import datetime
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Zone centroids derived from chennai_zones_2025.geojson (Shapely centroids) ──
# These are the representative query coordinates for WeatherAPI current-data requests.
# Zone ID -> (lat, lon, zone_label)
ZONE_COORDINATES: Dict[int, Tuple[float, float, str]] = {
    1:  (13.1785, 80.3015, "Thiruvottiyur"),
    2:  (13.1877, 80.2658, "Manali"),
    3:  (13.1498, 80.2170, "Madhavaram"),
    4:  (13.1291, 80.2681, "Tondiarpet"),
    5:  (13.0888, 80.2796, "Royapuram"),
    6:  (13.1098, 80.2384, "Thiru-Vi-Ka Nagar"),
    7:  (13.1082, 80.1712, "Ambattur"),
    8:  (13.0876, 80.2216, "Anna Nagar"),
    9:  (13.0479, 80.2558, "Teynampet"),
    10: (13.0450, 80.2101, "Kodambakkam"),
    11: (13.0485, 80.1690, "Valasaravakkam"),
    12: (13.0008, 80.1797, "Alandur"),
    13: (13.0008, 80.2382, "Adyar"),
    14: (12.9511, 80.2200, "Perungudi"),
    15: (12.9023, 80.2362, "Sholinganallur"),
}

WEATHERAPI_BASE_URL = "https://api.weatherapi.com/v1/current.json"

# Physical sanity bounds for incoming observations
_BOUNDS = {
    "temp_c":    (-5.0, 60.0),
    "precip_mm": (0.0, 500.0),
    "co":        (0.0, 100_000.0),   # µg/m³
    "no2":       (0.0, 5_000.0),
    "o3":        (0.0, 3_000.0),
    "so2":       (0.0, 5_000.0),
    "pm2_5":     (0.0, 1_500.0),
    "pm10":      (0.0, 3_000.0),
}


def _clamp(value: Optional[float], key: str) -> Optional[float]:
    """Return value if within physical bounds, else None (not NaN — caller decides)."""
    if value is None:
        return None
    lo, hi = _BOUNDS.get(key, (-1e12, 1e12))
    return float(value) if lo <= value <= hi else None


class WeatherAPIClient:
    """
    Fetches current weather + air-quality observations for all 15 Chennai GCC zones.

    Usage:
        client = WeatherAPIClient()
        obs = client.fetch_all_zones()   # Dict[zone_id -> Dict]

    Each observation dict contains:
        zone_id, zone_label, lat, lon, timestamp_utc,
        temp_c, precip_mm, humidity_pct, condition_text,
        co_ugm3, no2_ugm3, o3_ugm3, so2_ugm3, pm2_5_ugm3, pm10_ugm3,
        weatherapi_us_epa_index,   # METADATA ONLY — NOT CPCB AQI
        weatherapi_gb_defra_index, # METADATA ONLY
        fetch_ok (bool), error_msg (str or None)
    """

    def __init__(self, api_key: Optional[str] = None, request_delay_s: float = 0.25):
        self._key = api_key or os.getenv("WEATHERAPI_KEY")
        if not self._key:
            raise PermissionError(
                "WEATHERAPI_KEY is not configured. "
                "Set the environment variable WEATHERAPI_KEY before running the pipeline."
            )
        self.request_delay_s = request_delay_s

    def _fetch_zone(self, zone_id: int) -> Dict[str, Any]:
        """Fetch current data for a single zone. Returns observation dict."""
        lat, lon, zone_label = ZONE_COORDINATES[zone_id]
        q = f"{lat},{lon}"
        params = {"key": self._key, "q": q, "aqi": "yes"}
        url = f"{WEATHERAPI_BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "UrbanPulse-LiveClient/2.0"})

        base = {
            "zone_id": zone_id,
            "zone_label": zone_label,
            "lat": lat,
            "lon": lon,
            "timestamp_utc": None,
            "temp_c": None,
            "precip_mm": None,
            "humidity_pct": None,
            "condition_text": None,
            "co_ugm3": None,
            "no2_ugm3": None,
            "o3_ugm3": None,
            "so2_ugm3": None,
            "pm2_5_ugm3": None,
            "pm10_ugm3": None,
            "weatherapi_us_epa_index": None,
            "weatherapi_gb_defra_index": None,
            "air_quality_available": False,
            "fetch_ok": False,
            "error_msg": None,
        }

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            base["error_msg"] = f"HTTP {e.code}: {body[:120]}".replace(self._key, "***")
            return base
        except Exception as e:
            base["error_msg"] = str(e)[:120].replace(self._key, "***")
            return base

        loc = data.get("location", {})
        cur = data.get("current", {})
        aq  = cur.get("air_quality") or {}     # absent if aqi=yes is not honoured (e.g. plan/endpoint)

        # Timestamp: WeatherAPI provides epoch_utc
        epoch = cur.get("last_updated_epoch")
        if epoch:
            ts_utc = pd.Timestamp(epoch, unit="s", tz="UTC")
        else:
            ts_utc = pd.Timestamp.now(tz="UTC").floor("h")

        base.update({
            "timestamp_utc": ts_utc,
            "temp_c":          _clamp(cur.get("temp_c"),    "temp_c"),
            "precip_mm":       _clamp(cur.get("precip_mm"), "precip_mm"),
            "humidity_pct":    cur.get("humidity"),
            "condition_text":  cur.get("condition", {}).get("text"),
            # Air quality — raw concentrations, units as provided by WeatherAPI (µg/m³ for all)
            # CO from WeatherAPI is µg/m³ (not mg/m³ like CPCB standard);
            # preserved as-is for model input; label is co_ugm3.
            "co_ugm3":         _clamp(aq.get("co"),    "co"),
            "no2_ugm3":        _clamp(aq.get("no2"),   "no2"),
            "o3_ugm3":         _clamp(aq.get("o3"),    "o3"),
            "so2_ugm3":        _clamp(aq.get("so2"),   "so2"),
            "pm2_5_ugm3":      _clamp(aq.get("pm2_5"), "pm2_5"),
            "pm10_ugm3":       _clamp(aq.get("pm10"),  "pm10"),
            # Index metadata — NEVER label these as CPCB AQI
            "weatherapi_us_epa_index":   aq.get("us-epa-index"),
            "weatherapi_gb_defra_index": aq.get("gb-defra-index"),
            "air_quality_available": bool(aq),
            "fetch_ok": True,
            "error_msg": None,
        })
        return base

    def fetch_all_zones(self) -> Dict[int, Dict[str, Any]]:
        """
        Fetch current observations for all 15 GCC zones.
        Returns Dict[zone_id -> observation dict].
        Zones that fail to fetch are included with fetch_ok=False and error_msg set.
        API key is never included in the returned data.
        """
        results: Dict[int, Dict[str, Any]] = {}
        for zone_id in sorted(ZONE_COORDINATES.keys()):
            obs = self._fetch_zone(zone_id)
            results[zone_id] = obs
            if self.request_delay_s > 0:
                time.sleep(self.request_delay_s)
        return results

    def fetch_zone(self, zone_id: int) -> Dict[str, Any]:
        """Fetch current observation for a single zone."""
        if zone_id not in ZONE_COORDINATES:
            raise ValueError(f"zone_id {zone_id} not in ZONE_COORDINATES (valid: 1-15)")
        return self._fetch_zone(zone_id)
