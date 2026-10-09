"""
OpenAQ v3 API client for UrbanPulse (data collection layer only).

- Reads OPENAQ_API_KEY from the environment / project .env. The key is sent only
  in the X-API-Key request header and is never printed, logged or written to disk.
- Respects the documented rate-limit headers (X-Ratelimit-Remaining / -Reset) and
  retries 429 / 5xx / network errors with bounded backoff.
- Every request outcome is recorded in `self.request_log` (URL path + params,
  status, error) so audits can report failed requests reproducibly.

Endpoints used (validated against live responses on 2026-10-09):
  GET /v3/locations?bbox=minLon,minLat,maxLon,maxLat&limit&page
  GET /v3/locations/{id}/sensors
  GET /v3/sensors/{id}/hours?datetime_from&datetime_to&limit&page
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# Greater Chennai Corporation bounding box (from chennai_zones_2025.geojson, padded ~3 km)
CHENNAI_BBOX = (80.10, 12.82, 80.36, 13.27)


class OpenAQError(RuntimeError):
    """Raised when an OpenAQ request ultimately fails. Never contains the API key."""


class OpenAQClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        max_retries: int = 5,
        session: Optional[requests.Session] = None,
    ):
        self._key = api_key or os.getenv("OPENAQ_API_KEY")
        if not self._key:
            raise PermissionError(
                "OPENAQ_API_KEY is not configured. Add it to the project .env file."
            )
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._session = session
        self._local = threading.local()
        self.request_log: List[Dict[str, Any]] = []

    # Global throttle shared by all threads: <= ~55 request starts per minute
    # (documented limit 60/min), so concurrent workers cannot exceed the quota.
    _MIN_INTERVAL_S = 60.0 / 55.0
    _throttle_lock = threading.Lock()
    _next_slot = 0.0

    @property
    def session(self) -> requests.Session:
        if self._session is not None:
            return self._session
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
        return self._local.session

    @classmethod
    def _throttle(cls) -> None:
        with cls._throttle_lock:
            now = time.monotonic()
            wait = cls._next_slot - now
            cls._next_slot = max(now, cls._next_slot) + cls._MIN_INTERVAL_S
        if wait > 0:
            time.sleep(wait)

    # ── low level ────────────────────────────────────────────────────────────
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{OPENAQ_BASE_URL}{path}"
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            status = None
            try:
                self._throttle()
                resp = self.session.get(
                    url, params=params, headers={"X-API-Key": self._key}, timeout=self.timeout_s
                )
                status = resp.status_code
                self._respect_rate_limit(resp)
                if status == 200:
                    self.request_log.append({"path": path, "params": dict(params or {}),
                                             "status": status, "attempt": attempt, "error": None})
                    return resp.json()
                if status in (401, 403):
                    # Authentication problems are not retried; message never echoes the key.
                    last_err = f"HTTP {status}: authentication/authorization rejected"
                    break
                if status == 404:
                    last_err = "HTTP 404: resource not found"
                    break
                if status == 422:
                    last_err = f"HTTP 422: invalid parameters ({resp.text[:160]})"
                    break
                if status == 429:
                    wait = _header_float(resp, "Retry-After") or _header_float(resp, "X-Ratelimit-Reset") or 60.0
                    last_err = "HTTP 429: rate limited"
                    time.sleep(min(max(wait, 1.0), 120.0))
                    continue
                last_err = f"HTTP {status}: {resp.text[:160]}"
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = f"{type(e).__name__}: {str(e)[:160]}"
            except ValueError as e:  # invalid JSON
                last_err = f"Invalid JSON: {str(e)[:160]}"
            time.sleep(min(2 ** attempt, 30))

        self.request_log.append({"path": path, "params": dict(params or {}),
                                 "status": status, "attempt": attempt, "error": last_err})
        raise OpenAQError(f"GET {path} failed: {last_err}")

    @staticmethod
    def _respect_rate_limit(resp: requests.Response) -> None:
        remaining = _header_float(resp, "X-Ratelimit-Remaining")
        reset = _header_float(resp, "X-Ratelimit-Reset")
        if remaining is not None and remaining <= 1:
            time.sleep(min(max(reset or 60.0, 1.0), 120.0))

    def _paginate(self, path: str, params: Dict[str, Any], limit: int = 1000,
                  max_pages: int = 100) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            j = self._get(path, {**params, "limit": limit, "page": page})
            batch = j.get("results", []) or []
            results.extend(batch)
            if len(batch) < limit:
                break
        return results

    # ── endpoints ────────────────────────────────────────────────────────────
    def list_locations(self, bbox=CHENNAI_BBOX) -> List[Dict[str, Any]]:
        return self._paginate("/locations", {"bbox": ",".join(str(v) for v in bbox)}, limit=1000)

    def list_sensors(self, location_id: int) -> List[Dict[str, Any]]:
        return self._paginate(f"/locations/{location_id}/sensors", {}, limit=1000)

    def sensor_hours(self, sensor_id: int, datetime_from: str, datetime_to: str) -> List[Dict[str, Any]]:
        """Hourly aggregates for one sensor in [datetime_from, datetime_to) (ISO-8601 UTC)."""
        return self._paginate(
            f"/sensors/{sensor_id}/hours",
            {"datetime_from": datetime_from, "datetime_to": datetime_to},
            limit=1000,
        )


def _header_float(resp: requests.Response, name: str) -> Optional[float]:
    try:
        v = resp.headers.get(name)
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def flatten_hour_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one /sensors/{id}/hours result into a tabular row."""
    period = rec.get("period") or {}
    cov = rec.get("coverage") or {}
    param = rec.get("parameter") or {}
    flags = rec.get("flagInfo") or {}
    return {
        "hour_start_utc": ((period.get("datetimeFrom") or {}).get("utc")),
        "hour_end_utc": ((period.get("datetimeTo") or {}).get("utc")),
        "value": rec.get("value"),
        "parameter": param.get("name"),
        "units": param.get("units"),
        "observed_count": cov.get("observedCount"),
        "expected_count": cov.get("expectedCount"),
        "percent_coverage": cov.get("percentCoverage"),
        "has_flags": flags.get("hasFlags"),
    }
