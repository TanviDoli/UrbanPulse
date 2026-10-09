"""
Comprehensive tests for UrbanPulse Live Ingestion (Step 14A).
Tests:
1. Timestamp validity (rejects corrupt strings like '2025', parses UTC)
2. Unit validation (ug/m3, mg/m3, deg C, mm)
3. Station-to-zone mapping (9 CAAQMS stations to GCC zones)
4. Missing values handling (graceful handling of dropped telemetry)
5. Strict non-fabrication of observations for unmonitored zones
6. Explicit tracking of unmonitored zones (Zones 3, 6, 7, 8, 9, 11, 15)
7. Rainfall 24-hour accumulation (accumulation over 24h vs hourly rate)
8. Temperature climatological normal requirement (never fabricated)
9. End-to-end integration with existing frozen HMM & MRF models
"""

import json
import pytest
import numpy as np
import pandas as pd

from src.live.schemas import (
    validate_timestamp,
    validate_pollutant_unit,
    validate_physical_bounds,
    UNMONITORED_AQI_ZONES,
    UNMONITORED_RAINFALL_ZONES,
    LivePollutantObservation
)
from src.live.cpcb_client import CPCBClient
from src.live.imd_client import IMDClient
from src.live.pipeline import LiveIngestionPipeline
from src.alerts.alert_engine import evaluate_temperature_heatwave_alert, evaluate_rainfall_alert_24h


# ── 1. Timestamp Validity ────────────────────────────────────────────────────

def test_timestamp_validity():
    # Valid timestamps in standard Indian format dd-mm-yyyy hh:mm:ss
    ts1 = validate_timestamp("05-10-2026 12:00:00")
    assert isinstance(ts1, pd.Timestamp)
    assert ts1.tz is not None

    # Valid ISO format
    ts2 = validate_timestamp("2026-10-05T12:00:00Z")
    assert isinstance(ts2, pd.Timestamp)

    # Rejection of isolated concatenation corrupt artifact ('2025')
    with pytest.raises(ValueError, match="Corrupt or incomplete timestamp"):
        validate_timestamp("2025")

    # Rejection of None and NaN
    with pytest.raises(ValueError, match="Timestamp cannot be null"):
        validate_timestamp(None)

    with pytest.raises(ValueError, match="Timestamp cannot be null"):
        validate_timestamp(np.nan)


# ── 2. Units Validation ──────────────────────────────────────────────────────

def test_units_validation():
    # PM2.5 / PM10 / NO2 / SO2 / NH3 / Ozone must be ug/m3
    assert validate_pollutant_unit("PM2.5", "ug/m3")
    assert validate_pollutant_unit("PM10", "µg/m³")
    assert validate_pollutant_unit("NO2", "ug/m^3")
    assert validate_pollutant_unit("SO2", "ug/m3")
    assert validate_pollutant_unit("Ozone", "ug/m3")

    # CO must be mg/m3
    assert validate_pollutant_unit("CO", "mg/m3")
    assert not validate_pollutant_unit("CO", "ug/m3")

    # Temperature and rainfall
    assert validate_pollutant_unit("temperature", "deg C")
    assert validate_pollutant_unit("rainfall", "mm")


# ── 3. Station-to-Zone Mapping ───────────────────────────────────────────────

def test_station_zone_mapping():
    client = CPCBClient()

    # Monitored stations and their known GCC zones
    test_cases = [
        ("Alandur Bus Depot, Chennai - CPCB", 12, "Alandur"),
        ("Manali, Chennai - CPCB", 2, "Manali"),
        ("Manali Village, Chennai - TNPCB", 2, "Manali"),
        ("Kodungaiyur, Chennai - TNPCB", 4, "Tondiarpet"),
        ("Royapuram, Chennai - TNPCB", 5, "Royapuram"),
        ("Arumbakkam, Chennai - TNPCB", 10, "Kodambakkam"),
        ("Velachery Res. Area, Chennai - CPCB", 13, "Adyar"),
        ("Perungudi, Chennai - TNPCB", 14, "Perungudi"),
        ("Gandhi Nagar_Ennore, Chennai - TNPCB", 1, "Thiruvottiyur"),
    ]

    for st_name, expected_zone, expected_label in test_cases:
        mapped = client.map_station_to_zone(st_name)
        assert mapped is not None, f"Failed to map station {st_name}"
        assert mapped[0] == expected_zone
        assert mapped[1] == expected_label

    # Unknown or out-of-boundary station returns None
    assert client.map_station_to_zone("Coimbatore South - TNPCB") is None


# ── 4. Missing Values Handling ───────────────────────────────────────────────

def test_missing_values_handling():
    # Out of physical bounds values are converted to np.nan
    assert np.isnan(validate_physical_bounds("PM2.5", -5.0))
    assert np.isnan(validate_physical_bounds("PM2.5", 5000.0))
    assert np.isnan(validate_physical_bounds("CO", -1.0))
    assert np.isnan(validate_physical_bounds("temperature", 75.0))
    assert np.isnan(validate_physical_bounds("rainfall", -0.1))

    # Valid values pass through
    assert validate_physical_bounds("PM2.5", 45.0) == 45.0
    assert validate_physical_bounds("temperature", 34.5) == 34.5


# ── 5. Strict Non-Fabrication for Unmonitored Zones ──────────────────────────

def test_no_fabricated_observations_for_unmonitored_zones():
    client = CPCBClient()
    with open("data/live/sample_cpcb_response.json", "r") as f:
        sample_payload = json.load(f)

    obs = client.parse_records(sample_payload)
    zone_results = client.update_buffer_and_compute_aqi(obs)

    # Verify that all 7 explicitly unmonitored zones have is_monitored=False and aqi=NaN
    for unmon_z in UNMONITORED_AQI_ZONES:
        res = zone_results[unmon_z]
        assert res.is_monitored is False, f"Zone {unmon_z} must be marked unmonitored"
        assert np.isnan(res.aqi_value), f"Zone {unmon_z} must not have a fabricated AQI observation"
        assert res.valid_subindex_count == 0


# ── 6. Explicit Unmonitored Zones List ───────────────────────────────────────

def test_unmonitored_zones_integrity():
    # Chennai 15 zones: 7 unmonitored for AQI (Zones 3, 6, 7, 8, 9, 11, 15)
    expected_unmonitored_aqi = {3, 6, 7, 8, 9, 11, 15}
    assert UNMONITORED_AQI_ZONES == expected_unmonitored_aqi
    assert len(UNMONITORED_AQI_ZONES) == 7

    # Rainfall monitored only in zones 4, 5, 10, 14
    expected_monitored_rainfall = {4, 5, 10, 14}
    all_zones = set(range(1, 16))
    assert (all_zones - UNMONITORED_RAINFALL_ZONES) == expected_monitored_rainfall


# ── 7. Rainfall 24h Accumulation vs Hourly Rate ──────────────────────────────

def test_rainfall_24h_accumulation():
    imd = IMDClient()
    st_id = "43279"

    # Ingest 24 hourly rainfall values of 2.0 mm/h
    # Total 24h accumulation = 48.0 mm (less than 64.5 mm threshold)
    base_time = pd.Timestamp("2026-10-05 00:00:00", tz="UTC")
    accum = 0.0
    for hour in range(24):
        t = base_time + pd.Timedelta(hours=hour)
        accum = imd.compute_24h_rainfall_accumulation(st_id, hourly_rainfall=2.0, timestamp=t)

    assert accum == 48.0

    # Hourly rate is 2.0 mm, which is far below 64.5 mm
    # But if evaluated as 24h accumulation, alert engine evaluates correctly
    alert_sub = evaluate_rainfall_alert_24h(mu_accum_24=accum, var_accum_24=4.0, is_monitored=True)
    assert alert_sub['alert_status'] == 'NORMAL'

    # Now add heavy rainfall bringing 24h accumulation to 70.0 mm (>= 64.5 mm)
    t_heavy = base_time + pd.Timedelta(hours=24)
    accum_heavy = imd.compute_24h_rainfall_accumulation(st_id, hourly_rainfall=24.0, timestamp=t_heavy)
    alert_heavy = evaluate_rainfall_alert_24h(mu_accum_24=accum_heavy, var_accum_24=4.0, is_monitored=True)
    assert alert_heavy['alert_status'] in ['WATCH', 'WARNING']

    # Unmonitored zone check: must be UNMONITORED_INSUFFICIENT_DATA
    alert_unmon = evaluate_rainfall_alert_24h(mu_accum_24=accum_heavy, var_accum_24=4.0, is_monitored=False)
    assert alert_unmon['alert_status'] == 'UNMONITORED_INSUFFICIENT_DATA'
    assert np.isnan(alert_unmon['probability_crossing'])


# ── 8. Temperature Normal Requirement (Never Fabricated) ─────────────────────

def test_temperature_normal_requirement():
    # If normal_tmax is missing/None, alert status MUST be UNAVAILABLE_MISSING_CLIMATOLOGY
    alert_missing = evaluate_temperature_heatwave_alert(tmax_pred=39.0, tmax_var=1.5, normal_tmax=None)
    assert alert_missing['alert_status'] == 'UNAVAILABLE_MISSING_CLIMATOLOGY'
    assert np.isnan(alert_missing['probability_crossing'])

    # Even with Tmax >= 37C (e.g. 38.5C), 37C alone is NEVER a standalone heatwave trigger without normal
    assert alert_missing['alert_status'] == 'UNAVAILABLE_MISSING_CLIMATOLOGY'

    # When verified normal is provided (e.g. Normal = 33.0C, Effective threshold = max(37.0, 33.0 + 4.5) = 37.5C)
    alert_valid = evaluate_temperature_heatwave_alert(tmax_pred=39.0, tmax_var=1.0, normal_tmax=33.0)
    assert alert_valid['alert_status'] in ['WATCH', 'WARNING']
    assert not np.isnan(alert_valid['probability_crossing'])


# ── 9. Live Pipeline Integration With Existing Frozen Models ────────────────

def test_live_pipeline_integration():
    pipeline = LiveIngestionPipeline()

    with open("data/live/sample_cpcb_response.json", "r") as f:
        cpcb_sample = json.load(f)

    result = pipeline.run_aqi_hourly_cycle(cpcb_sample, horizon=1)

    assert result['variable'] == 'AQI'
    assert result['horizon'] == '1h'
    assert len(result['zones']) == 15

    # Check unmonitored zones have is_monitored=False and observed_aqi is NaN
    for z_res in result['zones']:
        zid = z_res['zone_id']
        if zid in UNMONITORED_AQI_ZONES:
            assert z_res['is_monitored'] is False
            assert np.isnan(z_res['observed_aqi'])
            # But MRF refined forecast is computed via spatial coupling!
            assert pd.notna(z_res['mrf_refined_forecast'])
            assert z_res['alert_status'] in ['NORMAL', 'WATCH', 'WARNING']
        else:
            assert z_res['is_monitored'] is True
