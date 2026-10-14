"""
Data freshness rules shared by the live pipeline (src/live/live_pipeline.py) and the
dashboard (dashboard/data_loader.py). Dependency-light on purpose: importing it does not
load models or API clients.

RULES
  R1 Observation freshness
     age = now - observation time. An AQI observation is CURRENT only if
     age <= AQI_OBS_MAX_AGE_HOURS (default 3 h, override with the environment variable
     URBANPULSE_AQI_MAX_AGE_HOURS). Older readings stay visible with their real timestamp
     and age but are labelled STALE. Applied identically to every station, and re-applied
     at dashboard view time because the output file itself ages.
  R2 Forecast timing
     target time = forecast origin + horizon (never moved forward). At evaluation time:
       EXPIRED      target time <= now (no longer a prediction of the future)
       STALE_INPUT  target in the future, but the input observation is older than R1 allows
       CURRENT      target in the future and input observation current
       UNKNOWN      target or input time missing / unparseable
     Only CURRENT forecasts count as evaluated; anything else is "not evaluated".
  R3 Display risk
     Low (green) only for an evaluated NORMAL forecast from measured input.
     Spatial estimates (MRF, no local measurement): NORMAL -> 'Estimate' (uncertain, grey);
     WATCH/WARNING -> Moderate/High, flagged as an estimate.
     Non-CURRENT forecasts are never green: 'Stale', or 'Stale warning' when the old
     forecast contained WATCH/WARNING (kept for review, never counted as an active alert).
  R4 Pipeline output
     Overdue when now > next_update_utc + PIPELINE_OVERDUE_GRACE_MINUTES. A one-off run
     (no scheduler) has no next update.
"""

import os
from typing import Any, Dict, Optional

import pandas as pd


def _env_hours(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


AQI_OBS_MAX_AGE_HOURS = _env_hours('URBANPULSE_AQI_MAX_AGE_HOURS', 3.0)
PIPELINE_OVERDUE_GRACE_MINUTES = 10

# R2 forecast freshness
CURRENT = 'CURRENT'
EXPIRED = 'EXPIRED'
STALE_INPUT = 'STALE_INPUT'
UNKNOWN = 'UNKNOWN'

# R3 display risk levels that are not an evaluated Low/Moderate/High
LEVEL_ESTIMATE = 'Estimate'
LEVEL_STALE = 'Stale'
LEVEL_STALE_WARNING = 'Stale warning'
LEVEL_UNAVAILABLE = 'Unavailable'

_EVALUATED_LEVEL = {'NORMAL': 'Low', 'WATCH': 'Moderate', 'WARNING': 'High'}

# R4 pipeline states
ON_SCHEDULE = 'ON_SCHEDULE'
OVERDUE = 'OVERDUE'
NO_SCHEDULE = 'NO_SCHEDULE'


def to_utc(t: Any) -> Optional[pd.Timestamp]:
    """Parse to a tz-aware UTC Timestamp; None if missing or unparseable (naive = UTC)."""
    if t is None:
        return None
    try:
        ts = pd.Timestamp(t)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def age_hours(t: Any, now: Any) -> Optional[float]:
    t, now = to_utc(t), to_utc(now)
    if t is None or now is None:
        return None
    return (now - t) / pd.Timedelta(hours=1)


def is_observation_current(observed_at: Any, now: Any, max_age_hours: Optional[float] = None) -> bool:
    """R1. Missing observation time is never current."""
    a = age_hours(observed_at, now)
    limit = AQI_OBS_MAX_AGE_HOURS if max_age_hours is None else max_age_hours
    return a is not None and a <= limit


def forecast_freshness(target_time: Any, input_observed_at: Any, now: Any,
                       max_age_hours: Optional[float] = None) -> str:
    """R2."""
    target, now_ts = to_utc(target_time), to_utc(now)
    if target is None or now_ts is None or to_utc(input_observed_at) is None:
        return UNKNOWN
    if target <= now_ts:
        return EXPIRED
    if not is_observation_current(input_observed_at, now_ts, max_age_hours):
        return STALE_INPUT
    return CURRENT


def aqi_forecast_input_time(zone_id: int, availability: Optional[str], zone_inputs: Optional[Dict],
                            origin: Any) -> Any:
    """
    Observation time behind a zone's AQI forecast: the zone's own last measurement for an
    HMM+MRF forecast (MODEL_FORECAST); the forecast origin (newest measured station hour)
    for an MRF spatial estimate, which has no local measurement.
    """
    if availability == 'MODEL_FORECAST':
        zi = (zone_inputs or {}).get(zone_id) or (zone_inputs or {}).get(str(zone_id)) or {}
        return zi.get('last_measured_utc') or origin
    return origin


def display_risk(alert_status: Optional[str], availability: Optional[str], freshness: str) -> Dict[str, Any]:
    """
    R3. Returns {'level', 'evaluated', 'stale_warning', 'is_estimate'}.
    evaluated=True only for a CURRENT forecast with a NORMAL/WATCH/WARNING status.
    """
    is_estimate = availability == 'MODEL_ESTIMATE'
    out = {'level': LEVEL_UNAVAILABLE, 'evaluated': False, 'stale_warning': False, 'is_estimate': is_estimate}
    if availability in (None, 'UNAVAILABLE') or alert_status not in _EVALUATED_LEVEL:
        return out
    if freshness != CURRENT:
        out['stale_warning'] = alert_status in ('WATCH', 'WARNING')
        out['level'] = LEVEL_STALE_WARNING if out['stale_warning'] else LEVEL_STALE
        return out
    out['evaluated'] = True
    out['level'] = LEVEL_ESTIMATE if (is_estimate and alert_status == 'NORMAL') else _EVALUATED_LEVEL[alert_status]
    return out


def pipeline_status(meta: Dict[str, Any], now: Any) -> Dict[str, Any]:
    """R4. {'state', 'minutes_to_next', 'overdue_minutes', 'output_age_hours'}."""
    now_ts = to_utc(now)
    nxt = to_utc(meta.get('next_update_utc'))
    out = {'state': NO_SCHEDULE, 'minutes_to_next': None, 'overdue_minutes': None,
           'output_age_hours': age_hours(meta.get('generated_at_utc'), now_ts)}
    if nxt is None or now_ts is None:
        return out
    delta_min = (nxt - now_ts) / pd.Timedelta(minutes=1)
    if delta_min < -PIPELINE_OVERDUE_GRACE_MINUTES:
        out.update(state=OVERDUE, overdue_minutes=int(-delta_min))
    else:
        out.update(state=ON_SCHEDULE, minutes_to_next=max(0, int(delta_min)))
    return out


def rules_summary() -> Dict[str, Any]:
    """Written into the pipeline output so the applied thresholds are recorded."""
    return {
        'aqi_observation_max_age_hours': AQI_OBS_MAX_AGE_HOURS,
        'pipeline_overdue_grace_minutes': PIPELINE_OVERDUE_GRACE_MINUTES,
        'forecast_states': [CURRENT, STALE_INPUT, EXPIRED, UNKNOWN],
        'note': 'Only CURRENT forecasts (future target, input observation within the max age) are '
                'evaluated or counted as active alerts; stale/expired WATCH/WARNING are kept as stale '
                'warnings for review.',
    }
