"""
Hourly temperature forecast for UrbanPulse (Open-Meteo reanalysis, one regression per horizon h).

  T(t+h) = b0 + b1*T(t) + b2*T(t+h-24) + b3*mean(T(t-23..t)) + b4*sin(2pi*hour/24) + b5*cos(...)

hour = IST hour of the TARGET time. All inputs are known at the origin t (t+h-24 <= t for h <= 24),
so the model nests persistence (b1 = 1) and same-hour-yesterday (b2 = 1). Coefficients are fitted
by least squares on TRAINING rows only; the predictive std is the training residual std.
Replaces nothing: the frozen daily-mean temperature HMMs are kept.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

TEMP_FEATURES = ['const', 't_now', 't_same_hour_yday', 't_mean_24h', 'hour_sin', 'hour_cos']


def temp_features(series: pd.Series, h: int) -> pd.DataFrame:
    """Features at every origin t for horizon h (1 <= h <= 24), using T up to t only."""
    if not 1 <= h <= 24:
        raise ValueError('horizon must be 1..24 h')
    y = series.astype(float)
    target_hour = (series.index + pd.Timedelta(hours=h)).tz_convert('Asia/Kolkata').hour.to_numpy()
    ang = 2 * np.pi * target_hour / 24.0
    return pd.DataFrame({
        'const': 1.0,
        't_now': y.to_numpy(),
        't_same_hour_yday': y.shift(24 - h).to_numpy(),          # T(t+h-24), already observed at t
        't_mean_24h': y.rolling(24, min_periods=24).mean().to_numpy(),
        'hour_sin': np.sin(ang), 'hour_cos': np.cos(ang),
    }, index=series.index)


class HourlyTempModel:
    def __init__(self, horizon_h: int, feature_names=None):
        self.horizon_h = horizon_h
        self.feature_names = list(feature_names or TEMP_FEATURES)
        self.coef_: Optional[np.ndarray] = None
        self.resid_std_: float = float('nan')
        self.n_train_ = 0

    @property
    def features(self):
        # Artifacts saved before feature_names existed use the original six features.
        return getattr(self, 'feature_names', None) or TEMP_FEATURES

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'HourlyTempModel':
        A = X[self.features].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.coef_
        self.resid_std_ = float(np.std(resid, ddof=len(self.features)))
        self.n_train_ = int(len(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.features].to_numpy(dtype=float) @ self.coef_

    def summary(self) -> Dict[str, float]:
        return {'horizon_h': self.horizon_h, 'resid_std_c': self.resid_std_, 'n_train': self.n_train_,
                **{f'coef_{n}': float(c) for n, c in zip(self.features, self.coef_)}}


# ── candidate features (experiment; production uses TEMP_FEATURES only) ──────
TEMP_CANDIDATE_EXTRA = ['t_lag1', 't_lag3', 't_same_hour_2d', 't_max_24h', 't_min_24h', 'hour_sin2', 'hour_cos2',
                        'month_sin', 'month_cos', 'rh_now', 'wind_u', 'wind_v', 'clim_target', 'anom_now']


def weather_origin_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Humidity and wind vector at the origin (columns relative_humidity_2m_pct, wind_speed_10m_kmh,
    wind_direction_10m_deg; direction = where the wind blows FROM)."""
    d = np.deg2rad(frame['wind_direction_10m_deg'].to_numpy(dtype=float))
    sp = frame['wind_speed_10m_kmh'].to_numpy(dtype=float)
    return pd.DataFrame({'rh_now': frame['relative_humidity_2m_pct'].to_numpy(dtype=float),
                         'wind_u': -sp * np.sin(d), 'wind_v': -sp * np.cos(d)}, index=frame.index)


def temp_candidate_features(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    """TEMP_FEATURES plus lags, 24 h extremes, 2nd diurnal harmonic, target-month seasonality and origin
    humidity/wind. All from values at or before the origin t. ('clim_target' and 'anom_now' are added
    later from TRAINING-only climatology.)"""
    y = frame['temperature_2m_c'].astype(float)
    base = temp_features(y, h)
    tgt = (frame.index + pd.Timedelta(hours=h)).tz_convert('Asia/Kolkata')
    ang = 2 * np.pi * tgt.hour.to_numpy() / 24.0
    mon = 2 * np.pi * (tgt.month.to_numpy() - 1) / 12.0
    extra = pd.DataFrame({
        't_lag1': y.shift(1).to_numpy(), 't_lag3': y.shift(3).to_numpy(),
        't_same_hour_2d': y.shift(48 - h).to_numpy(),
        't_max_24h': y.rolling(24, min_periods=24).max().to_numpy(),
        't_min_24h': y.rolling(24, min_periods=24).min().to_numpy(),
        'hour_sin2': np.sin(2 * ang), 'hour_cos2': np.cos(2 * ang),
        'month_sin': np.sin(mon), 'month_cos': np.cos(mon),
    }, index=frame.index)
    return pd.concat([base, extra, weather_origin_features(frame)], axis=1)
