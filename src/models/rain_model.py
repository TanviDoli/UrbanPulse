"""
Two-part (hurdle) rainfall model for UrbanPulse, trained on Open-Meteo reanalysis.

  Part 1  occurrence: P(Y >= WET_MM) from a logistic regression on forecast-time features.
  Part 2  amount:     Y | wet = WET_MM + Z,  Z ~ Gamma(shape, scale), one per IMD season, fitted by
                      the method of moments on TRAINING targets only (the excess over the wet
                      threshold, so P(Y >= WET_MM | wet) = 1 exactly).

  Predictive distribution:  F(x) = (1 - p) + p * G(x - WET_MM),  x >= 0  (point mass at 0 for dry).

One instance per target: an hourly RATE at t+h (mm/h) or the 24 h TOTAL over t+1..t+24 (mm).
Features (build_features) use y_1..y_t only. Prediction is plain numpy on stored parameters
(standardisation, coefficients, gamma parameters), so it is deterministic and inspectable.
This is not an HMM; the Gaussian rainfall HMMs in hmm_model.py are unchanged.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

WET_MM = 0.1                       # Open-Meteo precipitation resolution; dry = exactly 0 below this
MIN_WET_FOR_SEASON = 30            # fewer wet training targets in a season -> overall gamma

# IMD seasons by IST calendar month
SEASON_OF_MONTH = {1: 'winter', 2: 'winter', 3: 'pre_monsoon', 4: 'pre_monsoon', 5: 'pre_monsoon',
                   6: 'sw_monsoon', 7: 'sw_monsoon', 8: 'sw_monsoon', 9: 'sw_monsoon',
                   10: 'ne_monsoon', 11: 'ne_monsoon', 12: 'ne_monsoon'}

FEATURES = ['wet_now', 'log_rain_now', 'log_rain_3h', 'log_rain_24h', 'wet_frac_24h',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos']


def build_features(series: pd.Series) -> pd.DataFrame:
    """
    Forecast-time features at every origin t (UTC hourly index) from y_1..y_t only:
    current wetness/amount, trailing 3 h and 24 h totals, wet fraction of the last 24 h,
    and IST hour/month of the ORIGIN (the target hour is origin + fixed h).
    Trailing windows need complete data (NaN otherwise); nothing is filled.
    """
    y = series.astype(float)
    wet = (y >= WET_MM).astype(float).where(y.notna())
    ist = series.index.tz_convert('Asia/Kolkata')
    hour = 2 * np.pi * ist.hour.to_numpy() / 24.0
    month = 2 * np.pi * (ist.month.to_numpy() - 1) / 12.0
    return pd.DataFrame({
        'wet_now': wet.to_numpy(),
        'log_rain_now': np.log1p(y).to_numpy(),
        'log_rain_3h': np.log1p(y.rolling(3, min_periods=3).sum()).to_numpy(),
        'log_rain_24h': np.log1p(y.rolling(24, min_periods=24).sum()).to_numpy(),
        'wet_frac_24h': wet.rolling(24, min_periods=24).mean().to_numpy(),
        'hour_sin': np.sin(hour), 'hour_cos': np.cos(hour),
        'month_sin': np.sin(month), 'month_cos': np.cos(month),
    }, index=series.index)


RAIN_WEATHER_FEATURES = ['rh_now', 'rh_change_3h', 'wind_u', 'wind_v', 't_dev_24h', 'hour_sin2', 'hour_cos2']


def rain_weather_features(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Origin-time weather inputs for the candidate rain models (shared by training and live):
    humidity, 3 h humidity change, wind vector, temperature minus its trailing 24 h mean, and the
    2nd diurnal harmonic of the TARGET IST hour. frame: hourly UTC index with temperature_2m_c,
    relative_humidity_2m_pct, wind_speed_10m_kmh, wind_direction_10m_deg.
    """
    d = np.deg2rad(frame['wind_direction_10m_deg'].to_numpy(dtype=float))
    sp = frame['wind_speed_10m_kmh'].to_numpy(dtype=float)
    rh = frame['relative_humidity_2m_pct'].astype(float)
    t = frame['temperature_2m_c'].astype(float)
    ang = 2 * np.pi * (frame.index + pd.Timedelta(hours=h)).tz_convert('Asia/Kolkata').hour.to_numpy() / 24.0
    return pd.DataFrame({'rh_now': rh.to_numpy(), 'rh_change_3h': rh.diff(3).to_numpy(),
                         'wind_u': -sp * np.sin(d), 'wind_v': -sp * np.cos(d),
                         't_dev_24h': (t - t.rolling(24, min_periods=24).mean()).to_numpy(),
                         'hour_sin2': np.sin(2 * ang), 'hour_cos2': np.cos(2 * ang)}, index=frame.index)


def gamma_moments(amounts: np.ndarray) -> Optional[Tuple[float, float]]:
    """(shape, scale) by the method of moments; None if fewer than 2 values or no spread."""
    a = np.asarray(amounts, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2 or a.var() <= 0 or a.mean() <= 0:
        return None
    return float(a.mean() ** 2 / a.var()), float(a.var() / a.mean())


def crps_zero_inflated_gamma(y: np.ndarray, p: np.ndarray, shape: np.ndarray, scale: np.ndarray,
                             loc: float = WET_MM) -> np.ndarray:
    """
    Closed-form CRPS of X = 0 w.p. (1-p), X = loc + Z w.p. p, Z ~ Gamma(a, s), at observation y >= 0,
    from CRPS = E|X - y| - 0.5 E|X - X'|  with c = y - loc:
      E|X - y|  = (1-p) y + p [ c (2 G_a(c) - 1) + a s (1 - 2 G_{a+1}(c)) ]   (G = 0 for c < 0)
      E|X - X'| = 2 p (1-p) (loc + a s) + p^2 * 2 s / B(1/2, a)
    p = 0 gives |y| (a deterministic zero forecast); p = 1, loc = 0 gives the plain gamma CRPS.
    """
    y, p, a, s = (np.asarray(v, dtype=float) for v in (y, p, shape, scale))
    q = 1.0 - p
    c = y - loc
    g_a = stats.gamma.cdf(c, a, scale=s)
    g_a1 = stats.gamma.cdf(c, a + 1.0, scale=s)
    e_abs = c * (2 * g_a - 1) + a * s * (1 - 2 * g_a1)
    crps = q * y + p * e_abs - q * p * (loc + a * s) - p ** 2 * s / special.beta(0.5, a)
    return np.where(p == 0, y, crps)          # dry-certain forecast: no gamma term (also if a, s are NaN)


class TwoPartRainModel:
    """Occurrence (logistic) + seasonal gamma amounts for one target ('Nh' rate or '24h_total')."""

    def __init__(self, target_key: str, feature_names=None, wet_threshold: float = WET_MM, C: float = 1.0):
        self.target_key = target_key
        self.feature_names = list(feature_names or FEATURES)
        self.wet_threshold = wet_threshold
        self.C = C                         # fixed regularisation: no hyper-parameter search, no selection leakage
        self.mean_ = self.scale_ = self.coef_ = None
        self.intercept_ = 0.0
        self.gamma_by_season: Dict[str, Tuple[float, float]] = {}
        self.gamma_overall: Optional[Tuple[float, float]] = None
        self.n_train_ = 0

    # ── fitting (training rows only) ─────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: np.ndarray, target_seasons: np.ndarray) -> 'TwoPartRainModel':
        from sklearn.linear_model import LogisticRegression
        Xv = X[self.feature_names].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        wet = (y >= self.wet_threshold).astype(int)
        self.mean_ = Xv.mean(axis=0)
        sd = Xv.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        if wet.min() == wet.max():          # one class only: constant probability, no regression
            self.coef_ = np.zeros(Xv.shape[1])
            self.intercept_ = float(special.logit(np.clip(wet.mean(), 1e-6, 1 - 1e-6)))
        else:
            lr = LogisticRegression(C=self.C, max_iter=2000).fit((Xv - self.mean_) / self.scale_, wet)
            self.coef_, self.intercept_ = lr.coef_.ravel().astype(float), float(lr.intercept_[0])
        wet_mask = wet == 1
        excess = y - self.wet_threshold
        self.gamma_overall = gamma_moments(excess[wet_mask])
        seasons = np.asarray(target_seasons)
        self.gamma_by_season = {}
        for season in np.unique(seasons):
            amounts = excess[wet_mask & (seasons == season)]
            if len(amounts) >= MIN_WET_FOR_SEASON and (g := gamma_moments(amounts)) is not None:
                self.gamma_by_season[str(season)] = g
        self.n_train_ = int(len(y))
        return self

    # ── prediction ───────────────────────────────────────────────────────────
    def predict_proba_wet(self, X: pd.DataFrame) -> np.ndarray:
        z = ((X[self.feature_names].to_numpy(dtype=float) - self.mean_) / self.scale_) @ self.coef_ + self.intercept_
        return special.expit(z)

    def amount_params(self, target_seasons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gamma (shape, scale) per row; NaN if no amount distribution could be fitted."""
        fallback = self.gamma_overall or (np.nan, np.nan)
        pars = np.array([self.gamma_by_season.get(str(s), fallback) for s in np.asarray(target_seasons)], dtype=float)
        return pars[:, 0], pars[:, 1]

    def predict(self, X: pd.DataFrame, target_seasons: np.ndarray) -> Dict[str, np.ndarray]:
        """p_wet, gamma shape/scale, predictive mean (RMSE point forecast) and median (MAE point forecast)."""
        p = self.predict_proba_wet(X)
        a, s = self.amount_params(target_seasons)
        return {'p_wet': p, 'shape': a, 'scale': s, 'mean': mixture_mean(p, a, s, self.wet_threshold),
                'median': mixture_median(p, a, s, self.wet_threshold)}


def mixture_mean(p, a, s, loc: float = WET_MM) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.where(p == 0, 0.0, p * (loc + np.asarray(a) * np.asarray(s)))


def mixture_median(p, a, s, loc: float = WET_MM) -> np.ndarray:
    """0 when P(dry) >= 0.5, else loc + the gamma quantile solving (1-p) + p G(m - loc) = 0.5."""
    p, a, s = (np.asarray(v, dtype=float) for v in (p, a, s))
    out = np.zeros_like(p)
    wet = p > 0.5
    out[wet] = loc + stats.gamma.ppf((p[wet] - 0.5) / p[wet], a[wet], scale=s[wet])
    return out


def prob_exceed(p, a, s, threshold: float, loc: float = WET_MM) -> np.ndarray:
    """P(Y >= threshold) under the mixture; equals p at the wet threshold itself."""
    p = np.asarray(p, dtype=float)
    if threshold <= loc:
        return p
    return p * stats.gamma.sf(threshold - loc, np.asarray(a), scale=np.asarray(s))
