"""
24-hour rainfall TOTAL experiment on the 2017-2023 history. EXPERIMENT ONLY.

Target y(t) = sum of hourly precipitation over t+1..t+24 (mm) per unique grid cell, every hourly origin t.
Candidates (pre-declared in results/experiments/long_horizon_history/rain24_total/criteria.md):
  A0 existing production-input two-part model; B direct Tweedie regression; C hurdle (P>=0.1, P>=2.5,
  trace amounts, gamma regression above 2.5 mm), incl. a training-fold-only amount correction and recency
  weighting; D aggregation of 24 hourly two-part forecasts.
All forecasts are scored as predictive distributions (50 quantiles -> CRPS) against training-only
climatological / conditional-climatological distributions and persistence, plus event Brier skill (>= 2.5 mm),
point errors, bias and calibration, with 7-day-block bootstrap intervals.

Reuses the long_horizon_history data path (unique grid cells, folds, 2024 guards). 2024 is never opened.
Writes only under models/experiments/long_horizon_history/rain24_total and
results/experiments/long_horizon_history/rain24_total.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.models import long_horizon_history as lh
from src.models import train_and_validate_hmm as tv
from src.models.rain_model import FEATURES as PROD_FEATURES, SEASON_OF_MONTH, TwoPartRainModel, build_features, \
    mixture_mean, mixture_median, prob_exceed

ROOT = lh.ROOT
OUT_MODELS = os.path.join('models', 'experiments', 'long_horizon_history', 'rain24_total')
OUT_RESULTS = os.path.join('results', 'experiments', 'long_horizon_history', 'rain24_total')
CRITERIA = os.path.join(OUT_RESULTS, 'criteria.md')
ARTIFACT = os.path.join(OUT_MODELS, 'rain24_total_candidates_refit_2017_2023.joblib')
REPRO_JSON = os.path.join(OUT_RESULTS, 'baseline_reproduction.json')
CV_CSV = os.path.join(OUT_RESULTS, 'cv_selection.csv')
CV_GROUPS_CSV = os.path.join(OUT_RESULTS, 'cv_metrics_by_group.csv')
VAL_CSV = os.path.join(OUT_RESULTS, 'validation_2023_by_group.csv')
ROB_CSV = os.path.join(OUT_RESULTS, 'robustness_2025_2026_by_group.csv')
REL_CSV = os.path.join(OUT_RESULTS, 'reliability.csv')
SUMMARY_JSON = os.path.join(OUT_RESULTS, 'summary.json')
REPORT = os.path.join(OUT_RESULTS, 'report.md')
RUN_MANIFEST = os.path.join(OUT_RESULTS, 'run_manifest.json')
OUTPUTS = (ARTIFACT, REPRO_JSON, CV_CSV, CV_GROUPS_CSV, VAL_CSV, ROB_CSV, REL_CSV, SUMMARY_JSON, REPORT, RUN_MANIFEST)

WET_MM, EVENT_MM, HEAVY_MM = 0.1, 2.5, 64.5
EXCESS_LOC = 2.4                      # amounts >= 2.5 mm modelled as 2.4 + Gamma (> 0)
TAUS = (np.arange(50) + 0.5) / 50
HOURS = list(range(1, 25))
CANDIDATES = ['A0', 'B1', 'B2', 'C1', 'C2', 'C2_cal', 'C2_recent', 'D1']
EXT_RAW = ['log_rain_72h', 'log_rain_7d', 'wet_frac_72h', 'wet_frac_7d', 'log_hours_since_wet', 'rh_now', 'rh_mean_24h',
           'rh_change_24h', 'wind_u', 'wind_v', 'wind_u_24h', 'wind_v_24h', 't_dev_24h', 't_range_24h',
           'is_sw', 'is_ne', 'is_pre']
CLIM_FEATS = ['logit_pclim25', 'logit_pcond25', 'log_clim_mean']
GLM_ALPHA, LOGIT_C = 1e-4, 1.0
CAL_CLIP = (0.67, 1.5)
RECENT_HALFLIFE_Y = 3.0
MIN_SEASON_ROWS = 30
SEASONS = ['winter', 'pre_monsoon', 'sw_monsoon', 'ne_monsoon']


# ── frame: target, persistence, causal features ─────────────────────────────
def cell_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per origin t: target (t+1..t+24 total), persistence (t-23..t), hourly targets y_h1..y_h24 and features
    built from values at or before t only."""
    p = frame['precipitation_mm'].astype(float)
    f = build_features(p)                                                   # production features (origin hour/month)
    wet = (p >= WET_MM).astype(float).where(p.notna())
    idx = np.arange(len(p), dtype=float)
    # hours since the last wet hour (capped at 720 h); unknown after missing data / at the series start until the
    # next wet hour or until 720 valid hours have passed
    lw = pd.Series(np.where(wet.to_numpy() == 1, idx, np.nan)).ffill().to_numpy()
    lu = pd.Series(np.where(p.isna().to_numpy(), idx, np.nan)).ffill().fillna(-1.0).to_numpy()
    known = np.nan_to_num(lw, nan=-np.inf) > lu
    hours_since_wet = np.where(known, idx - np.nan_to_num(lw, nan=0.0), np.where(idx - lu > 720, 720.0, np.nan))
    rh = frame['relative_humidity_2m_pct'].astype(float)
    t = frame['temperature_2m_c'].astype(float)
    d = np.deg2rad(frame['wind_direction_10m_deg'].to_numpy(dtype=float))
    sp = frame['wind_speed_10m_kmh'].to_numpy(dtype=float)
    u, v = pd.Series(-sp * np.sin(d), index=p.index), pd.Series(-sp * np.cos(d), index=p.index)
    mid = (p.index + pd.Timedelta(hours=12)).tz_convert('Asia/Kolkata')
    season = np.array([SEASON_OF_MONTH[m] for m in mid.month])
    org = p.index.tz_convert('Asia/Kolkata')
    ext = pd.DataFrame({
        'log_rain_72h': np.log1p(p.rolling(72, min_periods=72).sum()).to_numpy(),
        'log_rain_7d': np.log1p(p.rolling(168, min_periods=168).sum()).to_numpy(),
        'wet_frac_72h': wet.rolling(72, min_periods=72).mean().to_numpy(),
        'wet_frac_7d': wet.rolling(168, min_periods=168).mean().to_numpy(),
        'log_hours_since_wet': np.log1p(np.minimum(hours_since_wet, 720.0)),
        'rh_now': rh.to_numpy(), 'rh_mean_24h': rh.rolling(24, min_periods=24).mean().to_numpy(),
        'rh_change_24h': rh.diff(24).to_numpy(),
        'wind_u': u.to_numpy(), 'wind_v': v.to_numpy(),
        'wind_u_24h': u.rolling(24, min_periods=24).mean().to_numpy(),
        'wind_v_24h': v.rolling(24, min_periods=24).mean().to_numpy(),
        't_dev_24h': (t - t.rolling(24, min_periods=24).mean()).to_numpy(),
        't_range_24h': (t.rolling(24, min_periods=24).max() - t.rolling(24, min_periods=24).min()).to_numpy(),
        'is_sw': (season == 'sw_monsoon').astype(float), 'is_ne': (season == 'ne_monsoon').astype(float),
        'is_pre': (season == 'pre_monsoon').astype(float),
    }, index=p.index)
    out = pd.concat([f, ext], axis=1)
    out['y'] = p.rolling(24, min_periods=24).sum().shift(-24).to_numpy()   # t+1 .. t+24
    out['persistence'] = p.rolling(24, min_periods=24).sum().to_numpy()    # t-23 .. t
    for h in HOURS:
        out[f'y_h{h}'] = p.shift(-h).to_numpy()
    out['season'], out['origin_month'] = season, org.month
    out['day'], out['year'] = org.strftime('%Y-%m-%d'), p.index.year
    out['origin_utc_hour'] = p.index.hour
    return out


def build_base(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for cell in sorted(frames):
        f = cell_features(frames[cell])
        f['cell'] = cell
        parts.append(f)
    df = pd.concat(parts)
    df.index.name = 'time_utc'
    need = PROD_FEATURES + [c for c in EXT_RAW] + ['y', 'persistence'] + [f'y_h{h}' for h in HOURS]
    return df.dropna(subset=need)


def make_split(base: pd.DataFrame, splits: Dict[str, tuple]) -> pd.DataFrame:
    """Rows whose origin AND window end (t+24) lie in one period; training-only (leave-one-out) climatology features."""
    t0 = base.index
    t1 = base.index + pd.Timedelta(hours=24)
    split = np.full(len(base), None, dtype=object)
    for sp, (lo, hi) in splits.items():
        lo, hi = pd.Timestamp(lo, tz='UTC'), pd.Timestamp(hi, tz='UTC')
        split[np.asarray((t0 >= lo) & (t1 <= hi))] = sp
    df = base[split != None].copy()                                              # noqa: E711
    df['split'] = split[split != None]                                           # noqa: E711
    for cell in sorted(df['cell'].unique()):
        df[f'is_{cell}'] = (df['cell'] == cell).astype(float)
    ev = (df['y'].to_numpy() >= EVENT_MM).astype(float)
    df['state'] = (df['persistence'].to_numpy() >= EVENT_MM).astype(int)
    df['logit_pclim25'] = tv._logit_clip(tv.train_encoding(df, ev, [['origin_month'], []], per_cell=True))
    df['logit_pcond25'] = tv._logit_clip(tv.train_encoding(df, ev, [['state', 'origin_month'], ['state'], []], per_cell=False))
    df['log_clim_mean'] = np.log1p(tv.train_encoding(df, df['y'].to_numpy(), [['origin_month'], []], per_cell=True))
    return df


def feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    cells = [c for c in df.columns if c.startswith('is_gc')]
    return {'prod': PROD_FEATURES + cells, 'ext': PROD_FEATURES + cells + EXT_RAW + CLIM_FEATS}


# ── distribution helpers ────────────────────────────────────────────────────
def crps_from_quantiles(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS approximated by the mean quantile score at TAUS (exact |q - y| for a point forecast)."""
    y = np.asarray(y, dtype=float)[:, None]
    return 2.0 * np.mean(((y < Q).astype(float) - TAUS[None, :]) * (Q - y), axis=1)


def _std_fit(X: np.ndarray):
    m, s = X.mean(axis=0), X.std(axis=0)
    return m, np.where(s > 0, s, 1.0)


def _logistic(X, o, w=None):
    from sklearn.linear_model import LogisticRegression
    o = np.asarray(o, dtype=int)
    if o.min() == o.max():
        return ('const', float(o.mean()))
    return LogisticRegression(C=LOGIT_C, max_iter=2000).fit(X, o, sample_weight=w)


def _logistic_p(model, X):
    if isinstance(model, tuple):
        return np.full(len(X), model[1])
    return model.predict_proba(X)[:, 1]


class RatioDistribution:
    """Predictive distribution = mean x empirical training ratio y/mean, by season (fallback: all seasons)."""

    def fit(self, mu, y, season):
        mu, y, season = np.asarray(mu, float), np.asarray(y, float), np.asarray(season)
        ok = mu > 0
        self.tables = {}
        for s in [None] + SEASONS:
            m = ok if s is None else ok & (season == s)
            if m.sum() >= MIN_SEASON_ROWS or s is None:
                r = np.sort(y[m] / mu[m]) if m.any() else np.zeros(1)
                self.tables[s] = {'sorted': r, 'q': np.quantile(r, TAUS), 'q50': float(np.quantile(r, 0.5)), 'mean': float(r.mean())}
        return self

    def _t(self, s):
        return self.tables.get(s, self.tables[None])

    def predict(self, mu, season, quantiles=True):
        mu, season = np.maximum(np.asarray(mu, float), 0.0), np.asarray(season)
        n = len(mu)
        out = {'mean': np.zeros(n), 'median': np.zeros(n), 'p25': np.zeros(n)}
        if quantiles:
            out['Q'] = np.zeros((n, len(TAUS)))
        for s in pd.unique(season):
            m = season == s
            t = self._t(s)
            out['mean'][m] = mu[m] * t['mean']
            out['median'][m] = mu[m] * t['q50']
            with np.errstate(divide='ignore'):
                c = np.where(mu[m] > 0, EVENT_MM / np.where(mu[m] > 0, mu[m], 1.0), np.inf)
            out['p25'][m] = 1.0 - np.searchsorted(t['sorted'], c, side='left') / len(t['sorted'])
            if quantiles:
                out['Q'][m] = mu[m][:, None] * t['q'][None, :]
        return out


# ── candidate models ────────────────────────────────────────────────────────
class A0Existing:
    """Existing production-input two-part 24 h-total model (zero-inflated seasonal gamma above 0.1 mm)."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, tr):
        self.m = TwoPartRainModel(tv.RAIN_OM_ACCUM_KEY, feature_names=self.feats).fit(tr[self.feats], tr['y'].to_numpy(),
                                                                                    tr['season'].to_numpy())
        return self

    def predict(self, df, quantiles=True):
        pr = self.m.predict(df[self.feats], df['season'].to_numpy())
        p, a, s = pr['p_wet'], pr['shape'], pr['scale']
        with np.errstate(invalid='ignore'):          # no gamma fitted (all-dry training) -> no amounts
            out = {k: np.nan_to_num(v, nan=0.0) for k, v in (('mean', mixture_mean(p, a, s)), ('median', mixture_median(p, a, s)),
                                                              ('p25', prob_exceed(p, a, s, EVENT_MM)))}
        if quantiles:
            tau = TAUS[None, :]
            pp = p[:, None]
            u = np.clip((tau - (1 - pp)) / np.where(pp > 0, pp, 1.0), 0, 1 - 1e-12)
            with np.errstate(invalid='ignore'):
                g = WET_MM + stats.gamma.ppf(u, a[:, None], scale=s[:, None])
            out['Q'] = np.maximum.accumulate(np.where(tau <= 1 - pp, 0.0, np.nan_to_num(g, nan=0.0)), axis=1)
        return out


class DirectTweedie:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, tr):
        from sklearn.linear_model import TweedieRegressor
        X = tr[self.feats].to_numpy(float)
        self.mu_, self.sd_ = _std_fit(X)
        y = tr['y'].to_numpy(float)
        self.zero = bool((y <= 0).all())
        if not self.zero:
            self.glm = TweedieRegressor(power=1.5, link='log', alpha=GLM_ALPHA, max_iter=1000).fit((X - self.mu_) / self.sd_, y)
            self.dist = RatioDistribution().fit(self._mean(tr), y, tr['season'].to_numpy())
        return self

    def _mean(self, df):
        return self.glm.predict((df[self.feats].to_numpy(float) - self.mu_) / self.sd_)

    def predict(self, df, quantiles=True):
        if self.zero:
            n = len(df)
            return {'mean': np.zeros(n), 'median': np.zeros(n), 'p25': np.zeros(n), **({'Q': np.zeros((n, len(TAUS)))} if quantiles else {})}
        return self.dist.predict(self._mean(df), df['season'].to_numpy(), quantiles)


class HurdleTotal:
    """P(>= 0.1), P(>= 2.5) logistic; trace amounts in [0.1, 2.5) empirical by season; amounts >= 2.5 mm =
    2.4 + Gamma with log-linear mean (gamma regression) and seasonal shape. Optional recency weights and a
    training-fold-only amount factor (fitted on the last training year with a model fitted on earlier years)."""

    def __init__(self, feats, recent: bool = False, calibrate: bool = False):
        self.feats, self.recent, self.calibrate = feats, recent, calibrate
        self.factor = {s: 1.0 for s in SEASONS}

    def _weights(self, tr):
        if not self.recent:
            return None
        return 0.5 ** ((tr['year'].max() - tr['year'].to_numpy()) / RECENT_HALFLIFE_Y)

    def _fit_core(self, tr):
        X = tr[self.feats].to_numpy(float)
        self.mu_, self.sd_ = _std_fit(X)
        Xs = (X - self.mu_) / self.sd_
        y, season, w = tr['y'].to_numpy(float), tr['season'].to_numpy(), self._weights(tr)
        self.l1 = _logistic(Xs, y >= WET_MM, w)
        self.l2 = _logistic(Xs, y >= EVENT_MM, w)
        big = y >= EVENT_MM
        self.trace, self.gam, self.shape = {}, None, {}
        for s in [None] + SEASONS:
            m = (y >= WET_MM) & (y < EVENT_MM) & ((season == s) if s is not None else True)
            if m.sum() >= MIN_SEASON_ROWS or s is None:
                v = y[m] if m.any() else np.array([WET_MM])
                self.trace[s] = {'q': np.quantile(v, np.linspace(0, 1, 201)), 'mean': float(v.mean())}
        if big.sum() >= 2:
            from sklearn.linear_model import GammaRegressor
            self.gam = GammaRegressor(alpha=GLM_ALPHA, max_iter=1000).fit(Xs[big], y[big] - EXCESS_LOC,
                                                                           sample_weight=None if w is None else w[big])
            r = (y[big] - EXCESS_LOC) / self.gam.predict(Xs[big])
            for s in [None] + SEASONS:
                rs = r if s is None else r[season[big] == s]
                if len(rs) >= MIN_SEASON_ROWS or s is None:
                    c, var = float(rs.mean()), float(rs.var())
                    self.shape[s] = (c, c * c / var if var > 0 else 1.0)       # mean factor, gamma shape
        return self

    def fit(self, tr):
        if self.calibrate:
            last = tr['year'].max()
            inner, hold = tr[tr['year'] < last], tr[tr['year'] == last]
            if len(inner) and len(hold):
                core = HurdleTotal(self.feats, recent=self.recent)._fit_core(inner)
                pr = core._parts(hold)
                y, season = hold['y'].to_numpy(float), hold['season'].to_numpy()
                for s in SEASONS:
                    m = season == s
                    if (y[m] >= EVENT_MM).sum() < MIN_SEASON_ROWS:
                        continue
                    num = y[m].sum() - ((pr['p1'] - pr['p2']) * pr['tmean'])[m].sum() - (pr['p2'][m] * EXCESS_LOC).sum()
                    den = (pr['p2'] * pr['mexc'])[m].sum()
                    if den > 0:
                        self.factor[s] = float(np.clip(num / den, *CAL_CLIP))
        return self._fit_core(tr)

    def _parts(self, df):
        Xs = (df[self.feats].to_numpy(float) - self.mu_) / self.sd_
        season = df['season'].to_numpy()
        p1 = _logistic_p(self.l1, Xs)
        p2 = np.minimum(_logistic_p(self.l2, Xs), p1)
        base = self.gam.predict(Xs) if self.gam is not None else np.full(len(df), np.nan)
        c = np.array([self.shape.get(s, self.shape.get(None, (1.0, 1.0)))[0] for s in season]) if self.shape else 1.0
        k = np.array([self.shape.get(s, self.shape.get(None, (1.0, 1.0)))[1] for s in season]) if self.shape else np.ones(len(df))
        fac = np.array([self.factor.get(s, 1.0) for s in season])
        mexc = base * c * fac
        tmean = np.array([self.trace.get(s, self.trace[None])['mean'] for s in season])
        return {'p1': p1, 'p2': p2, 'mexc': mexc, 'k': k, 'tmean': tmean, 'season': season}

    def predict(self, df, quantiles=True):
        pr = self._parts(df)
        p1, p2, mexc, k = pr['p1'], pr['p2'], pr['mexc'], pr['k']
        mexc_safe = np.nan_to_num(mexc, nan=0.0)
        out = {'p25': p2, 'mean': (p1 - p2) * pr['tmean'] + p2 * (EXCESS_LOC + mexc_safe)}
        taus = TAUS if quantiles else np.array([0.5])
        Q = self._quantiles(pr, taus)
        out['median'] = Q[:, 0] if not quantiles else self._quantiles(pr, np.array([0.5]))[:, 0]
        if quantiles:
            out['Q'] = Q
        return out

    def _quantiles(self, pr, taus):
        p1, p2, mexc, k, season = pr['p1'][:, None], pr['p2'][:, None], pr['mexc'][:, None], pr['k'][:, None], pr['season']
        tau = taus[None, :]
        Q = np.zeros((len(season), len(taus)))
        trace_q = np.zeros_like(Q)
        u_tr = np.clip((tau - (1 - p1)) / np.where(p1 - p2 > 0, p1 - p2, 1.0), 0, 1)
        for s in pd.unique(season):
            m = season == s
            grid = self.trace.get(s, self.trace[None])['q']
            trace_q[m] = np.interp(u_tr[m], np.linspace(0, 1, 201), grid)
        u_big = np.clip((tau - (1 - p2)) / np.where(p2 > 0, p2, 1.0), 0, 1 - 1e-12)
        with np.errstate(invalid='ignore', divide='ignore'):
            big_q = EXCESS_LOC + stats.gamma.ppf(u_big, k, scale=np.where(k > 0, mexc / k, np.nan))
        big_q = np.nan_to_num(big_q, nan=EVENT_MM)
        Q = np.where(tau <= 1 - p1, 0.0, np.where(tau <= 1 - p2, trace_q, big_q))
        return np.maximum.accumulate(np.maximum(Q, 0.0), axis=1)        # monotone at the trace/gamma junction


class AggregatedHourly:
    """Mean total = sum of 24 hourly two-part forecast means (leads 1..24); distribution via training ratios."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, tr):
        self.models = {h: TwoPartRainModel(f'{h}h', feature_names=self.feats).fit(tr[self.feats], tr[f'y_h{h}'].to_numpy(),
                                                                                   tr['season'].to_numpy()) for h in HOURS}
        self.dist = RatioDistribution().fit(self._mean(tr), tr['y'].to_numpy(float), tr['season'].to_numpy())
        return self

    def _mean(self, df):
        X, s = df[self.feats], df['season'].to_numpy()
        return np.sum([np.nan_to_num(self.models[h].predict(X, s)['mean']) for h in HOURS], axis=0)

    def predict(self, df, quantiles=True):
        return self.dist.predict(self._mean(df), df['season'].to_numpy(), quantiles)


def make_candidate(name: str, fs: Dict[str, List[str]]):
    return {'A0': lambda: A0Existing(fs['prod']), 'B1': lambda: DirectTweedie(fs['prod']), 'B2': lambda: DirectTweedie(fs['ext']),
            'C1': lambda: HurdleTotal(fs['prod']), 'C2': lambda: HurdleTotal(fs['ext']),
            'C2_cal': lambda: HurdleTotal(fs['ext'], calibrate=True), 'C2_recent': lambda: HurdleTotal(fs['ext'], recent=True),
            'D1': lambda: AggregatedHourly(fs['prod'])}[name]()


def n_features(name: str, fs) -> int:
    return len(fs['prod'] if name in ('A0', 'B1', 'C1', 'D1') else fs['ext'])


# ── baselines (training rows only) ──────────────────────────────────────────
def baselines(df: pd.DataFrame, ev: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Climatological and conditional-climatological distributions (cell x origin month; cell x state x origin
    month, fallback cell x origin month) from TRAINING rows, and persistence, for the `ev` rows."""
    tr = df[df['split'] == 'train']
    e = df[df['split'] == ev]

    def table(keys):
        g = tr.groupby(keys)['y']
        q = g.quantile(list(TAUS)).unstack()
        return {'Q': q, 'mean': g.mean(), 'p25': g.apply(lambda v: float((v >= EVENT_MM).mean())),
                'median': g.median(), 'n': g.size()}

    out = {}
    clim = table(['cell', 'origin_month'])
    cond = table(['cell', 'state', 'origin_month'])
    kc = pd.MultiIndex.from_arrays([e['cell'], e['origin_month']])
    kd = pd.MultiIndex.from_arrays([e['cell'], e['state'], e['origin_month']])
    for name, t, k in (('climatology', clim, kc), ('conditional_climatology', cond, kd)):
        Q = t['Q'].reindex(k).to_numpy()
        mean, p25, med = (t[c].reindex(k).to_numpy() for c in ('mean', 'p25', 'median'))
        n = t['n'].reindex(k).to_numpy()
        if name == 'conditional_climatology':                                 # sparse/missing group -> monthly clim
            bad = ~(n >= 20)
            Q[bad], mean[bad], p25[bad], med[bad] = (out['climatology'][c][bad] for c in ('Q', 'mean', 'p25', 'median'))
        out[name] = {'Q': Q, 'mean': mean, 'p25': p25, 'median': med}
    pers = e['persistence'].to_numpy(float)
    out['persistence'] = {'Q': np.repeat(pers[:, None], len(TAUS), axis=1), 'mean': pers,
                          'p25': (pers >= EVENT_MM).astype(float), 'median': pers}
    return out


# ── scoring ─────────────────────────────────────────────────────────────────
def row_losses(e: pd.DataFrame, pred: Dict[str, np.ndarray], base: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    y = e['y'].to_numpy(float)
    o = (y >= EVENT_MM).astype(float)
    L = {'crps': {'model': crps_from_quantiles(pred['Q'], y)}, 'brier': {'model': (pred['p25'] - o) ** 2},
         'se': {'model': (pred['mean'] - y) ** 2}, 'ae': {'model': np.abs(pred['median'] - y)}}
    for b, v in base.items():
        L['crps'][b] = crps_from_quantiles(v['Q'], y)
        L['brier'][b] = (v['p25'] - o) ** 2
        L['se'][b] = (v['mean'] - y) ** 2
        L['ae'][b] = np.abs(v['median'] - y)
    L['ae']['zero'] = np.abs(y)
    return L


def _skill(L, metric, m, blocks, with_ci):
    refs = {k: v[m].mean() for k, v in L[metric].items() if k != 'model' and not (metric != 'ae' and k == 'zero')}
    best = min(refs, key=refs.get)
    sk = 1.0 - L[metric]['model'][m].mean() / refs[best] if refs[best] > 0 else np.nan
    lo = hi = np.nan
    if with_ci:
        lo, hi = tv.day_block_ci(blocks[m], L[metric]['model'][m], L[metric][best][m])
    return sk, best, lo, hi


def summarize(e: pd.DataFrame, pred: Dict[str, np.ndarray], L: Dict[str, Any], thr: float, mask: np.ndarray,
              with_ci: bool, a0_crps: Optional[np.ndarray] = None) -> Dict[str, Any]:
    m = mask
    y = e['y'].to_numpy(float)
    o = y >= EVENT_MM
    blocks = lh._weeks(e['day'].to_numpy())
    rec: Dict[str, Any] = {'n': int(m.sum()), 'n_days': int(e.loc[m, 'day'].nunique()), 'events': int(o[m].sum()),
                           'event_days': int(e.loc[m & o, 'day'].nunique()),
                           'heavy_days': int(e.loc[m & (y >= HEAVY_MM), 'day'].nunique())}
    if not m.any():
        return rec
    for metric, name in (('crps', 'crpss'), ('brier', 'bss'), ('se', 'mse_skill'), ('ae', 'mae_skill')):
        sk, best, lo, hi = _skill(L, metric, m, blocks, with_ci and metric != 'ae')
        rec.update({name: sk, f'{name}_best_baseline': best, f'{name}_ci_low': lo, f'{name}_ci_high': hi})
    rec['crps_model'] = float(L['crps']['model'][m].mean())
    for b in ('climatology', 'conditional_climatology', 'persistence'):
        rec[f'crps_{b}'] = float(L['crps'][b][m].mean())
    if a0_crps is not None:
        rec['crps_improvement_vs_A0'] = 1.0 - L['crps']['model'][m].mean() / a0_crps[m].mean()
        lo, hi = tv.day_block_ci(blocks[m], L['crps']['model'][m], a0_crps[m]) if with_ci else (np.nan, np.nan)
        rec['crps_improvement_vs_A0_ci_low'], rec['crps_improvement_vs_A0_ci_high'] = lo, hi
    rec['mae_median'] = float(L['ae']['model'][m].mean())
    rec['mae_best_point_baseline'] = float(min(L['ae'][k][m].mean() for k in L['ae'] if k != 'model'))
    rec['rmse_mean'] = float(np.sqrt(L['se']['model'][m].mean()))
    rec['rmse_best_baseline'] = float(np.sqrt(min(L['se'][k][m].mean() for k in L['se'] if k != 'model')))
    rec['mean_pred_mm'], rec['mean_obs_mm'] = float(pred['mean'][m].mean()), float(y[m].mean())
    rec['bias_mm'] = rec['mean_pred_mm'] - rec['mean_obs_mm']
    rec['ratio_pred_obs'] = rec['mean_pred_mm'] / rec['mean_obs_mm'] if rec['mean_obs_mm'] > 0 else np.nan
    rec['brier_model'] = float(L['brier']['model'][m].mean())
    rec['mean_p25'], rec['obs_freq25'] = float(pred['p25'][m].mean()), float(o[m].mean())
    f = pred['p25'][m] >= thr
    oo = o[m]
    hits, fa, miss = int((f & oo).sum()), int((f & ~oo).sum()), int((~f & oo).sum())
    rec.update(threshold=thr, hits=hits, false_alarms=fa, misses=miss,
               pod=hits / (hits + miss) if hits + miss else np.nan, far=fa / (hits + fa) if hits + fa else np.nan,
               csi=hits / (hits + fa + miss) if hits + fa + miss else np.nan)
    q10, q90 = pred['Q'][m][:, 4:6].mean(axis=1), pred['Q'][m][:, 44:46].mean(axis=1)
    rec['coverage_10_90'] = float(((y[m] >= q10) & (y[m] <= q90)).mean())
    rec['frac_below_median'] = float((y[m] < pred['median'][m]).mean())
    return rec


def groups(e: pd.DataFrame) -> Dict[str, np.ndarray]:
    y, pers = e['y'].to_numpy(float), e['persistence'].to_numpy(float)
    g = {'ALL': np.ones(len(e), bool)}
    g.update({f'season={s}': (e['season'] == s).to_numpy() for s in SEASONS})
    g.update({f'cell={c}': (e['cell'] == c).to_numpy() for c in sorted(e['cell'].unique())})
    g.update({'observed=dry(<0.1)': y < WET_MM, 'observed=light(0.1-2.5)': (y >= WET_MM) & (y < EVENT_MM),
              'observed=rainy(>=2.5)': y >= EVENT_MM, 'antecedent=dry_24h': pers < WET_MM, 'antecedent=wet_24h': pers >= WET_MM,
              'daily_origin_18utc': (e['origin_utc_hour'] == 18).to_numpy()})
    return g


CI_GROUPS = ('ALL', 'season=sw_monsoon', 'season=ne_monsoon', 'season=winter', 'season=pre_monsoon')


def evaluate(df: pd.DataFrame, ev: str, models: Dict[str, Any], detail: bool) -> Dict[str, Any]:
    """Score fitted candidates on the `ev` rows of df (identical rows and baselines for all)."""
    e = df[df['split'] == ev]
    tr = df[df['split'] == 'train']
    base = baselines(df, ev)
    preds, thr, L = {}, {}, {}
    for name, mdl in models.items():
        preds[name] = mdl.predict(e)
        thr[name] = tv._choose_prob_threshold(mdl.predict(tr, quantiles=False)['p25'], (tr['y'].to_numpy() >= EVENT_MM).astype(float))
        L[name] = row_losses(e, preds[name], base)
    a0 = L['A0']['crps']['model'] if 'A0' in L else None
    rows, rel = [], []
    gs = groups(e) if detail else {'ALL': np.ones(len(e), bool)}
    for name in models:
        for gname, m in gs.items():
            rows.append({'candidate': name, 'group': gname,
                         **summarize(e, preds[name], L[name], thr[name], m, gname in CI_GROUPS, None if name == 'A0' else a0)})
        if detail:
            p, o = preds[name]['p25'], (e['y'].to_numpy() >= EVENT_MM)
            bins = np.clip((p * 10).astype(int), 0, 9)
            for b in range(10):
                mb = bins == b
                if mb.any():
                    rel.append({'candidate': name, 'bin': f'{b / 10:.1f}-{(b + 1) / 10:.1f}', 'n': int(mb.sum()),
                                'mean_p25': float(p[mb].mean()), 'obs_freq25': float(o[mb].mean())})
    return {'rows': pd.DataFrame(rows), 'reliability': pd.DataFrame(rel)}


# ── rules ───────────────────────────────────────────────────────────────────
def select(cv: pd.DataFrame, nfeat: Dict[str, int]) -> Dict[str, Any]:
    piv = {k: cv.pivot(index='fold', columns='candidate', values=k) for k in ('crpss', 'bss', 'mae_median')}
    a0 = {k: v['A0'] for k, v in piv.items()}
    info = {}
    for c in piv['crpss'].columns:
        if c == 'A0':
            continue
        cr = piv['crpss'][c]
        checks = {'crpss_positive_3of4': int((cr > 0).sum()) >= 3,
                  'beats_A0_3of4_and_mean': int((cr > a0['crpss']).sum()) >= 3 and cr.mean() > a0['crpss'].mean(),
                  'bss_mean_ge_A0': piv['bss'][c].mean() >= a0['bss'].mean(),
                  'mae_mean_le_A0': piv['mae_median'][c].mean() <= a0['mae_median'].mean()}
        info[c] = {'mean_crpss': float(cr.mean()), 'positive_folds': int((cr > 0).sum()),
                   'folds_beating_A0': int((cr > a0['crpss']).sum()), 'mean_bss': float(piv['bss'][c].mean()),
                   'mean_mae_median': float(piv['mae_median'][c].mean()), 'eligible': all(checks.values()), **checks}
    eligible = [c for c in info if info[c]['eligible']]
    pool = eligible or list(info)
    best = sorted(pool, key=lambda c: (-info[c]['mean_crpss'], nfeat[c]))[0]
    return {'selected': best if eligible else None, 'best_mean_if_none': None if eligible else best, 'candidates': info,
            'A0': {'mean_crpss': float(a0['crpss'].mean()), 'mean_bss': float(a0['bss'].mean()),
                   'mean_mae_median': float(a0['mae_median'].mean()), 'positive_folds': int((a0['crpss'] > 0).sum())}}


def validation_gate(rows: pd.DataFrame, cand: str) -> Dict[str, Any]:
    r = rows[rows['candidate'] == cand].set_index('group')
    a = r.loc['ALL']
    seas = {s: float(r.loc[f'season={s}', 'crpss']) for s in SEASONS if f'season={s}' in r.index}
    cells = r[r.index.str.startswith('cell=')]['crpss']
    checks = {
        'crpss_positive_ci': bool(a['crpss'] > 0 and np.isfinite(a['crpss_ci_low']) and a['crpss_ci_low'] > 0),
        'improves_on_A0_ci': bool(a.get('crps_improvement_vs_A0', np.nan) > 0 and a.get('crps_improvement_vs_A0_ci_low', np.nan) > 0),
        'bss_positive': bool(a['bss'] > 0),
        'total_accuracy': bool(a['mse_skill'] > 0 and 0.8 <= a['ratio_pred_obs'] <= 1.25),
        'no_hidden_failure': bool(seas.get('sw_monsoon', -1) > 0 and seas.get('ne_monsoon', -1) > 0
                                  and min(seas.values()) >= -0.02 and int((cells > 0).sum()) >= 6),
        'enough_event_days': bool(a['event_days'] >= 100)}
    return {'passed': all(checks.values()), **checks, 'season_crpss': seas, 'cells_positive': int((cells > 0).sum())}


# ── reproduction of the existing baseline ───────────────────────────────────
def reproduce_existing(frames) -> Dict[str, Any]:
    """Re-score the existing long_horizon_history 24 h-total results (R0 and selected R2) on 2023 with the
    unchanged existing code and compare with the saved summary."""
    saved = pd.read_csv(os.path.join(ROOT, lh.SUMMARY_CSV), float_precision='round_trip')
    saved = saved[saved['horizon'] == tv.RAIN_OM_ACCUM_KEY].set_index('variant')
    df, ph = lh._frames_for('rainfall', frames, tv.RAIN_OM_ACCUM_KEY, lh.VALIDATION)
    out = {}
    for name in saved.index:
        feats = lh._variants('rainfall', df)[name]
        _, _, s, _ = lh.eval_rain(df, ph, tv.RAIN_OM_ACCUM_KEY, feats, 'validate', float(saved.loc[name, 'cv_p_star_from_folds']))
        out[name] = {'saved_val_bss_ge2.5': float(saved.loc[name, 'val_skill']), 'reproduced_val_bss_ge2.5': s['skill'],
                     'saved_crpss_clim': float(saved.loc[name, 'val_crpss_clim']), 'reproduced_crpss_clim': s['crpss_clim'],
                     'match': bool(abs(s['skill'] - saved.loc[name, 'val_skill']) < 1e-9
                                   and abs(s['crpss_clim'] - saved.loc[name, 'val_crpss_clim']) < 1e-9),
                     'mean_pred_mm': s['mean_pred_mm'], 'mean_obs_mm': s['mean_obs_mm']}
    return out


# ── main ────────────────────────────────────────────────────────────────────
def _fit_all(df, names, fs):
    tr = df[df['split'] == 'train']
    return {n: make_candidate(n, fs).fit(tr) for n in names}


def run_experiment(save: bool = True, install_guard: bool = True) -> Dict[str, Any]:
    from src.collectors import open_meteo
    from src.collectors.open_meteo_history import HISTORY_MANIFEST, UNSEAL_ENV, load_history
    if os.environ.get(UNSEAL_ENV):
        raise RuntimeError(f'{UNSEAL_ENV} is set - refusing to run (2024 must stay sealed)')
    crit = os.path.join(ROOT, CRITERIA)
    if not os.path.exists(crit):
        raise FileNotFoundError('pre-declared criteria missing')
    if install_guard:
        sys.addaudithook(lh.sealed_open_guard)
    started = pd.Timestamp.now(tz='UTC').isoformat()
    dev = load_history(include_sealed=False)
    yrs = sorted(dev['time_utc'].dt.year.unique().tolist())
    if yrs != list(range(2017, 2024)):
        raise RuntimeError(f'unexpected development years {yrs}')
    cells = lh.grid_cells(dev)
    frames = lh.cell_frames(cells, dev)

    repro = reproduce_existing(frames)
    print('reproduction:', json.dumps(repro, default=str), flush=True)

    base = build_base(frames)
    cv_rows, cv_groups = [], []
    for fold, splits in lh.FOLDS.items():
        df = make_split(base, splits)
        fs = feature_sets(df)
        models = _fit_all(df, CANDIDATES, fs)
        res = evaluate(df, 'validate', models, detail=True)
        g = res['rows'].assign(fold=fold)
        cv_groups.append(g)
        cv_rows.append(g[g['group'] == 'ALL'])
        print(f'{fold} done', flush=True)
    cv = pd.concat(cv_rows, ignore_index=True)
    fs = feature_sets(make_split(base, lh.VALIDATION))
    sel = select(cv, {c: n_features(c, fs) for c in CANDIDATES})
    frozen = pd.Timestamp.now(tz='UTC').isoformat()
    winner = sel['selected'] or sel['best_mean_if_none']
    names = ['A0', winner]

    dfv = make_split(base, lh.VALIDATION)
    val = evaluate(dfv, 'validate', _fit_all(dfv, names, feature_sets(dfv)), detail=True)
    gate = validation_gate(val['rows'], winner)
    gate['passed'] = bool(gate['passed'] and sel['selected'] is not None)

    local = open_meteo.load_local()['hourly']
    rob_base = build_base(lh.cell_frames(cells, dev, local))
    dfr = make_split(rob_base, lh.ROBUST)
    refit = _fit_all(dfr, names, feature_sets(dfr))
    rob = evaluate(dfr, 'robust', refit, detail=True)
    ra = rob['rows'][(rob['rows']['candidate'] == winner) & (rob['rows']['group'] == 'ALL')].iloc[0]
    recommend = bool(gate['passed'] and ra['crpss'] > 0 and ra['bss'] > 0)

    with open(os.path.join(ROOT, HISTORY_MANIFEST), encoding='utf-8') as f:
        man = json.load(f)
    meta = {'started_utc': started, 'selection_frozen_utc': frozen, 'finished_utc': pd.Timestamp.now(tz='UTC').isoformat(),
            'criteria_file': CRITERIA, 'criteria_sha256': lh._sha256(crit), 'dev_sha256_manifest': man['files']['dev']['sha256'],
            'dev_years': yrs, 'sealed_file_opened': False, 'n_grid_cells': int(len(cells)),
            'robustness_source': 'data/external/open_meteo_chennai_hourly_2025_2026.csv (PREVIOUSLY EXAMINED; secondary only)',
            'robustness_file_sha256': lh._sha256(open_meteo.LOCAL_FILE),
            'target': 'sum of hourly precipitation over t+1..t+24 (mm); persistence = sum over t-23..t',
            'rows': {'validation_2023': int((dfv['split'] == 'validate').sum()), 'robust': int((dfr['split'] == 'robust').sum())}}
    summary = {'reproduction': repro, 'selection': sel, 'validated_candidate': winner, 'validation_gate': gate,
               'robust_all': ra.to_dict(), 'recommend_final_evaluation_review': recommend, 'run_meta': meta}
    result = {'cv': cv, 'cv_groups': pd.concat(cv_groups, ignore_index=True), 'val': val, 'rob': rob, 'summary': summary,
              'models': refit}
    if save:
        _save(result)
    return result


def _save(r):
    prod = {os.path.normpath(p) for p in tv.PRODUCTION_ARTIFACTS} | {os.path.normpath(p) for p in lh.OUTPUTS}
    assert not {os.path.normpath(p) for p in OUTPUTS} & prod
    os.makedirs(os.path.join(ROOT, OUT_MODELS), exist_ok=True)
    os.makedirs(os.path.join(ROOT, OUT_RESULTS), exist_ok=True)
    joblib.dump({'models': r['models'], 'summary': r['summary'],
                 'status': 'EXPERIMENT - refit on 2017-2023; not promoted; the 24 h total stays unavailable'},
                os.path.join(ROOT, ARTIFACT))
    r['cv'].to_csv(os.path.join(ROOT, CV_CSV), index=False)
    r['cv_groups'].to_csv(os.path.join(ROOT, CV_GROUPS_CSV), index=False)
    r['val']['rows'].to_csv(os.path.join(ROOT, VAL_CSV), index=False)
    r['rob']['rows'].to_csv(os.path.join(ROOT, ROB_CSV), index=False)
    pd.concat([r['val']['reliability'].assign(period='V2023'), r['rob']['reliability'].assign(period='R2025_26')]).to_csv(
        os.path.join(ROOT, REL_CSV), index=False)
    for path, obj in ((REPRO_JSON, r['summary']['reproduction']), (SUMMARY_JSON, r['summary']), (RUN_MANIFEST, r['summary']['run_meta'])):
        with open(os.path.join(ROOT, path), 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    print('Saved:', ', '.join(OUTPUTS[:-2] + (SUMMARY_JSON, RUN_MANIFEST)))


def _f(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return repr(x) if np.isfinite(x) else 'NaN'


def build_report() -> str:
    """report.md from the saved outputs only (no refitting, no new scoring)."""
    rd = lambda p: pd.read_csv(os.path.join(ROOT, p), float_precision='round_trip')   # noqa: E731
    with open(os.path.join(ROOT, SUMMARY_JSON), encoding='utf-8') as f:
        S = json.load(f)
    cv, val, rob, rel = rd(CV_CSV), rd(VAL_CSV), rd(ROB_CSV), rd(REL_CSV)
    meta, sel, gate, w = S['run_meta'], S['selection'], S['validation_gate'], S['validated_candidate']
    L = ['# 24-hour rainfall TOTAL experiment (2017-2023 history)', '',
         'EXPERIMENT ONLY - nothing promoted; the 24 h total stays unavailable in production. Values unrounded.', '',
         '## 1. Target, data and baseline reproduction', '',
         f"- Target: {meta['target']}; per unique grid cell ({meta['n_grid_cells']}), every hourly origin; rows whose window "
         'crosses a period boundary are excluded. Features use values at or before the origin only (tested).',
         f"- Development years {meta['dev_years']} (manifest SHA-256 `{meta['dev_sha256_manifest']}` verified); 2024 never opened.",
         f"- Criteria `{meta['criteria_file']}` SHA-256 `{meta['criteria_sha256']}`, written before scoring; run started "
         f"{meta['started_utc']}, selection frozen {meta['selection_frozen_utc']} (before any 2023 scoring of new candidates).",
         f"- Rows: 2023 validation {meta['rows']['validation_2023']}, 2025-26 secondary {meta['rows']['robust']} (cell-hours).", '',
         'Reproduction of the existing long_horizon_history 24 h-total result on 2023 (existing code, unchanged):', '',
         '| variant | saved Brier skill (>=2.5 mm) | reproduced | saved CRPS skill (lenient) | reproduced | match | mean pred / obs mm |',
         '|---|---|---|---|---|---|---|']
    for k, r in S['reproduction'].items():
        L.append(f"| {k} | {_f(r['saved_val_bss_ge2.5'])} | {_f(r['reproduced_val_bss_ge2.5'])} | {_f(r['saved_crpss_clim'])} | "
                 f"{_f(r['reproduced_crpss_clim'])} | {r['match']} | {_f(r['mean_pred_mm'])} / {_f(r['mean_obs_mm'])} |")
    L += ['', '## 2. Candidates and selection (folds 2019-2022)', '',
          'A0 existing two-part (production inputs) | B1/B2 direct Tweedie (production / extended features) | C1/C2 hurdle '
          '(P>=0.1, P>=2.5, trace amounts, gamma regression >= 2.5 mm) | C2_cal training-fold-only amount correction | '
          'C2_recent recency-weighted | D1 sum of 24 hourly two-part forecasts.', '',
          'CRPS skill vs best of climatological distribution / conditional climatological distribution / persistence:', '',
          '```', cv.pivot(index='candidate', columns='fold', values='crpss').assign(
              mean=lambda d: d.mean(axis=1)).to_string(float_format=_f), '```', '',
          'Brier skill (>= 2.5 mm) and MAE of the median (mm):', '', '```',
          cv.pivot(index='candidate', columns='fold', values='bss').assign(mean=lambda d: d.mean(axis=1)).to_string(float_format=_f),
          '', cv.pivot(index='candidate', columns='fold', values='mae_median').assign(mean=lambda d: d.mean(axis=1)).to_string(float_format=_f),
          '```', '', '| candidate | mean CRPS skill | folds > 0 | folds beating A0 | mean Brier skill | mean MAE | eligible |',
          '|---|---|---|---|---|---|---|',
          f"| A0 | {_f(sel['A0']['mean_crpss'])} | {sel['A0']['positive_folds']} | - | {_f(sel['A0']['mean_bss'])} | "
          f"{_f(sel['A0']['mean_mae_median'])} | reference |"]
    for c, i in sel['candidates'].items():
        L.append(f"| {c} | {_f(i['mean_crpss'])} | {i['positive_folds']} | {i['folds_beating_A0']} | {_f(i['mean_bss'])} | "
                 f"{_f(i['mean_mae_median'])} | {i['eligible']} |")
    L += ['', f"Selected: **{sel['selected']}**" if sel['selected'] else
          f"No eligible candidate; best mean shown for information only: {sel['best_mean_if_none']}", '']
    cols = ['crpss', 'crpss_ci_low', 'crpss_ci_high', 'crps_improvement_vs_A0', 'crps_improvement_vs_A0_ci_low',
            'crps_improvement_vs_A0_ci_high', 'bss', 'bss_ci_low', 'bss_ci_high', 'mse_skill', 'mse_skill_ci_low', 'mse_skill_ci_high',
            'crps_model', 'crps_climatology', 'crps_conditional_climatology', 'crps_persistence', 'mae_median',
            'mae_best_point_baseline', 'rmse_mean', 'rmse_best_baseline', 'mean_pred_mm', 'mean_obs_mm', 'bias_mm', 'ratio_pred_obs',
            'threshold', 'pod', 'far', 'csi', 'hits', 'misses', 'false_alarms', 'coverage_10_90', 'frac_below_median',
            'n', 'n_days', 'events', 'event_days', 'heavy_days']

    def block(rows, title):
        a = rows[rows['group'] == 'ALL'].set_index('candidate')[[c for c in cols if c in rows]].T
        return [title, '', '```', a.to_string(float_format=_f), '```', '']
    L += ['## 3. 2023 validation (scored once; 7-day-block 95 % intervals)', ''] + block(val, 'Pooled, all cells:')
    L += ['Validation checks for ' + w + ':', '', '```', json.dumps(gate, indent=1, default=str), '```', '']
    L += ['## 4. 2023 by unique grid cell, season and wet/dry condition', '']
    g = val[val['group'] != 'ALL'].pivot(index='group', columns='candidate',
                                         values=['crpss', 'bss', 'mae_median', 'ratio_pred_obs', 'misses', 'false_alarms', 'events', 'event_days'])
    L += ['```', g.to_string(float_format=_f), '```', '',
          '(Within an observed dry/light/rainy group the event outcome is constant, so Brier skill there is not meaningful; '
          'CRPS, MAE and the amount ratio are.)', '']
    L += ['## 5. Amount bias and calibration', '', 'Mean predicted / observed total by season (2023):', '', '```',
          val[val['group'].str.startswith('season=')].pivot(index='group', columns='candidate', values='ratio_pred_obs').to_string(float_format=_f),
          '```', '', 'Reliability of P(total >= 2.5 mm):', '', '```',
          rel.pivot_table(index=['period', 'bin'], columns='candidate', values=['mean_p25', 'obs_freq25', 'n']).to_string(float_format=_f),
          '```', '']
    L += ['## 6. Secondary check 2025-01-01..2026-10-07 (PREVIOUSLY EXAMINED data - not an untouched test; '
          'refit on 2017-2023; did not influence selection)', ''] + block(rob, 'Pooled, all cells:')
    gr = rob[rob['group'] != 'ALL'].pivot(index='group', columns='candidate', values=['crpss', 'bss', 'ratio_pred_obs'])
    L += ['```', gr.to_string(float_format=_f), '```', '',
          f"## 7. Outcome", '', f"- Validated candidate: {w}; validation passed: {gate['passed']}.",
          f"- Recommend a separate final-evaluation review: {S['recommend_final_evaluation_review']}.", '']
    rep = '\n'.join(L) + '\n'
    with open(os.path.join(ROOT, REPORT), 'w', encoding='utf-8') as f:
        f.write(rep)
    return rep


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='24 h rainfall total experiment (experiment outputs only)')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--report-only', action='store_true', help='rebuild report.md from saved outputs')
    a = ap.parse_args()
    if a.report_only:
        print(build_report())
    else:
        run_experiment(save=not a.no_save)
        if not a.no_save:
            build_report()
