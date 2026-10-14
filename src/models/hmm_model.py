"""
Hidden Markov Model (HMM) Implementation for UrbanPulse.
Provides:
1. GaussianHMM temporal models for AQI, Rainfall, and Temperature.
2. Training using Baum-Welch (EM) exclusively on train.csv.
3. Hidden state selection via BIC/AIC and validation loss.
4. Multi-step forecasting at supported horizons:
   - AQI: 1h, 3h, 6h, 24h
   - Rainfall: 1h, 3h, 6h, 24h
   - Temperature: 24h (daily regime)
5. Exact predictive distributions: E[Y_{t+h}] and Var(Y_{t+h}).
6. Model persistence under models/hmm/.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from hmmlearn import hmm

class TemporalHMM:
    """
    Gaussian Hidden Markov Model for temporal environmental forecasting at zone level.
    """
    def __init__(
        self,
        n_components: int = 4,
        variable: str = 'AQI',
        zone_id: Optional[int] = None,
        covariance_type: str = 'diag',
        n_iter: int = 200,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.variable = variable
        self.zone_id = zone_id
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            min_covar=1e-3,
            verbose=False
        )
        self.is_fitted = False
        self.state_order_ = None # maps internal state to sorted state by emission mean
        self.means_ = None
        self.variances_ = None
        self.transmat_ = None
        self.startprob_ = None
        self.trained_observations_ = 0
        self.training_time_range_ = None
        self.val_mae_ = None
        self.val_rmse_ = None
        self.bic_ = None
        self.aic_ = None

    def fit(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> 'TemporalHMM':
        """
        Fit GaussianHMM using EM algorithm.
        X: 2D array of shape (N, 1) of valid continuous observations.
        lengths: List of sequence lengths if multiple disjoint chains exist.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Drop NaNs before fitting
        valid_mask = ~np.isnan(X.flatten())
        if lengths is None:
            # Segment into contiguous non-NaN sub-chains of length >= 2
            isnan = np.isnan(X.flatten())
            diff = np.diff(np.concatenate(([1], isnan.astype(int), [1])))
            starts = np.where(diff == -1)[0]
            ends = np.where(diff == 1)[0]
            X_clean = []
            seq_lengths = []
            for s, e in zip(starts, ends):
                if e - s >= 2:
                    X_clean.append(X[s:e])
                    seq_lengths.append(e - s)
            if X_clean:
                X_clean = np.vstack(X_clean)
            else:
                valid_mask = ~np.isnan(X.flatten())
                X_clean = X[valid_mask]
                seq_lengths = [len(X_clean)]
        else:
            # Reconstruct segments without NaNs
            X_clean = []
            seq_lengths = []
            curr = 0
            for l in lengths:
                seg = X[curr : curr + l]
                curr += l
                # extract contiguous non-nan subsegments
                isnan = np.isnan(seg.flatten())
                diff = np.diff(np.concatenate(([1], isnan.astype(int), [1])))
                starts = np.where(diff == -1)[0]
                ends = np.where(diff == 1)[0]
                for s, e in zip(starts, ends):
                    if e - s >= 2: # need at least 2 steps for transition
                        X_clean.append(seg[s:e])
                        seq_lengths.append(e - s)
            if X_clean:
                X_clean = np.vstack(X_clean)
            else:
                valid_mask = ~np.isnan(X.flatten())
                X_clean = X[valid_mask]
                seq_lengths = [len(X_clean)]

        self.model.fit(X_clean, lengths=seq_lengths)
        self.is_fitted = True
        self.trained_observations_ = int(len(X_clean))
        
        # Sort states by emission mean for interpretability (State 0 = Lowest, State K-1 = Highest)
        raw_means = self.model.means_.flatten()
        sort_idx = np.argsort(raw_means)
        
        self.state_order_ = sort_idx
        self.means_ = raw_means[sort_idx]
        
        if self.covariance_type == 'diag':
            self.variances_ = self.model.covars_.flatten()[sort_idx]
        else:
            self.variances_ = np.array([self.model.covars_[i, i] for i in sort_idx])
            
        # Reorder transition matrix and startprob
        self.transmat_ = self.model.transmat_[sort_idx, :][:, sort_idx]
        self.startprob_ = self.model.startprob_[sort_idx]
        
        return self

    def score(self, X: np.ndarray, lengths: Optional[List[int]] = None) -> float:
        """Compute log-likelihood of observation sequence."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        valid = ~np.isnan(X.flatten())
        if valid.sum() == 0:
            return -np.inf
        if lengths is None:
            isnan = np.isnan(X.flatten())
            diff = np.diff(np.concatenate(([1], isnan.astype(int), [1])))
            starts = np.where(diff == -1)[0]
            ends = np.where(diff == 1)[0]
            X_clean = []
            seq_lengths = []
            for s, e in zip(starts, ends):
                if e - s >= 2:
                    X_clean.append(X[s:e])
                    seq_lengths.append(e - s)
            if X_clean:
                X_clean = np.vstack(X_clean)
            else:
                X_clean = X[valid]
                seq_lengths = [len(X_clean)]
        else:
            X_clean = X[valid]
            seq_lengths = lengths
        return float(self.model.score(X_clean, lengths=seq_lengths))

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion (lower is better)."""
        valid = ~np.isnan(X.flatten())
        N = int(valid.sum())
        log_lik = self.score(X)
        K = self.n_components
        # Free parameters: (K-1) startprob + K(K-1) transmat + K means + K vars
        p = (K - 1) + K * (K - 1) + 2 * K
        return -2.0 * log_lik + p * np.log(N)

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion (lower is better)."""
        log_lik = self.score(X)
        K = self.n_components
        p = (K - 1) + K * (K - 1) + 2 * K
        return -2.0 * log_lik + 2.0 * p

    def filter_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        Sequential forward filtering through sequence that may contain NaNs.
        Strict textbook HMM filtering with missing data:
        1. Time update (transition):
           gamma_{t|t-1} = gamma_{t-1} @ A (or startprob for t=0)
        2. Measurement update:
           - If seq[t] is observed (not NaN):
             lik = N(seq[t]; mu, sigma^2)
             gamma_t \\propto gamma_{t|t-1} * lik
           - If seq[t] is NaN (missing):
             gamma_t = gamma_{t|t-1}  (NO measurement update, pure Markov transition)
             NaN remains strictly missing; NO forward-fill, NO zero-fill.
        Returns:
            gamma_history: array of shape (len(seq), n_components) of filtered state beliefs
        """
        n_steps = len(seq)
        if n_steps == 0:
            return np.empty((0, self.n_components))
            
        gamma_hist = np.zeros((n_steps, self.n_components))
        stds = np.sqrt(self.variances_)
        gamma = self.startprob_.copy()
        
        for t in range(n_steps):
            if t > 0:
                gamma_prior = gamma @ self.transmat_
            else:
                gamma_prior = self.startprob_.copy()
                
            val = seq[t]
            if not np.isnan(val):
                diff = (val - self.means_) / stds
                lik = np.exp(-0.5 * diff**2) / (np.sqrt(2.0 * np.pi) * stds)
                numer = gamma_prior * lik
                denom = np.sum(numer)
                if denom > 0:
                    gamma = numer / denom
                else:
                    gamma = gamma_prior
            else:
                # Missing observation: NO measurement update, state transitions naturally
                gamma = gamma_prior
                
            gamma_hist[t] = gamma
            
        return gamma_hist

    def get_filtered_state_distribution(self, seq: np.ndarray) -> np.ndarray:
        """
        Computes the current posterior / filtered state probability gamma_t = P(S_t | Y_{1:t}).
        Properly handles NaNs via sequential time-evolution without fabricating observations.
        """
        if len(seq) == 0:
            return self.startprob_.copy()
        gamma_hist = self.filter_sequence(seq)
        return gamma_hist[-1]

    def filter_and_forecast_sequence(
        self,
        seq: np.ndarray,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast sequential filtering and forecasting across an entire timeline.
        Returns:
            y_hat_series: np.ndarray of shape (len(seq),) containing forecasted values
            var_series: np.ndarray of shape (len(seq),) containing predictive variances
        """
        gamma_hist = self.filter_sequence(seq)
        A_h = np.linalg.matrix_power(self.transmat_, horizon)
        p_future = gamma_hist @ A_h
        
        y_hat = np.sum(p_future * self.means_, axis=1)
        second_moment = np.sum(p_future * (self.variances_ + self.means_**2), axis=1)
        var_pred = np.maximum(1e-4, second_moment - y_hat**2)
        
        if self.variable in ['AQI', 'rainfall']:
            y_hat = np.maximum(0.0, y_hat)
            
        return y_hat, var_pred

    def forecast_accumulation_sequence(self, seq: np.ndarray, n_steps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        Causal forecast of the accumulated total S = Y_{t+1} + ... + Y_{t+n_steps} at every step t
        (vectorised form of src.alerts.alert_engine.compute_24h_rainfall_accumulation):
          E[S]   = sum_k gamma_t A^k mu
          Var(S) = sum_k Var(Y_{t+k}) + 2 sum_{j<k} Cov(Y_{t+j}, Y_{t+k}),
          E[Y_{t+j} Y_{t+k}] = gamma_t A^j diag(mu) A^(k-j) mu.
        Returns (mean_series, var_series), each of shape (len(seq),).
        """
        gamma_hist = self.filter_sequence(seq)
        powers = [np.linalg.matrix_power(self.transmat_, k) for k in range(1, n_steps + 1)]
        p = np.stack([gamma_hist @ P for P in powers], axis=1)              # (T, n_steps, K)
        m_k = p @ self.means_                                               # (T, n_steps)
        var_single = p @ (self.variances_ + self.means_ ** 2) - m_k ** 2
        diag_mu = np.diag(self.means_)
        cov = np.zeros(len(gamma_hist))
        for j in range(n_steps):
            for k in range(j + 1, n_steps):
                w = powers[j] @ diag_mu @ powers[k - j - 1] @ self.means_
                cov += gamma_hist @ w - m_k[:, j] * m_k[:, k]
        mean = m_k.sum(axis=1)
        var = np.maximum(1e-4, var_single.sum(axis=1) + 2.0 * cov)
        if self.variable in ['AQI', 'rainfall']:
            mean = np.maximum(0.0, mean)
        return mean, var

    def forecast(self, seq: np.ndarray, horizon: int = 1) -> Tuple[float, float, np.ndarray]:
        """
        Compute probabilistic forecast at horizon h:
        1. gamma_t = P(S_t | Y_{1:t})
        2. P(S_{t+h} = j | Y_{1:t}) = gamma_t @ (A^h)
        3. E[Y_{t+h}] = sum_j P(S_{t+h} = j) * mu_j
        4. Var(Y_{t+h}) = sum_j P(S_{t+h} = j) * (sigma_j^2 + mu_j^2) - E[Y]^2
        
        Returns: (expected_value, variance, projected_state_probs)
        """
        gamma_t = self.get_filtered_state_distribution(seq)
        A_h = np.linalg.matrix_power(self.transmat_, horizon)
        p_future = gamma_t @ A_h
        
        mean_forecast = float(np.sum(p_future * self.means_))
        second_moment = np.sum(p_future * (self.variances_ + self.means_**2))
        var_forecast = float(max(1e-4, second_moment - mean_forecast**2))
        
        if self.variable in ['AQI', 'rainfall']:
            mean_forecast = max(0.0, mean_forecast)
            
        return mean_forecast, var_forecast, p_future

    def save(self, filepath: str):
        """Save model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved TemporalHMM model for {self.variable} to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TemporalHMM':
        """Load model artifact from disk."""
        return joblib.load(filepath)
