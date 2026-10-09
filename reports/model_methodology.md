# UrbanPulse: Hidden Markov Model (HMM) Methodology & Validation Report
**Project:** Spatio-Temporal Early Warning System for Chennai - AQI, Rainfall & Temperature Forecasting  
**Author:** Lead ML / Data-Engineering Developer  
**Status:** Step 10 Completed (Zone-Specific Architecture Retrofit) - HMM Temporal Models Implemented and Validated (Pre-MRF Phase)

---

## 1. Architectural Principles: Strict Zone-Specific Temporal Modeling

Following the research-grade specification of UrbanPulse, temporal dynamics are modeled using **independent, zone-specific Hidden Markov Models**.
The core architectural principles are:
1. **One HMM per Zone * Variable:** Each monitored zone possesses its own independent Markov chain parameterized specifically for that administrative boundary.
2. **Strict Prohibition of Cross-Zone Concatenation:** Observations from different zones are **never** concatenated, blended, or pooled into a shared temporal sequence.
3. **Chronological Integrity:** Each zone's sequence is ordered strictly monotonically by timestamp ($t_1 < t_2 < \dots < t_N$). Gaps due to sensor downtime are treated as disjoint sub-sequences via segment lengths; transitions are never learned across missing temporal gaps.
4. **Non-Fabrication Policy:** Unmonitored zones or zones with insufficient observations are explicitly marked as `UNAVAILABLE`. No synthetic values are imputed. The temporal unary potential for these zones is left flat (uninformative), delegating spatial inference to the spatial Markov Random Field (MRF).

---

## 2. Mathematical Formulation of the Zone-Specific HMM

For zone $z \in \{1, \dots, 15\}$ and environmental variable $v \in \{\text{AQI}, \text{Rainfall}, \text{Temperature}\}$, let $Y_{z, t} \in \mathbb{R}$ denote the observation at timestep $t$, and let $S_{z, t} \in \{1, 2, \dots, K_z\}$ represent the latent regime.

The generative parameters $\theta_z = \{\boldsymbol{\pi}_z, \mathbf{A}_z, \boldsymbol{\mu}_z, \boldsymbol{\sigma}^2_z\}$ are:
1. **Initial State Distribution:**
   $$\pi_{z, k} = P(S_{z, 1} = k), \quad \sum_{k=1}^{K_z} \pi_{z, k} = 1$$
2. **Transition Probability Matrix $\mathbf{A}_z \in \mathbb{R}^{K_z \times K_z}$:**
   $$A_{z, ij} = P(S_{z, t+1} = j \mid S_{z, t} = i), \quad \sum_{j=1}^{K_z} A_{z, ij} = 1$$
3. **Continuous Gaussian Emission Densities:**
   $$P(Y_{z, t} = y \mid S_{z, t} = k) = \frac{1}{\sqrt{2\pi\sigma_{z, k}^2}} \exp\left( -\frac{(y - \mu_{z, k})^2}{2\sigma_{z, k}^2} \right)$$

### Multi-Step Horizon Projections
Given historical context $\mathbf{y}_{z, 1:t}$ up to time $t$:
1. **Filtered State Belief (Forward Posterior):**
   $$\gamma_{z, t}(k) = P(S_{z, t} = k \mid \mathbf{y}_{z, 1:t})$$
2. **Horizon $h$ State Projection ($h \in \{1, 3, 6, 24\}$ hours):**
   $$\mathbf{p}_{z, t+h} = \boldsymbol{\gamma}_{z, t} \mathbf{A}_z^h$$
3. **Point Forecast (Conditional Expectation):**
   $$\hat{Y}_{z, t+h} = \mathbb{E}[Y_{z, t+h} \mid \mathbf{y}_{z, 1:t}] = \sum_{j=1}^{K_z} p_{z, t+h}(j) \mu_{z, j}$$
4. **Predictive Variance:**
   $$\text{Var}(Y_{z, t+h} \mid \mathbf{y}_{z, 1:t}) = \sum_{j=1}^{K_z} p_{z, t+h}(j) (\sigma_{z, j}^2 + \mu_{z, j}^2) - \hat{Y}_{z, t+h}^2$$

---

## 3. Data Governance & Partition Roles

| Partition | Date Range | Rows | Role |
|---|---|:---:|---|
| **`train.csv`** | 2024-01-01 to 2025-05-26 | 184,320 | **Parameter Fitting only.** All Baum-Welch EM runs use exclusively this partition within that specific zone. |
| **`validate.csv`** | 2025-05-27 to 2025-09-13 | 39,600 | **Model/State Selection only.** Out-of-sample 1-step MAE is evaluated on this partition for each candidate $K$. Best $K$ is retained. |
| **`test.csv`** | 2025-09-14 to 2025-12-31 | 39,240 | **Completely held out.** Strictly untouched during model fitting, K selection, and hyperparameter tuning. Reserved exclusively for final HMM vs HMM+MRF comparison. |

---

## 4. Minimum Observation Criteria & Zone Data Availability

To prevent training unstable models on sparse observations, the following empirical thresholds are enforced:
- **Continuous Hourly Variables (AQI, Rainfall):**
  - Requires $N_{\text{train}} \ge 2,000$ valid hourly observations (~83 days).
  - Requires $N_{\text{val}} \ge 500$ valid hourly observations for out-of-sample state selection.
  - *Result for AQI:* 7 zones trained (Zones 2, 4, 5, 10, 12, 13, 14). Zone 1 has $N_{\text{train}}=934$ and $N_{\text{val}}=0$ (sensor discontinued in March 2024) and is classified as `UNAVAILABLE`. Zones 3, 6, 7, 8, 9, 11, 15 are unmonitored (`UNAVAILABLE`).
  - *Result for Rainfall:* 4 zones trained (Zones 4, 5, 10, 14). All other 11 zones lack continuous operational rainfall sensors and are classified as `UNAVAILABLE`.
- **Daily Variable (Temperature):**
  - Requires $N_{\text{train}} \ge 100$ valid daily observations.
  - IMD daily temperature provides 174 calendar days across all zones in `train.csv`.
  - Because IMD records end on 2024-06-22, `validate.csv` contains 0 observed temperature records. Thus, the previously documented **Temperature Exception** applies: $K$ selection is governed by Bayesian Information Criterion (BIC) on the zone's daily training sequence. All 15 zones train an independent daily HMM on their partition.

---

## 5. Model Training Status by Zone and Variable (45 Combinations)

| Zone ID | Zone Label | Variable | Status | Train Obs | Val Obs | Selected K | Reason / Note |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| Zone  1 | Thiruvottiyur | AQI | **UNAVAILABLE** | 934 | 0 | - | Insufficient observations (Train N=934 < 2000 or Val N=0 < 500) |
| Zone  2 | Manali | AQI | **TRAINED** | 10180 | 2582 | 6 | Trained successfully (Selected K=6) |
| Zone  3 | Madhavaram | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone  4 | Tondiarpet | AQI | **TRAINED** | 11792 | 2482 | 6 | Trained successfully (Selected K=6) |
| Zone  5 | Royapuram | AQI | **TRAINED** | 11067 | 2585 | 6 | Trained successfully (Selected K=6) |
| Zone  6 | Thiru-Vi-Ka Nagar | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone  7 | Ambattur | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone  8 | Anna Nagar | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone  9 | Teynampet | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone 10 | Kodambakkam | AQI | **TRAINED** | 9908 | 2169 | 6 | Trained successfully (Selected K=6) |
| Zone 11 | Valasaravakkam | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone 12 | Alandur | AQI | **TRAINED** | 11522 | 1792 | 6 | Trained successfully (Selected K=6) |
| Zone 13 | Adyar | AQI | **TRAINED** | 11705 | 2182 | 6 | Trained successfully (Selected K=6) |
| Zone 14 | Perungudi | AQI | **TRAINED** | 12110 | 2607 | 6 | Trained successfully (Selected K=6) |
| Zone 15 | Sholinganallur | AQI | **UNAVAILABLE** | 0 | 0 | - | Unmonitored zone (0 observations) |
| Zone  1 | Thiruvottiyur | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  2 | Manali | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  3 | Madhavaram | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  4 | Tondiarpet | Rainfall | **TRAINED** | 11936 | 2563 | 3 | Trained successfully (Selected K=3) |
| Zone  5 | Royapuram | Rainfall | **TRAINED** | 11360 | 2595 | 4 | Trained successfully (Selected K=4) |
| Zone  6 | Thiru-Vi-Ka Nagar | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  7 | Ambattur | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  8 | Anna Nagar | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  9 | Teynampet | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone 10 | Kodambakkam | Rainfall | **TRAINED** | 10092 | 2284 | 4 | Trained successfully (Selected K=4) |
| Zone 11 | Valasaravakkam | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone 12 | Alandur | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone 13 | Adyar | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone 14 | Perungudi | Rainfall | **TRAINED** | 12037 | 1787 | 4 | Trained successfully (Selected K=4) |
| Zone 15 | Sholinganallur | Rainfall | **UNAVAILABLE** | 0 | 0 | - | Unmonitored rainfall sensor (0 observations) |
| Zone  1 | Thiruvottiyur | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  2 | Manali | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  3 | Madhavaram | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  4 | Tondiarpet | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  5 | Royapuram | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  6 | Thiru-Vi-Ka Nagar | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  7 | Ambattur | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  8 | Anna Nagar | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone  9 | Teynampet | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone 10 | Kodambakkam | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone 11 | Valasaravakkam | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone 12 | Alandur | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone 13 | Adyar | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone 14 | Perungudi | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |
| Zone 15 | Sholinganallur | Temperature | **TRAINED** | 174 | 0 | 4 | Trained successfully on zone daily partition (Selected K=4 via BIC) |

---

## 6. Validation Results by Zone, Variable, and Horizon

Evaluation performed strictly on held-out `validate.csv` (2025-05-27 to 2025-09-13) for AQI and Rainfall, and on sequential daily forecasting for Temperature:

| Zone ID | Zone Label | Variable | Model | Horizon | MAE | RMSE | Sample Count |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Zone  2 | Manali | AQI | HMM | 1h | 13.751 | 16.720 | 2581 |
| Zone  2 | Manali | AQI | HMM | 3h | 13.553 | 16.774 | 2581 |
| Zone  2 | Manali | AQI | HMM | 6h | 13.304 | 17.045 | 2581 |
| Zone  2 | Manali | AQI | HMM | 24h | 14.518 | 20.921 | 2581 |
| Zone  4 | Tondiarpet | AQI | HMM | 1h | 4.870 | 7.949 | 2481 |
| Zone  4 | Tondiarpet | AQI | HMM | 3h | 4.867 | 8.036 | 2481 |
| Zone  4 | Tondiarpet | AQI | HMM | 6h | 4.901 | 8.189 | 2481 |
| Zone  4 | Tondiarpet | AQI | HMM | 24h | 5.658 | 9.448 | 2481 |
| Zone  5 | Royapuram | AQI | HMM | 1h | 4.650 | 6.915 | 2584 |
| Zone  5 | Royapuram | AQI | HMM | 3h | 4.521 | 6.621 | 2584 |
| Zone  5 | Royapuram | AQI | HMM | 6h | 4.360 | 6.237 | 2584 |
| Zone  5 | Royapuram | AQI | HMM | 24h | 3.745 | 4.798 | 2584 |
| Zone 10 | Kodambakkam | AQI | HMM | 1h | 11.742 | 13.856 | 2168 |
| Zone 10 | Kodambakkam | AQI | HMM | 3h | 11.301 | 13.364 | 2168 |
| Zone 10 | Kodambakkam | AQI | HMM | 6h | 10.687 | 12.672 | 2168 |
| Zone 10 | Kodambakkam | AQI | HMM | 24h | 8.469 | 10.226 | 2168 |
| Zone 12 | Alandur | AQI | HMM | 1h | 8.458 | 11.849 | 1791 |
| Zone 12 | Alandur | AQI | HMM | 3h | 8.219 | 11.681 | 1791 |
| Zone 12 | Alandur | AQI | HMM | 6h | 7.891 | 11.497 | 1791 |
| Zone 12 | Alandur | AQI | HMM | 24h | 8.093 | 11.495 | 1791 |
| Zone 13 | Adyar | AQI | HMM | 1h | 6.956 | 27.713 | 2181 |
| Zone 13 | Adyar | AQI | HMM | 3h | 6.829 | 28.115 | 2181 |
| Zone 13 | Adyar | AQI | HMM | 6h | 6.711 | 28.731 | 2181 |
| Zone 13 | Adyar | AQI | HMM | 24h | 7.634 | 32.154 | 2181 |
| Zone 14 | Perungudi | AQI | HMM | 1h | 9.347 | 12.584 | 2606 |
| Zone 14 | Perungudi | AQI | HMM | 3h | 9.057 | 12.026 | 2606 |
| Zone 14 | Perungudi | AQI | HMM | 6h | 8.666 | 11.303 | 2606 |
| Zone 14 | Perungudi | AQI | HMM | 24h | 7.242 | 9.601 | 2606 |
| Zone  4 | Tondiarpet | Rainfall | HMM | 1h | 0.398 | 2.728 | 2562 |
| Zone  4 | Tondiarpet | Rainfall | HMM | 3h | 0.458 | 2.637 | 2562 |
| Zone  4 | Tondiarpet | Rainfall | HMM | 6h | 0.491 | 2.636 | 2562 |
| Zone  4 | Tondiarpet | Rainfall | HMM | 24h | 0.502 | 2.640 | 2562 |
| Zone  5 | Royapuram | Rainfall | HMM | 1h | 0.347 | 2.456 | 2594 |
| Zone  5 | Royapuram | Rainfall | HMM | 3h | 0.365 | 2.389 | 2594 |
| Zone  5 | Royapuram | Rainfall | HMM | 6h | 0.382 | 2.412 | 2594 |
| Zone  5 | Royapuram | Rainfall | HMM | 24h | 0.397 | 2.437 | 2594 |
| Zone 10 | Kodambakkam | Rainfall | HMM | 1h | 0.277 | 1.021 | 2283 |
| Zone 10 | Kodambakkam | Rainfall | HMM | 3h | 0.415 | 0.855 | 2283 |
| Zone 10 | Kodambakkam | Rainfall | HMM | 6h | 0.571 | 0.867 | 2283 |
| Zone 10 | Kodambakkam | Rainfall | HMM | 24h | 0.914 | 1.110 | 2283 |
| Zone 14 | Perungudi | Rainfall | HMM | 1h | 1.624 | 5.185 | 1786 |
| Zone 14 | Perungudi | Rainfall | HMM | 3h | 2.816 | 5.913 | 1786 |
| Zone 14 | Perungudi | Rainfall | HMM | 6h | 3.104 | 6.130 | 1786 |
| Zone 14 | Perungudi | Rainfall | HMM | 24h | 3.132 | 6.152 | 1786 |
| Zone  1 | Thiruvottiyur | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  2 | Manali | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  3 | Madhavaram | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  4 | Tondiarpet | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  5 | Royapuram | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  6 | Thiru-Vi-Ka Nagar | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  7 | Ambattur | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  8 | Anna Nagar | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone  9 | Teynampet | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone 10 | Kodambakkam | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone 11 | Valasaravakkam | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone 12 | Alandur | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone 13 | Adyar | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone 14 | Perungudi | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |
| Zone 15 | Sholinganallur | Temperature | HMM | 24h | 0.752 | 0.989 | 170 |

---

## 7. Model Artifact Structure

The trained models are explicitly keyed by `variable -> zone_id` and saved under `models/hmm/`:
- **Aggregated Dictionaries:**
  - `models/hmm/hmm_aqi.joblib`: Dict `{zone_id: TemporalHMM}` for monitored zones [2, 4, 5, 10, 12, 13, 14]
  - `models/hmm/hmm_rainfall.joblib`: Dict `{zone_id: TemporalHMM}` for monitored zones [4, 5, 10, 14]
  - `models/hmm/hmm_temperature.joblib`: Dict `{zone_id: TemporalHMM}` for zones [1..15]
  - `models/hmm/hmm_models_by_zone.joblib`: Dict `{(variable, zone_id): TemporalHMM}` (26 total models)
- **Individual Zone Artifacts:**
  - `models/hmm/hmm_aqi_zone_{z}.joblib`
  - `models/hmm/hmm_rainfall_zone_{z}.joblib`
  - `models/hmm/hmm_temperature_zone_{z}.joblib`
- **Audit Records:**
  - `results/metrics/model_training_status.csv` (all 45 combinations audited)
  - `results/metrics/validation_metrics_hmm.csv` (59 multi-horizon validation evaluations)
  - `results/metrics/state_selection_hmm.csv` (all candidate K BIC/AIC/val-MAE evaluations)
  - `results/predictions/validation_predictions_hmm.csv` (timestamped point predictions and predictive variances)

---

## 8. Spatial Markov Random Field (MRF) — Mathematical Formulation & Results

### 8.1 Mathematical Formulation

The UrbanPulse MRF is a **Continuous Gaussian Markov Random Field** over the
15-zone Chennai administrative graph. Spatial inference is exact (no approximations).

**Random variables:** `X = (X_1, ..., X_15)` — the latent environmental field
(AQI, rainfall, or temperature) across all 15 zones at a given time step.

**Graph:** `G = (V, E)` where `|V|=15` and `|E|=29` verified shared-boundary edges
from `data/processed/zone_adjacency.csv`. Edge weights are normalized:
```
  w_{uv} = L_{uv} / L_bar     (L_bar = mean boundary length across all 29 edges)
```

**Energy function:**
```
  E(X) = 0.5 * sum_{u: monitored} lambda_u * (X_u - y_hat_u)^2
       + 0.5 * beta * sum_{(u,v) in E} w_{uv} * (X_u - X_v)^2
```

**Probability model:**
```
  P(X | observations) prop-to exp(-E(X))
```

**Unary potential** (per zone u):
```
  phi_u(X_u) = exp(-0.5 * lambda_u * (X_u - y_hat_u)^2)
  lambda_u = 1 / max(1e-4, sigma_hat_u^2)    if zone u has HMM forecast
  lambda_u = 0.0                              if zone u is unmonitored
```
  where `y_hat_u` and `sigma_hat_u^2` are the HMM predictive mean and variance.
  **For unmonitored zones, lambda_u = 0 and b_u = 0 — no observation is fabricated.**

**Pairwise potential** (per edge (u,v)):
```
  psi_{uv}(X_u, X_v) = exp(-0.5 * beta * w_{uv} * (X_u - X_v)^2)
```

**Exact Gaussian Inference:**
The energy is quadratic in X, so the posterior is an exact multivariate Gaussian:
```
  Precision matrix:  Q = Lambda + beta * L_w      (Lambda = diag(lambda_1,...,lambda_15))
  Information vector: b = Lambda * y_hat
  Posterior mean:    mu_MRF = Q^{-1} b             (solve via numpy.linalg.solve)
  Posterior cov:     Sigma_MRF = Q^{-1}            (marginal variances = diag(Sigma_MRF))
```

**Numerical stability:** For `beta=0` (grid-search candidate) with unmonitored zones,
Q acquires zero eigenvalues. A Tikhonov regularizer `eps * I` (eps ~ 1e-9) is added
only in this degenerate case — it does not bias any observation or prior mean.
For all `beta > 0` with the connected graph, Q is strictly SPD without regularization.

### 8.2 Data Governance for MRF

| Component | Data Source | Justification |
|-----------|-------------|---------------|
| HMM unary parameters (y_hat, sigma^2) | Derived from zone-specific HMMs trained on `train.csv` | Training parameters from training partition only |
| Beta calibration — AQI | `validate.csv` (1h horizon MAE grid search) | Model selection on validation partition |
| Beta calibration — Rainfall | `validate.csv` (1h horizon MAE grid search) | Model selection on validation partition |
| Beta calibration — Temperature | Chronological tail of `train.csv` (last 20% of days) | validate.csv has zero temperature observations (all NaN) |
| Final MRF evaluation | `validate.csv` (AQI, Rainfall); train-tail (Temperature) | Same sources as beta calibration — no leakage |
| `test.csv` | **NEVER LOADED** | Strictly held out for final evaluation |

**Temperature governance note:** The validation split (`validate.csv`) contains
zero temperature ground-truth observations (`temperature_observed == False` for
all rows; `temperature` column is entirely NaN in that split). Beta for the
Temperature MRF was therefore calibrated on the chronologically held-out last
20% of training days, which is strictly disjoint from the 80% prefix used as
HMM context. This is explicitly documented as a train-tail calibration.

### 8.3 Spatial Coupling Parameter Beta

| Variable | Calibration Source | Optimal Beta | Val MAE (1h/1-day) |
|----------|--------------------|--------------|-------------------|
| AQI | validate.csv | 0.00010 | 6.865 AQI units |
| Rainfall | validate.csv | 0.00000 | 0.487 mm/h |
| Temperature | train-tail (last 20% of train days) | 0.00000 | 1.110 C |

### 8.4 Beta Calibration Scores (Full Grid)

**AQI — Calibration on validate.csv (1h MAE):**

| Beta | Val MAE (1h) | Val RMSE (1h) | N |
|------|-------------|--------------|---|
| 0.00010 | 6.865 | 14.690 | 16399 |
| 0.00005 | 7.084 | 14.351 | 16399 |
| 0.00001 | 7.573 | 14.208 | 16399 |
| 0.00000 | 7.742 | 14.198 | 16399 |
| 0.00033 | 7.742 | 16.524 | 16399 |
| 0.00100 | 9.873 | 19.558 | 16399 |
| 0.00500 | 13.321 | 23.685 | 16399 |
| 0.01000 | 14.734 | 25.097 | 16399 |

**Rainfall — Calibration on validate.csv (1h MAE):**

| Beta | Val MAE (1h) | Val RMSE (1h) | N |
|------|-------------|--------------|---|
| 0.00000 | 0.487 | 2.614 | 9229 |
| 0.00050 | 0.490 | 2.633 | 9229 |
| 0.00100 | 0.495 | 2.653 | 9229 |
| 0.00500 | 0.547 | 2.761 | 9229 |
| 0.01000 | 0.588 | 2.839 | 9229 |
| 0.05000 | 0.712 | 3.063 | 9229 |
| 0.10000 | 0.754 | 3.146 | 9229 |
| 0.20000 | 0.782 | 3.208 | 9229 |

**Temperature — Calibration on train-tail hold-out (1-day MAE):**

| Beta | Tail MAE (1-day) | Tail RMSE | N_samples |
|------|----------------|-----------|-----------|
| 0.00000 | 1.110 | 1.303 | 495 |
| 0.00010 | 1.110 | 1.303 | 495 |
| 0.00050 | 1.110 | 1.303 | 495 |
| 0.00100 | 1.110 | 1.303 | 495 |
| 0.00500 | 1.110 | 1.303 | 495 |
| 0.01000 | 1.110 | 1.303 | 495 |
| 0.05000 | 1.110 | 1.303 | 495 |
| 0.10000 | 1.110 | 1.303 | 495 |
| 0.50000 | 1.110 | 1.303 | 495 |
| 1.00000 | 1.110 | 1.303 | 495 |

### 8.5 HMM+MRF Validation Results

**AQI — HMM+MRF Validation Metrics (validate.csv, per zone, per horizon):**

| Zone | Zone Label | Horizon | MAE | RMSE | N |
|------|-----------|---------|-----|------|---|
| 2 | Manali | 1h | 11.967 | 15.121 | 2582 |
| 4 | Tondiarpet | 1h | 4.481 | 7.643 | 2482 |
| 5 | Royapuram | 1h | 3.955 | 5.549 | 2585 |
| 10 | Kodambakkam | 1h | 8.105 | 9.620 | 2169 |
| 12 | Alandur | 1h | 6.003 | 9.764 | 1792 |
| 13 | Adyar | 1h | 6.668 | 31.387 | 2182 |
| 14 | Perungudi | 1h | 6.691 | 8.816 | 2607 |
| 2 | Manali | 3h | 11.811 | 15.309 | 2582 |
| 4 | Tondiarpet | 3h | 4.480 | 7.756 | 2482 |
| 5 | Royapuram | 3h | 3.853 | 5.318 | 2585 |
| 10 | Kodambakkam | 3h | 8.032 | 9.665 | 2169 |
| 12 | Alandur | 3h | 5.838 | 10.057 | 1792 |
| 13 | Adyar | 3h | 6.736 | 31.745 | 2182 |
| 14 | Perungudi | 3h | 6.758 | 8.976 | 2607 |
| 2 | Manali | 6h | 11.668 | 15.799 | 2582 |
| 4 | Tondiarpet | 6h | 4.534 | 7.950 | 2482 |
| 5 | Royapuram | 6h | 3.726 | 5.027 | 2585 |
| 10 | Kodambakkam | 6h | 8.297 | 10.123 | 2169 |
| 12 | Alandur | 6h | 5.853 | 10.545 | 1792 |
| 13 | Adyar | 6h | 6.965 | 32.224 | 2182 |
| 14 | Perungudi | 6h | 6.907 | 9.286 | 2607 |
| 2 | Manali | 24h | 13.923 | 20.477 | 2582 |
| 4 | Tondiarpet | 24h | 5.476 | 9.380 | 2482 |
| 5 | Royapuram | 24h | 3.495 | 4.347 | 2585 |
| 10 | Kodambakkam | 24h | 12.636 | 15.047 | 2169 |
| 12 | Alandur | 24h | 8.010 | 13.333 | 1792 |
| 13 | Adyar | 24h | 9.094 | 34.148 | 2182 |
| 14 | Perungudi | 24h | 8.558 | 11.760 | 2607 |

**Rainfall — HMM+MRF Validation Metrics (validate.csv, per zone, per horizon):**

| Zone | Zone Label | Horizon | MAE | RMSE | N |
|------|-----------|---------|-----|------|---|
| 4 | Tondiarpet | 1h | 0.277 | 2.068 | 2563 |
| 5 | Royapuram | 1h | 0.192 | 1.763 | 2595 |
| 10 | Kodambakkam | 1h | 0.208 | 0.580 | 2284 |
| 14 | Perungudi | 1h | 1.574 | 4.922 | 1787 |
| 4 | Tondiarpet | 3h | 0.417 | 2.436 | 2563 |
| 5 | Royapuram | 3h | 0.291 | 2.099 | 2595 |
| 10 | Kodambakkam | 3h | 0.358 | 0.526 | 2284 |
| 14 | Perungudi | 3h | 2.820 | 5.897 | 1787 |
| 4 | Tondiarpet | 6h | 0.483 | 2.596 | 2563 |
| 5 | Royapuram | 6h | 0.362 | 2.329 | 2595 |
| 10 | Kodambakkam | 6h | 0.534 | 0.679 | 2284 |
| 14 | Perungudi | 6h | 3.118 | 6.164 | 1787 |
| 4 | Tondiarpet | 24h | 0.502 | 2.639 | 2563 |
| 5 | Royapuram | 24h | 0.397 | 2.436 | 2595 |
| 10 | Kodambakkam | 24h | 0.911 | 1.094 | 2284 |
| 14 | Perungudi | 24h | 3.147 | 6.190 | 1787 |

**Temperature — HMM+MRF Metrics (train-tail hold-out, 24h horizon, all 15 zones):**

| Zone | Zone Label | Horizon | MAE | RMSE | N | Eval Source |
|------|-----------|---------|-----|------|---|------------|
| 1 | Thiruvottiyur | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 2 | Manali | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 3 | Madhavaram | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 4 | Tondiarpet | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 5 | Royapuram | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 6 | Thiru-Vi-Ka Nagar | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 7 | Ambattur | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 8 | Anna Nagar | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 9 | Teynampet | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 10 | Kodambakkam | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 11 | Valasaravakkam | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 12 | Alandur | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 13 | Adyar | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 14 | Perungudi | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |
| 15 | Sholinganallur | 24h | 1.110 | 1.303 | 33 | train_tail_held_out_20pct |

### 8.6 Model Artifact Structure (MRF)

- `models/hmm_mrf/mrf_aqi.joblib`: Calibrated SpatialMRF for AQI
- `models/hmm_mrf/mrf_rainfall.joblib`: Calibrated SpatialMRF for Rainfall
- `models/hmm_mrf/mrf_temperature.joblib`: Calibrated SpatialMRF for Temperature
- `models/hmm_mrf/mrf_all_models.joblib`: Registry dict `{variable: SpatialMRF}`
- `results/metrics/mrf_calibration_scores.csv`: Full beta grid search scores
- `results/metrics/validation_metrics_mrf.csv`: Per-zone per-horizon HMM+MRF metrics
- `results/predictions/validation_predictions_mrf.csv`: Timestamped predictions with uncertainty

### 8.7 Physical Context, Hypotheses, and Behavioral Findings

All physical interpretations presented below are **contextual hypotheses** intended to interpret the learned model parameters and empirical behavior, not direct empirical findings proven by the HMM/MRF:

1. **AQI Spatial Coupling (Optimal $\beta = 0.00010$):**
   - Incorporating spatial coupling across adjacent zones reduces 1h validation MAE from 7.742 (HMM alone) to 6.865 AQI units (an 11.3% error reduction).
   - *Contextual Hypothesis:* Particulate matter (PM2.5/PM10) undergoes regional atmospheric diffusion and transport across adjacent urban zones, creating spatial correlation that the Gaussian MRF successfully exploits to regularize local sensor noise.
   - For unmonitored zones (Zones 1, 3, 6, 7, 8, 9, 11, 15), the MRF infers posterior concentrations purely through boundary-weighted Laplacian smoothing from adjacent monitored zones ($\lambda_u = 0, b_u = 0$).

2. **Rainfall Spatial Decoupling (Optimal $\beta = 0.00000$):**
   - The validation grid search demonstrated that increasing $\beta$ strictly degrades 1h rainfall MAE (from 0.487 mm/h at $\beta=0$ to 0.782 mm/h at $\beta=0.2$). The HMM temporal predictive distributions alone are optimal.
   - *Contextual Hypothesis:* Coastal tropical rainfall in Chennai during pre-monsoon and convective periods often occurs as localized cell bursts rather than city-wide uniform precipitation. Enforcing Laplacian smoothness across adjacent zones artificially smears sharp localized rain events into dry neighboring zones, increasing overall error.

3. **Temperature Spatial Invariance ($\beta = 0.00000$):**
   - Temperature MAE remained identical (1.110°C) across all 10 candidate $\beta$ values from $0.0$ to $1.0$ on the chronologically held-out train-tail.
   - *Data Context Note:* In the available training dataset, recorded surface temperature values are identical across all 15 administrative zones per day (spatially uniform regional meteorological observation). Because $y_{\text{hat}, u} = y_{\text{hat}, v}$ for all pairs $(u, v)$, the pairwise difference $(X_u - X_v)$ is zero, rendering the spatial Laplacian penalty mathematically invariant to $\beta$.

---

## 9. Probabilistic Alert Engine Methodology & Operational Decision Rules

### 9.1 Separation of Official Physical Thresholds from Model Decision Rules

A critical governance requirement of UrbanPulse is the rigorous distinction between **official physical hazard standards** (established by government agencies) and **operational probabilistic decision rules** (established by UrbanPulse to manage forecast uncertainty).

#### A. Official Physical Standards:
1. **AQI:**
   * **Official Threshold:** $\text{AQI} > 200.0$
   * **Source:** Central Pollution Control Board (CPCB) National Air Quality Index Standard.
   * **Category:** 'Poor' air quality category boundary (201–300).
   * **Scale:** 0–50 Good, 51–100 Satisfactory, 101–200 Moderate, 201–300 Poor, 301–400 Very Poor, 401–500 Severe.
   * **Event Evaluated:** $P(\text{AQI} > 200.0)$.

2. **Rainfall:**
   * **Official Threshold:** $\ge 64.5\text{ mm} / 24\text{ hours}$
   * **Source:** India Meteorological Department (IMD) Standard Meteorological Classification for 'Heavy Rainfall'.
   * **Temporal Resolution Rule:** The official standard is explicitly defined on a 24-hour accumulation basis. **Hourly rainfall forecasts are NEVER directly compared to 64.5 mm.**
   * **Accumulation Formulation:** Evaluated strictly on the 24-hour forecast accumulation sum $S_{24, t} = \sum_{k=1}^{24} Y_{t+k}$ derived from the Markov chain predictive distribution.
   * **Event Evaluated:** $P(S_{24, t} \ge 64.5\text{ mm})$.

3. **Temperature:**
   * **Official Criterion:** Joint condition for coastal stations (such as Chennai):
     $$\text{Tmax} \ge 37.0^\circ\text{C} \quad \text{AND} \quad \text{Departure from Normal} \ge +4.5^\circ\text{C}$$
   * **Source:** IMD Official Heatwave Criteria for Coastal Stations.
   * **Rigorous Negative Constraint:** $\text{Tmax} \ge 37.0^\circ\text{C}$ is **NEVER** used as a standalone heatwave trigger.
   * **Climatological Normal Governance:** A verified climatological normal $\text{Normal\_Tmax}$ is required for operational declaration. Because normal Tmax climatology is not present in the project data, missing normals are **NEVER fabricated**. If climatology is absent, the alert status is designated `UNAVAILABLE_MISSING_CLIMATOLOGY`.
   * **Test Period Status:** The 2025 test dataset (`test.csv`) contains zero ground-truth temperature records (`temperature_observed == False` throughout). No empirical temperature alerts are claimed for the test split.

#### B. UrbanPulse Operational Risk Decision Rules:
The probability cutoffs used to trigger alert statuses are **UrbanPulse risk-management decision rules**, and are **NOT official CPCB or IMD categories**:
* **`NORMAL`:** $P(\text{exceedance}) < 0.20$ (Low operational likelihood of crossing dangerous threshold).
* **`WATCH`:** $0.20 \le P(\text{exceedance}) < 0.60$ (Elevated operational risk; warrants heightened monitoring).
* **`WARNING`:** $P(\text{exceedance}) \ge 0.60$ (High operational confidence of crossing dangerous threshold; warrants proactive public advisory).

### 9.2 Mathematical Formulation of Threshold Exceedance

Under the exact Gaussian posterior distribution $\mathcal{N}(\mu, \sigma^2)$ generated by the HMM+MRF:
$$P(X \ge \tau) = \frac{1}{2} \operatorname{erfc}\left(\frac{\tau - \mu}{\sigma \sqrt{2}}\right)$$
* For unmonitored rainfall zones (where $\beta = 0.0$ and observation evidence is genuinely absent), no artificial regularizer variance is used to compute probabilities; status is strictly designated `UNMONITORED_INSUFFICIENT_DATA`.
* For monitored rainfall zones, $\mu$ and $\sigma^2$ are computed via exact closed-form 24-hour accumulation integration accounting for all multi-step transition auto-covariances.


