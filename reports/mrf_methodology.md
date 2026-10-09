# UrbanPulse: Markov Random Field (MRF) Methodology & Validation Report
**Project:** Spatio-Temporal Early Warning System for Chennai - AQI, Rainfall & Temperature Forecasting  
**Author:** Lead ML / Data-Engineering Developer  
**Status:** Step 11 Completed - Probabilistic MRF Spatial Component Implemented and Validated (Pre-Final Evaluation Phase)

---

## 1. Architectural Role of the MRF in UrbanPulse

In UrbanPulse, the spatio-temporal early warning system is composed strictly of:
$$\text{HMM (Temporal)} + \text{MRF (Spatial)} = \text{Spatio-Temporal Pipeline}$$

The HMM models the dynamic, hour-to-hour temporal regime evolution for each monitored administrative zone independently.
The **Markov Random Field (MRF)** models the **spatial dependence, physical continuity, and boundary coupling** between adjacent Chennai zones across the topological graph.

---

## 2. Mathematical Formulation of the Continuous Gaussian MRF

Let $G = (V, E)$ be the undirected spatial graph over Chennai's 15 administrative zones:
- $V = \{1, 2, \dots, 15\}$ (the 15 administrative zones).
- $E = \{(u, v)\}$ (the **29 verified shared-boundary edges** from `zone_adjacency.csv`).
- $w_{uv} = \frac{L_{uv}}{\bar{L}}$ is the normalized shared boundary length, where $\bar{L} = 0.054906$ is the mean boundary length across all 29 edges.

Let $\mathbf{X} = (X_1, X_2, \dots, X_{15})^T \in \mathbb{R}^{15}$ represent the spatial field of environmental indicator values at forecast horizon $h$.

The joint posterior probability density given historical observations $\mathbf{Y}_{1:t}$ follows the Gibbs distribution:
$$P(\mathbf{X} \mid \mathbf{Y}_{1:t}) = \frac{1}{Z} \exp\left( -E(\mathbf{X}) \right)$$

### 2.1 Energy Function Decomposition
The total field energy $E(\mathbf{X})$ is the sum of unary and pairwise potential energies:
$$E(\mathbf{X}) = \sum_{u \in V} \phi_u(X_u) + \sum_{(u, v) \in E} \psi_{uv}(X_u, X_v)$$

### 2.2 Unary Potentials from HMM Predictive Distributions
For each zone $u$, the temporal HMM produces a Gaussian predictive distribution with conditional expectation $\hat{y}_u = \mathbb{E}[Y_{u, t+h} \mid \mathbf{y}_{u, 1:t}]$ and predictive variance $\sigma_u^2 = \text{Var}(Y_{u, t+h} \mid \mathbf{y}_{u, 1:t})$.

The unary potential is defined as the negative log-density of this predictive distribution:
$$\phi_u(X_u) = -\log P(X_u \mid \mathbf{y}_{u, 1:t}) = \frac{1}{2} \lambda_u (X_u - \hat{y}_u)^2 + \text{const}$$
where:
$$\lambda_u = \frac{1}{\sigma_u^2}$$
is the **unary precision** (confidence of the HMM prediction).
- **Monitored Zones:** $\lambda_u = 1 / \sigma_u^2$. A high-confidence HMM prediction (small $\sigma_u^2$) exerts a strong restoring force pinning $X_u$ near $\hat{y}_u$.
- **Unmonitored / Discontinued Zones:** Zones without an active HMM have no local temporal observation evidence. They are assigned an uninformative prior precision $\lambda_u = \epsilon = 10^{-6}$ with the regional mean. These zones contribute virtually zero unary pinning, so their posterior value is determined entirely by spatial propagation from neighboring zones through the pairwise potentials.

### 2.3 Spatial Pairwise Potentials
For each adjacent pair $(u, v) \in E$:
$$\psi_{uv}(X_u, X_v) = \frac{1}{2} \beta w_{uv} (X_u - X_v)^2$$
where $\beta > 0$ is the global spatial coupling coefficient, and $w_{uv} > 0$ is the normalized shared boundary length.

### 2.4 Precision Matrix & Exact Posterior Distribution
Combining unary and pairwise terms yields a **Gaussian Markov Random Field**:
$$E(\mathbf{X}) = \frac{1}{2} (\mathbf{X} - \hat{\mathbf{y}})^T \boldsymbol{\Lambda} (\mathbf{X} - \hat{\mathbf{y}}) + \frac{1}{2} \mathbf{X}^T \left( \beta \mathbf{L}_w \right) \mathbf{X}$$
where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_{15})$, and $\mathbf{L}_w$ is the weighted graph Laplacian:
$$(\mathbf{L}_w)_{uu} = \sum_{v \in \mathcal{N}(u)} w_{uv}, \quad (\mathbf{L}_w)_{uv} = -w_{uv} \quad \text{for } (u, v) \in E$$

The precision matrix $\mathbf{Q} \in \mathbb{R}^{15 \times 15}$ is:
$$\mathbf{Q} = \boldsymbol{\Lambda} + \beta \mathbf{L}_w$$
and the linear information vector $\mathbf{b} \in \mathbb{R}^{15}$ is:
$$\mathbf{b} = \boldsymbol{\Lambda} \hat{\mathbf{y}}$$

Because $\boldsymbol{\Lambda} \succ 0$ and $\mathbf{L}_w \succeq 0$, the precision matrix $\mathbf{Q}$ is strictly symmetric positive definite (SPD).
The exact posterior distribution is multivariate Gaussian:
$$\mathbf{X} \mid \mathbf{Y}_{1:t} \sim \mathcal{N}\left( \boldsymbol{\mu}_{\text{MRF}}, \, \mathbf{\Sigma}_{\text{MRF}} \right)$$
where:
$$\boldsymbol{\mu}_{\text{MRF}} = \mathbf{Q}^{-1} \mathbf{b}$$
$$\mathbf{\Sigma}_{\text{MRF}} = \mathbf{Q}^{-1}$$
and marginal predictive variances are $\text{Var}(X_u) = (\mathbf{Q}^{-1})_{uu}$.

---

## 3. Probabilistic Inference Methods

`SpatialMRF` implements three genuine probabilistic inference procedures:

1. **Exact Cholesky Solve (`infer_exact` - Default):**
   Computes the Cholesky factorization $\mathbf{Q} = \mathbf{R}^T \mathbf{R}$ and solves $\mathbf{R}^T \mathbf{R} \boldsymbol{\mu} = \mathbf{b}$ via forward and back substitution. Exact, deterministic, and executes in $< 0.1\text{ ms}$ for the 15-zone graph.
2. **Gaussian Belief Propagation (`infer_belief_propagation`):**
   Asynchronous Gauss-Seidel message passing over the 29 graph edges:
   $$\mu_u^{(k+1)} = \frac{1}{\lambda_u + \beta \sum_{v \in \mathcal{N}(u)} w_{uv}} \left( \lambda_u \hat{y}_u + \beta \sum_{v \in \mathcal{N}(u)} w_{uv} \mu_v^{(k)} \right)$$
   Guaranteed to converge to the exact MAP estimate because $\mathbf{Q}$ is strictly diagonally dominant.
3. **Gibbs MCMC Sampling (`infer_gibbs_sampling`):**
   Draws continuous field samples from the full conditional distributions:
   $$P(X_u \mid \mathbf{X}_{-u}) = \mathcal{N}\left( X_u; \, \frac{\lambda_u \hat{y}_u + \beta \sum_{v \in \mathcal{N}(u)} w_{uv} X_v}{\lambda_u + \beta \sum_{v \in \mathcal{N}(u)} w_{uv}}, \, \frac{1}{\lambda_u + \beta \sum_{v \in \mathcal{N}(u)} w_{uv}} \right)$$
   Generates empirical posterior samples, verifying the analytical distribution.

### Why This Is NOT Heuristic Smoothing
The MRF does **not** perform uniform neighbor averaging or rule-based adjustments.
The influence of neighbors is strictly modulated by:
- **HMM Predictive Confidence:** If zone $u$ has high temporal certainty (small $\sigma_u^2$, large $\lambda_u$), the unary term dominates ($Q_{uu} \approx \lambda_u$) and neighbor values have virtually zero influence. If $\sigma_u^2$ is large, spatial neighbors exert proportionately more pulling force.
- **Topological Boundary Length:** Neighbors sharing longer boundaries ($w_{uv} > 1$) exert greater mutual coupling than those sharing narrow corners.

---

## 4. Parameter Estimation & Calibration Governance

### 4.1 Strict Partition Isolation
- Parameter estimation & empirical spatial covariance: computed using **`train.csv` only**.
- Hyperparameter search & coupling $\beta$ calibration: evaluated using **`validate.csv` only**.
- Final evaluation: **`test.csv` has remained completely untouched and isolated**.

### 4.2 Calibration Grid Search on Validation Partition
The spatial coupling strength $\beta$ was calibrated by evaluating rolling 1-step-ahead validation MAE across candidate values:

| Variable | Candidate $\beta$ | Val MAE (1h) | Val RMSE (1h) | Sample Count |
|:---|:---:|:---:|:---:|:---:|
| AQI |   0.000000 | 8.524 | 15.140 | 16392 |
| AQI |   0.000010 | 8.382 | 15.161 | 16392 |
| AQI |   0.000050 | 7.940 | 15.272 | 16392 |
| AQI |   0.000100 | 7.710 | 15.516 | 16392 |
| AQI |   0.000330 | 8.337 | 17.074 | 16392 |
| AQI |   0.001000 | 10.387 | 19.938 | 16392 |
| AQI |   0.005000 | 13.667 | 23.953 | 16392 |
| AQI |   0.010000 | 15.017 | 25.341 | 16392 |
| Rainfall |   0.000000 | 0.591 | 3.037 | 9225 |
| Rainfall |   0.000500 | 0.587 | 3.023 | 9225 |
| Rainfall |   0.001000 | 0.587 | 3.020 | 9225 |
| Rainfall |   0.005000 | 0.614 | 3.037 | 9225 |
| Rainfall |   0.010000 | 0.643 | 3.065 | 9225 |
| Rainfall |   0.050000 | 0.739 | 3.179 | 9225 |
| Rainfall |   0.100000 | 0.773 | 3.229 | 9225 |
| Rainfall |   0.200000 | 0.796 | 3.267 | 9225 |

### 4.3 Selected Calibrated Coupling Parameters
- **Air Quality Index (AQI):** $\beta^* = 0.000100$ ($10^{-4}$).
  - At $\beta = 0$ (HMM alone, no spatial coupling): Val MAE = 8.524.
  - At calibrated $\beta^* = 0.0001$: Val MAE = 7.710.
- **Rainfall:** $\beta^* = 0.000500$.
  - At $\beta = 0$: Val MAE = 0.591 mm.
  - At calibrated $\beta^* = 0.0005$: Val MAE = 0.587 mm.
- **Temperature:** $\beta = 0.010000$.
  - *Provenance Note:* IMD daily temperature is a common regional measurement across Chennai. Coupling $\beta = 0.01$ maintains spatial coherence across administrative boundaries without artificial distortion.

---

## 5. Multi-Horizon Validation Results (HMM + MRF)

Evaluated on held-out `validate.csv` (2025-05-27 to 2025-09-13):

| Zone ID | Zone Label | Variable | Model | Horizon | MAE | RMSE | Sample Count |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Zone  2 | Manali | AQI | HMM+MRF | 1h | 13.668 | 17.067 | 2581 |
| Zone  4 | Tondiarpet | AQI | HMM+MRF | 1h | 4.874 | 8.102 | 2481 |
| Zone  5 | Royapuram | AQI | HMM+MRF | 1h | 4.284 | 6.009 | 2584 |
| Zone 10 | Kodambakkam | AQI | HMM+MRF | 1h | 8.769 | 10.476 | 2169 |
| Zone 12 | Alandur | AQI | HMM+MRF | 1h | 7.484 | 11.974 | 1792 |
| Zone 13 | Adyar | AQI | HMM+MRF | 1h | 7.107 | 31.581 | 2181 |
| Zone 14 | Perungudi | AQI | HMM+MRF | 1h | 7.687 | 10.142 | 2606 |
| Zone  2 | Manali | AQI | HMM+MRF | 3h | 13.457 | 17.148 | 2581 |
| Zone  4 | Tondiarpet | AQI | HMM+MRF | 3h | 4.868 | 8.198 | 2481 |
| Zone  5 | Royapuram | AQI | HMM+MRF | 3h | 4.182 | 5.782 | 2584 |
| Zone 10 | Kodambakkam | AQI | HMM+MRF | 3h | 8.723 | 10.441 | 2169 |
| Zone 12 | Alandur | AQI | HMM+MRF | 3h | 7.328 | 12.121 | 1792 |
| Zone 13 | Adyar | AQI | HMM+MRF | 3h | 7.136 | 31.920 | 2181 |
| Zone 14 | Perungudi | AQI | HMM+MRF | 3h | 7.637 | 10.138 | 2606 |
| Zone  2 | Manali | AQI | HMM+MRF | 6h | 13.221 | 17.469 | 2581 |
| Zone  4 | Tondiarpet | AQI | HMM+MRF | 6h | 4.915 | 8.367 | 2481 |
| Zone  5 | Royapuram | AQI | HMM+MRF | 6h | 4.073 | 5.520 | 2584 |
| Zone 10 | Kodambakkam | AQI | HMM+MRF | 6h | 8.941 | 10.761 | 2169 |
| Zone 12 | Alandur | AQI | HMM+MRF | 6h | 7.338 | 12.427 | 1792 |
| Zone 13 | Adyar | AQI | HMM+MRF | 6h | 7.248 | 32.364 | 2181 |
| Zone 14 | Perungudi | AQI | HMM+MRF | 6h | 7.622 | 10.248 | 2606 |
| Zone  2 | Manali | AQI | HMM+MRF | 24h | 14.854 | 21.349 | 2581 |
| Zone  4 | Tondiarpet | AQI | HMM+MRF | 24h | 5.774 | 9.676 | 2481 |
| Zone  5 | Royapuram | AQI | HMM+MRF | 24h | 3.792 | 4.792 | 2584 |
| Zone 10 | Kodambakkam | AQI | HMM+MRF | 24h | 12.585 | 15.169 | 2169 |
| Zone 12 | Alandur | AQI | HMM+MRF | 24h | 9.290 | 14.551 | 1792 |
| Zone 13 | Adyar | AQI | HMM+MRF | 24h | 9.038 | 34.199 | 2181 |
| Zone 14 | Perungudi | AQI | HMM+MRF | 24h | 8.658 | 12.074 | 2606 |
| Zone  4 | Tondiarpet | Rainfall | HMM+MRF | 1h | 0.393 | 2.717 | 2562 |
| Zone  5 | Royapuram | Rainfall | HMM+MRF | 1h | 0.339 | 2.433 | 2594 |
| Zone 10 | Kodambakkam | Rainfall | HMM+MRF | 1h | 0.255 | 0.885 | 2284 |
| Zone 14 | Perungudi | Rainfall | HMM+MRF | 1h | 1.665 | 5.244 | 1787 |
| Zone  4 | Tondiarpet | Rainfall | HMM+MRF | 3h | 0.457 | 2.637 | 2562 |
| Zone  5 | Royapuram | Rainfall | HMM+MRF | 3h | 0.363 | 2.389 | 2594 |
| Zone 10 | Kodambakkam | Rainfall | HMM+MRF | 3h | 0.401 | 0.801 | 2284 |
| Zone 14 | Perungudi | Rainfall | HMM+MRF | 3h | 2.832 | 5.954 | 1787 |
| Zone  4 | Tondiarpet | Rainfall | HMM+MRF | 6h | 0.490 | 2.636 | 2562 |
| Zone  5 | Royapuram | Rainfall | HMM+MRF | 6h | 0.383 | 2.412 | 2594 |
| Zone 10 | Kodambakkam | Rainfall | HMM+MRF | 6h | 0.559 | 0.854 | 2284 |
| Zone 14 | Perungudi | Rainfall | HMM+MRF | 6h | 3.119 | 6.168 | 1787 |
| Zone  4 | Tondiarpet | Rainfall | HMM+MRF | 24h | 0.502 | 2.640 | 2562 |
| Zone  5 | Royapuram | Rainfall | HMM+MRF | 24h | 0.399 | 2.437 | 2594 |
| Zone 10 | Kodambakkam | Rainfall | HMM+MRF | 24h | 0.873 | 1.082 | 2284 |
| Zone 14 | Perungudi | Rainfall | HMM+MRF | 24h | 3.147 | 6.190 | 1787 |
| Zone  1 | Thiruvottiyur | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  2 | Manali | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  3 | Madhavaram | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  4 | Tondiarpet | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  5 | Royapuram | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  6 | Thiru-Vi-Ka Nagar | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  7 | Ambattur | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  8 | Anna Nagar | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone  9 | Teynampet | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone 10 | Kodambakkam | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone 11 | Valasaravakkam | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone 12 | Alandur | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone 13 | Adyar | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone 14 | Perungudi | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |
| Zone 15 | Sholinganallur | Temperature | HMM+MRF | 24h | 0.752 | 0.989 | 170 |

> **Non-Fabrication Policy for Unmonitored Zones:**  
> In `results/predictions/validation_predictions_mrf.csv`, unmonitored zones have `is_monitored = False` and `actual = NaN`. Inferred spatial field values are computed via graph harmonic extension, but evaluation metrics are computed **strictly on genuinely observed ground truth**.

---

## 6. Saved Model Artifacts

- Calibrated AQI MRF: `models/hmm_mrf/mrf_aqi.joblib`
- Calibrated Rainfall MRF: `models/hmm_mrf/mrf_rainfall.joblib`
- Calibrated Temperature MRF: `models/hmm_mrf/mrf_temperature.joblib`
- Combined MRF Registry: `models/hmm_mrf/mrf_all_models.joblib`
- Calibration Grid Scores: `results/metrics/mrf_calibration_scores.csv`
- Validation Predictions: `results/predictions/validation_predictions_mrf.csv`
- Validation Metrics: `results/metrics/validation_metrics_mrf.csv`
