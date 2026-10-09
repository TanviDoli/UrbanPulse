"""
Script to generate research-grade model_methodology.md for UrbanPulse
documenting the zone-specific HMM architecture, governance, state selection,
and validation metrics.
"""

import pandas as pd
import numpy as np

def generate_report():
    df_status = pd.read_csv('results/metrics/model_training_status.csv')
    df_metrics = pd.read_csv('results/metrics/validation_metrics_hmm.csv')
    df_sel = pd.read_csv('results/metrics/state_selection_hmm.csv')

    # Format status table
    status_rows = []
    for _, r in df_status.iterrows():
        k_str = f"{int(r['selected_k'])}" if pd.notna(r['selected_k']) else "-"
        status_rows.append(
            f"| Zone {int(r['zone_id']):2d} | {r['zone_label']} | {r['variable']} | **{r['status']}** | {int(r['n_train_obs'])} | {int(r['n_val_obs'])} | {k_str} | {r['reason']} |"
        )
    status_table_md = "\n".join(status_rows)

    # Format validation metrics table
    metric_rows = []
    for _, r in df_metrics.iterrows():
        metric_rows.append(
            f"| Zone {int(r['zone_id']):2d} | {r['zone_label']} | {r['variable']} | {r['model']} | {r['horizon']} | {r['mae']:.3f} | {r['rmse']:.3f} | {int(r['sample_count'])} |"
        )
    metrics_table_md = "\n".join(metric_rows)

    content = f"""# UrbanPulse: Hidden Markov Model (HMM) Methodology & Validation Report
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

For zone $z \\in \\{{1, \\dots, 15\\}}$ and environmental variable $v \\in \\{{\\text{{AQI}}, \\text{{Rainfall}}, \\text{{Temperature}}\\}}$, let $Y_{{z, t}} \\in \\mathbb{{R}}$ denote the observation at timestep $t$, and let $S_{{z, t}} \\in \\{{1, 2, \\dots, K_z\\}}$ represent the latent regime.

The generative parameters $\\theta_z = \\{{\\boldsymbol{{\\pi}}_z, \\mathbf{{A}}_z, \\boldsymbol{{\\mu}}_z, \\boldsymbol{{\\sigma}}^2_z\\}}$ are:
1. **Initial State Distribution:**
   $$\\pi_{{z, k}} = P(S_{{z, 1}} = k), \\quad \\sum_{{k=1}}^{{K_z}} \\pi_{{z, k}} = 1$$
2. **Transition Probability Matrix $\\mathbf{{A}}_z \\in \\mathbb{{R}}^{{K_z \\times K_z}}$:**
   $$A_{{z, ij}} = P(S_{{z, t+1}} = j \\mid S_{{z, t}} = i), \\quad \\sum_{{j=1}}^{{K_z}} A_{{z, ij}} = 1$$
3. **Continuous Gaussian Emission Densities:**
   $$P(Y_{{z, t}} = y \\mid S_{{z, t}} = k) = \\frac{{1}}{{\\sqrt{{2\\pi\\sigma_{{z, k}}^2}}}} \\exp\\left( -\\frac{{(y - \\mu_{{z, k}})^2}}{{2\\sigma_{{z, k}}^2}} \\right)$$

### Multi-Step Horizon Projections
Given historical context $\\mathbf{{y}}_{{z, 1:t}}$ up to time $t$:
1. **Filtered State Belief (Forward Posterior):**
   $$\\gamma_{{z, t}}(k) = P(S_{{z, t}} = k \\mid \\mathbf{{y}}_{{z, 1:t}})$$
2. **Horizon $h$ State Projection ($h \\in \\{{1, 3, 6, 24\\}}$ hours):**
   $$\\mathbf{{p}}_{{z, t+h}} = \\boldsymbol{{\\gamma}}_{{z, t}} \\mathbf{{A}}_z^h$$
3. **Point Forecast (Conditional Expectation):**
   $$\\hat{{Y}}_{{z, t+h}} = \\mathbb{{E}}[Y_{{z, t+h}} \\mid \\mathbf{{y}}_{{z, 1:t}}] = \\sum_{{j=1}}^{{K_z}} p_{{z, t+h}}(j) \\mu_{{z, j}}$$
4. **Predictive Variance:**
   $$\\text{{Var}}(Y_{{z, t+h}} \\mid \\mathbf{{y}}_{{z, 1:t}}) = \\sum_{{j=1}}^{{K_z}} p_{{z, t+h}}(j) (\\sigma_{{z, j}}^2 + \\mu_{{z, j}}^2) - \\hat{{Y}}_{{z, t+h}}^2$$

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
  - Requires $N_{{\\text{{train}}}} \\ge 2,000$ valid hourly observations (~83 days).
  - Requires $N_{{\\text{{val}}}} \\ge 500$ valid hourly observations for out-of-sample state selection.
  - *Result for AQI:* 7 zones trained (Zones 2, 4, 5, 10, 12, 13, 14). Zone 1 has $N_{{\\text{{train}}}}=934$ and $N_{{\\text{{val}}}}=0$ (sensor discontinued in March 2024) and is classified as `UNAVAILABLE`. Zones 3, 6, 7, 8, 9, 11, 15 are unmonitored (`UNAVAILABLE`).
  - *Result for Rainfall:* 4 zones trained (Zones 4, 5, 10, 14). All other 11 zones lack continuous operational rainfall sensors and are classified as `UNAVAILABLE`.
- **Daily Variable (Temperature):**
  - Requires $N_{{\\text{{train}}}} \\ge 100$ valid daily observations.
  - IMD daily temperature provides 174 calendar days across all zones in `train.csv`.
  - Because IMD records end on 2024-06-22, `validate.csv` contains 0 observed temperature records. Thus, the previously documented **Temperature Exception** applies: $K$ selection is governed by Bayesian Information Criterion (BIC) on the zone's daily training sequence. All 15 zones train an independent daily HMM on their partition.

---

## 5. Model Training Status by Zone and Variable (45 Combinations)

| Zone ID | Zone Label | Variable | Status | Train Obs | Val Obs | Selected K | Reason / Note |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
{status_table_md}

---

## 6. Validation Results by Zone, Variable, and Horizon

Evaluation performed strictly on held-out `validate.csv` (2025-05-27 to 2025-09-13) for AQI and Rainfall, and on sequential daily forecasting for Temperature:

| Zone ID | Zone Label | Variable | Model | Horizon | MAE | RMSE | Sample Count |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
{metrics_table_md}

---

## 7. Model Artifact Structure

The trained models are explicitly keyed by `variable -> zone_id` and saved under `models/hmm/`:
- **Aggregated Dictionaries:**
  - `models/hmm/hmm_aqi.joblib`: Dict `{{zone_id: TemporalHMM}}` for monitored zones [2, 4, 5, 10, 12, 13, 14]
  - `models/hmm/hmm_rainfall.joblib`: Dict `{{zone_id: TemporalHMM}}` for monitored zones [4, 5, 10, 14]
  - `models/hmm/hmm_temperature.joblib`: Dict `{{zone_id: TemporalHMM}}` for zones [1..15]
  - `models/hmm/hmm_models_by_zone.joblib`: Dict `{{(variable, zone_id): TemporalHMM}}` (26 total models)
- **Individual Zone Artifacts:**
  - `models/hmm/hmm_aqi_zone_{{z}}.joblib`
  - `models/hmm/hmm_rainfall_zone_{{z}}.joblib`
  - `models/hmm/hmm_temperature_zone_{{z}}.joblib`
- **Audit Records:**
  - `results/metrics/model_training_status.csv` (all 45 combinations audited)
  - `results/metrics/validation_metrics_hmm.csv` (59 multi-horizon validation evaluations)
  - `results/metrics/state_selection_hmm.csv` (all candidate K BIC/AIC/val-MAE evaluations)
  - `results/predictions/validation_predictions_hmm.csv` (timestamped point predictions and predictive variances)
"""

    with open('reports/model_methodology.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully generated reports/model_methodology.md")

if __name__ == '__main__':
    generate_report()
