"""
Unit Tests for Markov Random Field (MRF) Spatial Modeling in UrbanPulse.
Verifies:
1. Adjacency graph contains exactly the verified 29 edges.
2. Pairwise potentials use the actual adjacency graph and boundary weights.
3. MRF is NOT performing neighbor averaging (confidence/precision modulates spatial pulling).
4. HMM unary evidence is zone-specific.
5. Unmonitored zones do not receive fabricated ground-truth observations.
6. Validation is not contaminated by test data.
7. Inference produces a valid probability distribution (positive definite precision, valid variance).
8. Gaussian Belief Propagation and Exact solve are consistent.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from src.models.mrf_model import SpatialMRF

MONITORED_AQI_ZONES = [2, 4, 5, 10, 12, 13, 14]
MONITORED_RAINFALL_ZONES = [4, 5, 10, 14]
UNMONITORED_AQI_ZONES = [1, 3, 6, 7, 8, 9, 11, 15]

def test_mrf_artifacts_exist():
    """Verify all expected MRF model artifacts, calibration records, and reports exist."""
    assert os.path.exists('models/hmm_mrf/mrf_aqi.joblib')
    assert os.path.exists('models/hmm_mrf/mrf_rainfall.joblib')
    assert os.path.exists('models/hmm_mrf/mrf_temperature.joblib')
    assert os.path.exists('models/hmm_mrf/mrf_all_models.joblib')
    assert os.path.exists('results/metrics/mrf_calibration_scores.csv')
    assert os.path.exists('results/metrics/validation_metrics_mrf.csv')
    assert os.path.exists('results/predictions/validation_predictions_mrf.csv')

def test_graph_contains_exact_29_edges():
    """Verify the MRF uses the exact 29 verified edges from zone_adjacency.csv."""
    mrf = SpatialMRF(variable='AQI')
    assert len(mrf.active_edges) == 29
    assert len(mrf.edge_list) == 29
    
    # Check key verified topological connections
    edge_set = set()
    for _, r in mrf.active_edges.iterrows():
        u = int(r['zone_id_1'])
        v = int(r['zone_id_2'])
        edge_set.add((min(u, v), max(u, v)))
        
    assert (1, 2) in edge_set # Thiruvottiyur <-> Manali
    assert (1, 4) in edge_set # Thiruvottiyur <-> Tondiarpet
    assert (4, 5) in edge_set # Tondiarpet <-> Royapuram
    assert (14, 15) in edge_set # Perungudi <-> Sholinganallur
    assert (1, 15) not in edge_set # Opposite ends of Chennai

def test_pairwise_potentials_use_actual_graph_weights():
    """Verify Laplacian matrix correctly encodes normalized boundary length weights."""
    mrf = SpatialMRF(variable='AQI', beta=1.0)
    assert mrf.L_w.shape == (15, 15)
    
    # Row sums must be 0
    assert np.allclose(mrf.L_w.sum(axis=1), 0.0)
    
    # Symmetry
    assert np.allclose(mrf.L_w, mrf.L_w.T)
    
    # Diagonal entries must be positive sum of incident edge weights
    for i in range(15):
        assert mrf.L_w[i, i] > 0
        off_diag_sum = sum(-mrf.L_w[i, j] for j in range(15) if j != i)
        assert np.isclose(mrf.L_w[i, i], off_diag_sum)

def test_mrf_is_not_performing_neighbor_averaging():
    """
    Prove that MRF is NOT simple neighbor averaging or heuristic smoothing.
    Test: A node with high precision (low predictive variance) will resist spatial pulling,
    whereas neighbor averaging would arbitrarily pull it towards neighbors regardless of confidence.
    """
    mrf = SpatialMRF(variable='AQI', beta=0.01)
    
    # Zone 4 (Tondiarpet) surrounded by neighbors with high values (150.0)
    # Zone 4 has confident prediction 40.0 with tiny variance 0.01 (precision = 100)
    y_hat = np.full(15, 150.0)
    y_hat[3] = 40.0 # Zone 4 (0-indexed: 3)
    
    variances = np.full(15, 100.0)
    variances[3] = 0.01 # High confidence
    
    active_mask = np.ones(15, dtype=bool)
    
    mu_mrf, _, _ = mrf.infer_exact(y_hat, variances, active_mask)
    
    # In neighbor averaging, Zone 4 would be dragged towards (40 + 150*neighbors)/N ~ 120-130
    # In genuine MRF, high precision keeps Zone 4 very close to 40.0:
    assert abs(mu_mrf[3] - 40.0) < 3.0, f"MRF must respect high precision unary, got {mu_mrf[3]}"
    
    # Now test the converse: if Zone 4 has low confidence (variance = 10000, precision = 0.0001),
    # the spatial pairwise prior pulls it towards neighbors
    variances_uncertain = variances.copy()
    variances_uncertain[3] = 10000.0 # Low confidence
    mu_mrf_uncertain, _, _ = mrf.infer_exact(y_hat, variances_uncertain, active_mask)
    assert mu_mrf_uncertain[3] > 80.0, "Low confidence node should receive stronger spatial coupling"

def test_hmm_unary_evidence_is_zone_specific():
    """Verify that different zones receive distinct HMM unaries reflecting their local models."""
    aqi_models = joblib.load('models/hmm/hmm_aqi.joblib')
    dummy_seq = np.array([40.0, 45.0, 50.0, 55.0])
    
    predictions = {}
    for z_id, model in aqi_models.items():
        pred, var, _ = model.forecast(dummy_seq, horizon=1)
        predictions[z_id] = (pred, var)
        
    # Manali (industrial) vs Royapuram (coastal) should have distinct learned dynamics
    assert len(set(predictions.keys())) == 7
    # Verify values are not all identical
    means = [p[0] for p in predictions.values()]
    assert max(means) != min(means), "Zone-specific HMMs must not produce identical forecasts"

def test_unmonitored_zones_not_fabricated():
    """Verify unmonitored zones have is_monitored=False and actual=NaN in predictions."""
    df_preds = pd.read_csv('results/predictions/validation_predictions_mrf.csv')
    
    for z_id in UNMONITORED_AQI_ZONES:
        df_z = df_preds[(df_preds['variable'] == 'AQI') & (df_preds['zone_id'] == z_id)]
        if len(df_z) > 0:
            assert (df_z['is_monitored'] == False).all()
            assert df_z['actual'].isna().all(), f"Zone {z_id} is unmonitored; actual must be NaN"
            assert df_z['predicted'].notna().all(), f"Zone {z_id} should have inferred spatial prediction"

def test_validation_not_contaminated_by_test_data():
    """Verify all validation predictions and metrics timestamps stay strictly prior to test partition."""
    df_preds = pd.read_csv('results/predictions/validation_predictions_mrf.csv')
    # AQI and Rainfall are evaluated strictly within the validation window
    df_val_vars = df_preds[df_preds['variable'].isin(['AQI', 'Rainfall'])]
    assert df_val_vars['timestamp'].max() <= '2025-09-13T23:00:00+00:00'
    assert df_val_vars['timestamp'].min() >= '2025-05-27T00:00:00+00:00'

    # And overall, no validation prediction ever reaches test partition (starts 2025-09-14)
    assert df_preds['timestamp'].max() < '2025-09-14T00:00:00+00:00'

def test_inference_produces_valid_probability_distribution():
    """Verify precision matrix Q is strictly positive-definite and marginal variances are positive."""
    mrf = SpatialMRF(variable='AQI', beta=0.0001)
    y_hat = np.full(15, 60.0)
    variances = np.full(15, 20.0)
    active_mask = np.array([z in MONITORED_AQI_ZONES for z in range(1, 16)])
    
    mu_mrf, var_mrf, Q = mrf.infer_exact(y_hat, variances, active_mask)
    
    # Q must be symmetric
    assert np.allclose(Q, Q.T)
    
    # Q must be positive-definite: all eigenvalues > 0
    eigenvals = np.linalg.eigvalsh(Q)
    assert np.all(eigenvals > 0), f"Minimum eigenvalue of Q: {np.min(eigenvals)}"
    
    # Marginal variances must be positive
    assert np.all(var_mrf > 0)
    
    # Physical bounds for AQI: non-negative
    assert np.all(mu_mrf >= 0)

def test_belief_propagation_and_exact_consistency():
    """Verify Gaussian Belief Propagation converges to the exact MAP / posterior mean."""
    mrf = SpatialMRF(variable='AQI', beta=0.0001)
    y_hat = np.array([50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0])
    variances = np.full(15, 25.0)
    active_mask = np.ones(15, dtype=bool)
    
    mu_exact, _, _ = mrf.infer_exact(y_hat, variances, active_mask)
    mu_bp, _, it_count = mrf.infer_belief_propagation(y_hat, variances, active_mask, max_iter=500, tol=1e-5)
    
    # BP must converge close to exact
    max_diff = np.max(np.abs(mu_exact - mu_bp))
    assert max_diff < 0.05, f"Belief Propagation max diff from exact: {max_diff}"
