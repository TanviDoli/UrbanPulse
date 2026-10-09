"""
Exact Probabilistic Markov Random Field (MRF) for Spatial Modeling in UrbanPulse.
Strictly follows research-grade specifications:
1. Continuous Gaussian Markov Random Field over Chennai's 15 administrative zones.
2. Uses the verified 29 shared-boundary edges and normalized weights w_uv = L_uv / L_bar.
3. Energy formulation:
   E(X) = 0.5 * sum_{u in Monitored} lambda_u * (X_u - y_hat_u)^2 + 0.5 * beta * sum_{(u,v) in E} w_{uv} * (X_u - X_v)^2
   P(X | observations) \\propto \\exp(-E(X))
4. Exact Gaussian Inference (Production Method):
   Q = Lambda + beta * L_w
   b = Lambda * y_hat
   mu_MRF = Q^{-1} b
   Sigma_MRF = Q^{-1}
5. Unmonitored Zone Treatment:
   - Observation evidence is genuinely absent: lambda_u = 0.0, b_u = 0.0.
   - NO regional_mean, NO global_mean, NO synthetic zero, NO fabricated observation.
   - Inferred values in unmonitored zones are determined purely by spatial boundary coupling.
   - For beta > 0 the connected Laplacian ensures Q is SPD.
   - For beta = 0 (degenerate / no-coupling candidate during grid search) unmonitored
     zones create zero eigenvalues; a standard Tikhonov regularizer eps*I (eps~1e-9) is
     applied to restore SPD without introducing any observation bias.
6. Strict Isolation: test.csv is never loaded or used.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class SpatialMRF:
    """
    Continuous Gaussian Markov Random Field over the 15-zone Chennai graph.
    Combines zone-specific HMM temporal predictive distributions with spatial pairwise potentials.
    Production inference: Exact Gaussian solve via precision matrix decomposition.
    """
    def __init__(
        self,
        variable: str = 'AQI',
        beta: float = 0.0001,
        adjacency_path: str = 'data/processed/zone_adjacency.csv'
    ):
        self.variable = variable
        self.beta = beta
        self.adjacency_path = adjacency_path
        self.n_nodes = 15
        
        # Load and verify 29 edges
        self.edges_df = pd.read_csv(adjacency_path)
        self.active_edges = self.edges_df[self.edges_df['adjacent'] == 1].copy()
        assert len(self.active_edges) == 29, f"Expected 29 verified edges, found {len(self.active_edges)}"
        
        # Calculate mean boundary length and normalized weights w_uv = L_uv / L_bar
        self.mean_boundary_length = float(self.active_edges['shared_boundary_length'].mean())
        self.active_edges['weight'] = self.active_edges['shared_boundary_length'] / self.mean_boundary_length
        
        # Build normalized weighted Laplacian matrix L_w (15 x 15)
        self.L_w = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        self.edge_list = []
        for _, r in self.active_edges.iterrows():
            u = int(r['zone_id_1']) - 1 # 0-indexed internally
            v = int(r['zone_id_2']) - 1
            w = float(r['weight'])
            self.edge_list.append((u, v, w))
            self.L_w[u, v] -= w
            self.L_w[v, u] -= w
            self.L_w[u, u] += w
            self.L_w[v, v] += w
            
        assert np.allclose(self.L_w.sum(axis=1), 0.0), "Laplacian rows must sum to zero"
        assert np.allclose(self.L_w, self.L_w.T), "Laplacian must be symmetric"

    def compute_energy(self, X: np.ndarray, y_hat: np.ndarray, lambdas: np.ndarray) -> float:
        """
        Compute total MRF energy:
        E(X) = 0.5 * sum_u lambda_u * (X_u - y_hat_u)^2 + 0.5 * beta * X^T L_w X
        """
        unary_energy = 0.5 * np.sum(lambdas * (X - y_hat)**2)
        pairwise_energy = 0.5 * self.beta * float(X.T @ self.L_w @ X)
        return float(unary_energy + pairwise_energy)

    def infer_exact(
        self,
        y_hat: np.ndarray,
        variances: np.ndarray,
        active_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact Gaussian MRF Inference (Production Method).
        
        Q = Lambda + beta * L_w
        b = Lambda * y_hat
        mu = Q^{-1} b
        Sigma = Q^{-1}
        
        Strict Non-Fabrication Rules:
        - For monitored zones with active forecasts:
            lambda_i = 1.0 / max(1e-4, variances[i])
            b_i = lambda_i * y_hat[i]
        - For unmonitored / unavailable zones:
            lambda_i = 0.0  (Strictly zero observation precision)
            b_i = 0.0       (Strictly zero observation evidence)
            NO regional_mean, NO global_mean, NO zero-fill, NO prior mean.
        
        Returns:
            mu_mrf: Posterior expected values across all 15 zones (15,)
            var_mrf: Posterior marginal variances across all 15 zones (15,)
            Q: Precision matrix (15, 15)
        """
        lambdas = np.zeros(self.n_nodes, dtype=float)
        b = np.zeros(self.n_nodes, dtype=float)
        
        for i in range(self.n_nodes):
            if active_mask[i] and pd.notna(y_hat[i]) and pd.notna(variances[i]) and variances[i] > 0:
                lambdas[i] = 1.0 / max(1e-4, float(variances[i]))
                b[i] = lambdas[i] * float(y_hat[i])
            else:
                # Genuinely absent observation evidence
                lambdas[i] = 0.0
                b[i] = 0.0
                
        # Precision matrix Q = Lambda + beta * L_w
        Q = np.diag(lambdas) + self.beta * self.L_w
        
        # Verify symmetry
        assert np.allclose(Q, Q.T), "Precision matrix Q must be symmetric"
        
        # Ensure strict positive-definiteness.
        # When beta=0 and unmonitored zones exist (lambda_u=0), Q is singular.
        # A Tikhonov regularizer eps*I (eps=1e-9) is added as a standard numerical
        # stabilization — it does NOT introduce any observation or prior mean.
        # For all beta > 0, the connected Laplacian ensures SPD and eps has no effect.
        eigenvals = np.linalg.eigvalsh(Q)
        if np.min(eigenvals) <= 0:
            eps = max(1e-9, -np.min(eigenvals) + 1e-9)
            Q = Q + eps * np.eye(self.n_nodes)
            eigenvals = np.linalg.eigvalsh(Q)
        assert np.min(eigenvals) > 0, f"Q not SPD even after regularization. Min eig: {np.min(eigenvals)}"
        
        # Solve Q * mu = b via Cholesky factorization
        # Since Q is SPD, solve is exact and numerically stable
        mu_mrf = np.linalg.solve(Q, b)
        
        # Marginal variances from diagonal of inverse precision matrix
        inv_Q = np.linalg.inv(Q)
        var_mrf = np.diag(inv_Q)
        
        # Enforce physical non-negativity for continuous concentrations/rates
        if self.variable in ['AQI', 'rainfall']:
            mu_mrf = np.maximum(0.0, mu_mrf)
            
        return mu_mrf, var_mrf, Q

    def infer_belief_propagation(
        self,
        y_hat: np.ndarray,
        variances: np.ndarray,
        active_mask: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Iterative Gaussian Loopy Belief Propagation (GaBP) / message passing.
        Updates node beliefs iteratively toward the stationary Gaussian posterior:
          mu^(k+1) = D_Q^-1 (b - (Q - D_Q) mu^(k))
        Converges to the exact posterior mean on strictly diagonally dominant Q.
        """
        mu_exact, var_exact, Q = self.infer_exact(y_hat, variances, active_mask)
        
        # Setup GaBP message passing
        lambdas = np.zeros(self.n_nodes, dtype=float)
        b = np.zeros(self.n_nodes, dtype=float)
        for i in range(self.n_nodes):
            if active_mask[i] and pd.notna(y_hat[i]) and pd.notna(variances[i]) and variances[i] > 0:
                lambdas[i] = 1.0 / max(1e-4, float(variances[i]))
                b[i] = lambdas[i] * float(y_hat[i])

        D_diag = np.diag(Q)
        R = Q - np.diag(D_diag)
        mu = np.copy(y_hat)
        
        it_count = 0
        for it in range(max_iter):
            it_count += 1
            mu_next = (b - R @ mu) / D_diag
            if np.max(np.abs(mu_next - mu)) < tol:
                mu = mu_next
                break
            mu = mu_next

        if self.variable in ['AQI', 'rainfall']:
            mu = np.maximum(0.0, mu)

        return mu, var_exact, it_count

    def infer(
        self,
        y_hat: np.ndarray,
        variances: np.ndarray,
        active_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Production inference entry point.
        Executes exact Gaussian MRF inference.
        """
        mu, var, _ = self.infer_exact(y_hat, variances, active_mask)
        return mu, var

    def save(self, filepath: str):
        """Save MRF model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved SpatialMRF model for {self.variable} to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SpatialMRF':
        """Load MRF model artifact from disk."""
        return joblib.load(filepath)
