"""
GMM mixture Hellinger distance via Linear Programming.

For GMMs p_i = sum_k a_ik N(mu_ik, S_ik) and p_j = sum_l a_jl N(mu_jl, S_jl):

D_H(i,j) = min_W sum_{k,l} W_kl * H(component_k of i, component_l of j)
s.t. W_kl >= 0; sum_l W_kl = a_ik (forall k); sum_k W_kl = a_jl (forall l)

This is an optimal transport problem. Solved with scipy.optimize.linprog.
Component Hellinger terms use the closed-form Gaussian values.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import linprog

from .hellinger import hellinger_gaussian, hellinger_gaussian_pairwise


@dataclass
class GMM:
    """Gaussian Mixture Model representation."""
    weights: np.ndarray      # shape (K,) - mixture weights, sum to 1
    means: np.ndarray        # shape (K, 3) - component means
    covs: np.ndarray         # shape (K, 3, 3) - component covariances
    
    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.covs = np.asarray(self.covs, dtype=float)
        
        # Validate
        K = self.weights.shape[0]
        assert self.means.shape == (K, 3), f"means shape mismatch: {self.means.shape} vs ({K}, 3)"
        assert self.covs.shape == (K, 3, 3), f"covs shape mismatch: {self.covs.shape} vs ({K}, 3, 3)"
        assert np.isclose(self.weights.sum(), 1.0), f"weights must sum to 1, got {self.weights.sum()}"
    
    @property
    def n_components(self) -> int:
        return self.weights.shape[0]
    
    @property
    def weighted_mean(self) -> np.ndarray:
        """Mean position of the mixture (weighted average of component means)."""
        return np.sum(self.weights[:, None] * self.means, axis=0)
    
    def is_gaussian(self) -> bool:
        """Check if this is a single Gaussian (K=1)."""
        return self.n_components == 1
    
    def to_gaussian(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert K=1 GMM to (mu, Sigma)."""
        assert self.is_gaussian()
        return self.means[0], self.covs[0]


def hellinger_gmm_component(gmm1: GMM, gmm2: GMM) -> np.ndarray:
    """
    Compute component-wise Hellinger distance matrix between two GMMs.
    
    Parameters
    ----------
    gmm1, gmm2 : GMM
        Two GMM objects
    
    Returns
    -------
    ndarray, shape (K1, K2)
        H[i, j] = Hellinger distance between component i of gmm1 and component j of gmm2
    """
    K1 = gmm1.n_components
    K2 = gmm2.n_components
    H = np.zeros((K1, K2))
    
    for i in range(K1):
        for j in range(K2):
            H[i, j] = hellinger_gaussian(gmm1.means[i], gmm1.covs[i],
                                          gmm2.means[j], gmm2.covs[j])
    
    return H


def _solve_ot_lp(cost_matrix: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Solve optimal transport LP: min_W sum W_kl * C_kl
    s.t. W >= 0, sum_l W_kl = a_k, sum_k W_kl = b_l
    
    Uses scipy.optimize.linprog with method='highs'.
    Drops one redundant equality constraint to avoid numerical infeasibility.
    """
    K1, K2 = cost_matrix.shape
    
    # Flatten cost matrix
    c = cost_matrix.ravel()
    
    # Equality constraints: sum_l W_kl = a_k for all k (K1 constraints)
    #                       sum_k W_kl = b_l for all l (K2 constraints)
    # Total K1 + K2 constraints, but one is redundant (sum a = sum b = 1)
    # Drop the last constraint from the second group
    
    n_vars = K1 * K2
    
    # Build A_eq matrix
    A_eq = np.zeros((K1 + K2 - 1, n_vars))
    
    # Row constraints: sum_l W_kl = a_k
    for k in range(K1):
        for l in range(K2):
            idx = k * K2 + l
            A_eq[k, idx] = 1.0
    
    # Column constraints: sum_k W_kl = b_l (drop last one)
    for l in range(K2 - 1):
        row = K1 + l
        for k in range(K1):
            idx = k * K2 + l
            A_eq[row, idx] = 1.0
    
    b_eq = np.concatenate([a, b[:-1]])
    
    # Bounds: W >= 0
    bounds = [(0, None)] * n_vars
    
    # Solve
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: use product weights approximation
        W = np.outer(a, b)
        return float(np.sum(W * cost_matrix))
    
    return float(result.fun)


def hellinger_gmm_lp(gmm1: GMM, gmm2: GMM) -> float:
    """
    GMM Hellinger distance via linear programming (optimal transport).
    
    Parameters
    ----------
    gmm1, gmm2 : GMM
        Two GMM objects
    
    Returns
    -------
    float
        Hellinger distance in [0, 1]
    """
    # K=1 shortcut - reduces to Gaussian closed form
    if gmm1.is_gaussian() and gmm2.is_gaussian():
        mu1, S1 = gmm1.to_gaussian()
        mu2, S2 = gmm2.to_gaussian()
        return hellinger_gaussian(mu1, S1, mu2, S2)
    
    # Component cost matrix
    cost = hellinger_gmm_component(gmm1, gmm2)
    
    # Solve OT problem
    distance = _solve_ot_lp(cost, gmm1.weights, gmm2.weights)
    
    return distance


def hellinger_gmm_approx(gmm1: GMM, gmm2: GMM) -> float:
    """
    GMM Hellinger distance via product-weights approximation (upper bound).
    
    D_H ≈ sum_{k,l} a_ik * a_jl * H(component_k, component_l)
    
    This is always >= the true OT distance.
    
    Parameters
    ----------
    gmm1, gmm2 : GMM
        Two GMM objects
    
    Returns
    -------
    float
        Approximate Hellinger distance in [0, 1]
    """
    cost = hellinger_gmm_component(gmm1, gmm2)
    W = np.outer(gmm1.weights, gmm2.weights)
    return float(np.sum(W * cost))


def hellinger_gmm(gmm1: GMM, gmm2: GMM, method: str = 'lp') -> float:
    """
    Dispatcher for GMM Hellinger distance.
    
    Parameters
    ----------
    gmm1, gmm2 : GMM
        Two GMM objects
    method : str
        'lp' for optimal transport (exact), 'approx' for product-weights approximation
    
    Returns
    -------
    float
        Hellinger distance in [0, 1]
    """
    if method == 'lp':
        return hellinger_gmm_lp(gmm1, gmm2)
    elif method == 'approx':
        return hellinger_gmm_approx(gmm1, gmm2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lp' or 'approx'")