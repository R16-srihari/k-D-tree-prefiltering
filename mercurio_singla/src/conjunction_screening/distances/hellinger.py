"""Hellinger distance implementation for Gaussian and GMM distributions.

For two Gaussians p = N(mu1, S1) and q = N(mu2, S2):

    BC(p, q) = |S1|^{1/4} |S2|^{1/4} / |Sbar|^{1/2} * exp(-1/8 * d^T Sbar^{-1} d)
    H^2(p, q) = 1 - BC(p, q)
    H = sqrt(H^2) in [0, 1]

where Sbar = (S1 + S2) / 2, d = mu2 - mu1.

For GMMs, the component-level Hellinger distances form a cost matrix and the
mixture distance is the optimal-transport coupling of mixture weights:

    D(P, Q) = min_W sum_{i,j} W_ij * d_ij
    s.t. sum_j W_ij = alpha_i, sum_i W_ij = beta_j, W_ij >= 0
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import inv
from scipy.optimize import linprog

from ..models import Gaussian, GaussianMixture
from .base import ProbabilisticDistance


def _hellinger_gaussian_core(mu1: np.ndarray, S1: np.ndarray,
                             mu2: np.ndarray, S2: np.ndarray) -> float:
    """Core Hellinger distance between two Gaussians.

    Returns H in [0, 1]. Identical distributions give 0; widely separated
    distributions approach 1.
    """
    d = mu2 - mu1
    Sbar = 0.5 * (S1 + S2)

    sign1, logdet1 = slogdet(S1)
    sign2, logdet2 = slogdet(S2)
    sign_bar, logdet_bar = slogdet(Sbar)

    if sign1 <= 0 or sign2 <= 0 or sign_bar <= 0:
        # Non-positive definite covariance: cannot compute closed form.
        return 1.0

    # log(BC) = 0.25 * (logdet1 + logdet2) - 0.5 * logdet_bar
    log_bc = 0.25 * (logdet1 + logdet2) - 0.5 * logdet_bar

    # Mahalanobis term: -0.125 * d^T Sbar^{-1} d
    try:
        Sbar_inv = inv(Sbar)
        mahal = float(d @ Sbar_inv @ d)
        log_bc -= 0.125 * mahal
    except np.linalg.LinAlgError:
        return 1.0

    # Clamp for numerical stability
    log_bc = float(np.clip(log_bc, -700.0, 0.0))
    bc = float(np.clip(np.exp(log_bc), 0.0, 1.0))

    h2 = float(np.clip(1.0 - bc, 0.0, 1.0))
    return float(np.sqrt(h2))


def _solve_ot_lp(cost_matrix: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Solve the optimal-transport LP for mixture weights.

    min_W sum_{k,l} W_kl * C_kl
    s.t. W >= 0, sum_l W_kl = a_k, sum_k W_kl = b_l

    One redundant equality constraint is dropped (sum a = sum b = 1) to
    avoid numerical infeasibility.
    """
    K1, K2 = cost_matrix.shape
    n_vars = K1 * K2

    c = cost_matrix.ravel()

    A_eq = np.zeros((K1 + K2 - 1, n_vars))
    # Row constraints: sum_l W_kl = a_k
    for k in range(K1):
        for l in range(K2):
            A_eq[k, k * K2 + l] = 1.0
    # Column constraints: sum_k W_kl = b_l (drop last)
    for l in range(K2 - 1):
        row = K1 + l
        for k in range(K1):
            A_eq[row, k * K2 + l] = 1.0

    b_eq = np.concatenate([a, b[:-1]])
    bounds = [(0, None)] * n_vars

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        # Fallback: product-weights approximation (upper bound)
        W = np.outer(a, b)
        return float(np.sum(W * cost_matrix))

    return float(result.fun)


def hellinger_gmm_component(gmm1: GaussianMixture, gmm2: GaussianMixture) -> np.ndarray:
    """Component-wise Hellinger distance matrix between two GMMs.

    Parameters
    ----------
    gmm1, gmm2 : GaussianMixture

    Returns
    -------
    np.ndarray, shape (K1, K2)
        H[i, j] = Hellinger distance between component i of gmm1 and
        component j of gmm2.
    """
    K1 = gmm1.n_components
    K2 = gmm2.n_components
    H = np.zeros((K1, K2))
    for i in range(K1):
        for j in range(K2):
            H[i, j] = _hellinger_gaussian_core(
                gmm1.components[i].mean, gmm1.components[i].covariance,
                gmm2.components[j].mean, gmm2.components[j].covariance,
            )
    return H


def hellinger_gmm_lp(gmm1: GaussianMixture, gmm2: GaussianMixture) -> float:
    """GMM Hellinger distance via linear programming (optimal transport)."""
    if gmm1.is_gaussian() and gmm2.is_gaussian():
        return _hellinger_gaussian_core(
            gmm1.components[0].mean, gmm1.components[0].covariance,
            gmm2.components[0].mean, gmm2.components[0].covariance,
        )

    cost = hellinger_gmm_component(gmm1, gmm2)
    return _solve_ot_lp(cost, gmm1.weights, gmm2.weights)


def hellinger_gmm_approx(gmm1: GaussianMixture, gmm2: GaussianMixture) -> float:
    """GMM Hellinger distance via product-weights approximation (upper bound).

    D_H ≈ sum_{k,l} a_k * b_l * H(component_k, component_l)
    """
    cost = hellinger_gmm_component(gmm1, gmm2)
    W = np.outer(gmm1.weights, gmm2.weights)
    return float(np.sum(W * cost))


def hellinger_gmm(gmm1: GaussianMixture, gmm2: GaussianMixture,
                method: str = "lp") -> float:
    """Dispatcher for GMM Hellinger distance.

    Parameters
    ----------
    gmm1, gmm2 : GaussianMixture
    method : str
        "lp" for optimal transport (exact), "approx" for product-weights.
    """
    if method == "lp":
        return hellinger_gmm_lp(gmm1, gmm2)
    if method == "approx":
        return hellinger_gmm_approx(gmm1, gmm2)
    raise ValueError(f"Unknown method: {method!r}. Use 'lp' or 'approx'.")


class HellingerDistance(ProbabilisticDistance):
    """Hellinger distance between uncertainty distributions.

    Supports Gaussian-Gaussian (closed form), Gaussian-GMM, and GMM-GMM
    pairs. A Gaussian is treated as a single-component GMM internally.
    """

    def __init__(self, method: str = "lp") -> None:
        """Parameters
        ----------
        method : str
            GMM solution method: "lp" (optimal transport) or "approx".
        """
        self.method = method

    def compute(self, distribution_a, distribution_b) -> float:
        """Compute the Hellinger distance between two distributions.

        Parameters
        ----------
        distribution_a : Gaussian | GaussianMixture
        distribution_b : Gaussian | GaussianMixture

        Returns
        -------
        float
            Hellinger distance in [0, 1].
        """
        from ..models import Gaussian, GaussianMixture

        # Normalize inputs to GaussianMixture for uniform handling
        def _as_gmm(dist):
            if isinstance(dist, GaussianMixture):
                return dist
            if isinstance(dist, Gaussian):
                return GaussianMixture(
                    weights=np.array([1.0]),
                    components=[dist],
                )
            raise TypeError(
                f"Unsupported distribution type: {type(dist).__name__}. "
                "Expected Gaussian or GaussianMixture."
            )

        gmm_a = _as_gmm(distribution_a)
        gmm_b = _as_gmm(distribution_b)
        return hellinger_gmm(gmm_a, gmm_b, method=self.method)
