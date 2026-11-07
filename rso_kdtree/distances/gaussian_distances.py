"""Gaussian-level probabilistic distance measures.

Implements four distances between 3D Gaussian distributions:

- Hellinger distance (metric, in [0, 1])
- Wasserstein-2 distance (metric)
- Mahalanobis distance (pooled covariance, not a full distribution distance)
- Symmetric KL divergence (divergence, not a metric)

All use slogdet and linear solves for numerical stability.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import slogdet
from scipy.linalg import inv, solve

from .base import SUPPORTED_METRICS, ProbabilisticDistance


def _validate_gaussian(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate a Gaussian mean/covariance pair."""
    mean = np.asarray(mean, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    if mean.shape != (3,):
        raise ValueError(f"Gaussian mean must have shape (3,), got {mean.shape}")
    if cov.shape != (3, 3):
        raise ValueError(f"Gaussian covariance must have shape (3, 3), got {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("Gaussian covariance must be symmetric")
    return mean, cov


def _psd_sqrtm(cov: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition.

    Clamps small negative eigenvalues from numerical roundoff to zero.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    sqrt_evals = np.sqrt(eigenvalues)
    return (eigenvectors * sqrt_evals) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Gaussian-level distances
# ---------------------------------------------------------------------------

def hellinger_gaussian(mean_a, cov_a, mean_b, cov_b) -> float:
    r"""Hellinger distance between two 3D Gaussians.

    H = \sqrt{1 - \frac{|\Sigma_p|^{1/4}|\Sigma_q|^{1/4}}
                     {|\Sigma_{pq}|^{1/2}}
                 \exp\!\Big[-\tfrac12(\mu_p-\mu_q)^T\Sigma_{pq}^{-1}
                              (\mu_p-\mu_q)\Big]}
    
    Returns values in [0, 1]. Identical distributions give 0.
    """
    mean_a, cov_a = _validate_gaussian(mean_a, cov_a)
    mean_b, cov_b = _validate_gaussian(mean_b, cov_b)

    d = mean_a - mean_b
    sbar = 0.5 * (cov_a + cov_b)

    sign_a, logdet_a = slogdet(cov_a)
    sign_b, logdet_b = slogdet(cov_b)
    sign_bar, logdet_bar = slogdet(sbar)

    if sign_a <= 0 or sign_b <= 0 or sign_bar <= 0:
        return 1.0

    log_bc_coeff = 0.25 * (logdet_a + logdet_b) - 0.5 * logdet_bar
    log_bc = 0

    try:
        mahal = d @ solve(sbar, d)
        log_bc -= 0.5 * mahal
    except np.linalg.LinAlgError:
        return 1.0

    log_bc = float(np.clip(log_bc, -700.0, 0.0))
    bc = float(np.clip(np.exp(log_bc_coeff+log_bc), 0.0, 1.0))
    h2 = float(np.clip(1.0 - bc, 0.0, 1.0))
    return float(np.sqrt(h2))


def wasserstein_gaussian(mean_a, cov_a, mean_b, cov_b) -> float:
    """Gaussian 2-Wasserstein distance.

    W_2^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2*(S1^{1/2} S2 S1^{1/2})^{1/2})
    """
    mean_a, cov_a = _validate_gaussian(mean_a, cov_a)
    mean_b, cov_b = _validate_gaussian(mean_b, cov_b)

    mean_diff = mean_a - mean_b
    mean_term = float(mean_diff @ mean_diff)

    sqrt_a = _psd_sqrtm(cov_a)
    # M = sqrt(S1) @ S2 @ sqrt(S1)
    m = sqrt_a @ cov_b @ sqrt_a
    # Symmetrize to reduce numerical asymmetry
    m = 0.5 * (m + m.T)
    sqrt_m = _psd_sqrtm(m)

    trace_term = float(np.trace(cov_a + cov_b - 2.0 * sqrt_m))
    w2_squared = mean_term + trace_term
    w2_squared = max(w2_squared, 0.0)  # guard against small negative from roundoff
    return float(np.sqrt(w2_squared))


def mahalanobis_gaussian(mean_a, cov_a, mean_b, cov_b) -> float:
    """Pooled-covariance Mahalanobis distance between two Gaussians.

    S_pool = (S1 + S2) / 2
    D = sqrt(d^T S_pool^{-1} d)

    Note: compares mean separation normalized by pooled covariance;
    does not independently account for covariance differences.
    """
    mean_a, cov_a = _validate_gaussian(mean_a, cov_a)
    mean_b, cov_b = _validate_gaussian(mean_b, cov_b)

    d = mean_a - mean_b
    s_pool = 0.5 * (cov_a + cov_b)

    try:
        d_mahal = d @ solve(s_pool, d)
    except np.linalg.LinAlgError:
        return float("inf")

    d_mahal = max(d_mahal, 0.0)
    return float(np.sqrt(d_mahal))


def symmetric_kl_gaussian(mean_a, cov_a, mean_b, cov_b) -> float:
    """Symmetric KL divergence between two 3D Gaussians.

    D_SKL = 1/4 [ tr(S2^{-1} S1 + S1^{-1} S2) - 2d
                  + d^T (S1^{-1} + S2^{-1}) d ]
    where d = 3.

    This is a divergence, not a metric (triangle inequality does not hold).
    """
    mean_a, cov_a = _validate_gaussian(mean_a, cov_a)
    mean_b, cov_b = _validate_gaussian(mean_b, cov_b)

    d = mean_a - mean_b
    dim = 3

    try:
        inv_a = inv(cov_a)
        inv_b = inv(cov_b)
    except np.linalg.LinAlgError:
        return float("inf")

    trace_term = float(np.trace(inv_b @ cov_a + inv_a @ cov_b))
    mean_term = float(d @ (inv_a + inv_b) @ d)

    skl = 0.25 * (trace_term - 2.0 * dim + mean_term)
    skl = max(skl, 0.0)  # guard against small negative from roundoff
    return float(skl)


# ---------------------------------------------------------------------------
# Distance class implementations
# ---------------------------------------------------------------------------

class HellingerDistance(ProbabilisticDistance):
    """Hellinger distance between GMMs via component-coupling LP."""

    name = "hellinger" 

    def compute(self, gmm_a, gmm_b) -> float:
        from .gmm_distances import gmm_distance
        return gmm_distance(gmm_a, gmm_b, "hellinger")

    def compute_gaussian(self, mean_a, cov_a, mean_b, cov_b) -> float:
        return hellinger_gaussian(mean_a, cov_a, mean_b, cov_b)


class WassersteinDistance(ProbabilisticDistance):
    """Wasserstein-2 distance between GMMs via component-coupling LP."""

    name = "wasserstein"

    def compute(self, gmm_a, gmm_b) -> float:
        from .gmm_distances import gmm_distance
        return gmm_distance(gmm_a, gmm_b, "wasserstein")

    def compute_gaussian(self, mean_a, cov_a, mean_b, cov_b) -> float:
        return wasserstein_gaussian(mean_a, cov_a, mean_b, cov_b)


class MahalanobisDistance(ProbabilisticDistance):
    """Pooled-covariance Mahalanobis distance between GMMs via component-coupling LP."""

    name = "mahalanobis"

    def compute(self, gmm_a, gmm_b) -> float:
        from .gmm_distances import gmm_distance
        return gmm_distance(gmm_a, gmm_b, "mahalanobis")

    def compute_gaussian(self, mean_a, cov_a, mean_b, cov_b) -> float:
        return mahalanobis_gaussian(mean_a, cov_a, mean_b, cov_b)


class SymmetricKLDistance(ProbabilisticDistance):
    """Symmetric KL divergence between GMMs via component-coupling LP."""

    name = "kl"

    def compute(self, gmm_a, gmm_b) -> float:
        from .gmm_distances import gmm_distance
        return gmm_distance(gmm_a, gmm_b, "kl")

    def compute_gaussian(self, mean_a, cov_a, mean_b, cov_b) -> float:
        return symmetric_kl_gaussian(mean_a, cov_a, mean_b, cov_b)


# Map metric name to Gaussian-level function
GAUSSIAN_DISTANCE_FNS = {
    "hellinger": hellinger_gaussian,
    "wasserstein": wasserstein_gaussian,
    "mahalanobis": mahalanobis_gaussian,
    "kl": symmetric_kl_gaussian,
}


def get_gaussian_distance_fn(metric_name: str):
    """Return the Gaussian-level distance function for a metric name."""
    if metric_name not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric: {metric_name!r}. "
            f"Supported: {sorted(SUPPORTED_METRICS)}"
        )
    return GAUSSIAN_DISTANCE_FNS[metric_name]
