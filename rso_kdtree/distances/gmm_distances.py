"""GMM distance computation via component-coupling optimal transport (LP).

For GMMs P = sum_i alpha_i P_i and Q = sum_j beta_j Q_j:

    min_W  sum_{i,j} W_ij * C_ij
    s.t.   sum_j W_ij = alpha_i   (forall i)
           sum_i W_ij = beta_j    (forall j)
           W_ij >= 0

where C_ij = d(P_i, Q_j) is the Gaussian-level distance between components.

Solved with scipy.optimize.linprog (HiGHS solver).

Note: For Wasserstein this is a transport approximation. For Hellinger,
Mahalanobis, and KL it is a component-level comparison, not the exact
distribution-level distance.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog

from .gaussian_distances import get_gaussian_distance_fn


def _build_cost_matrix(gmm_a, gmm_b, distance_fn) -> np.ndarray:
    """Build the component cost matrix C_ij = d(component_i of a, component_j of b)."""
    means_a = gmm_a.means
    covs_a = gmm_a.covariances
    means_b = gmm_b.means
    covs_b = gmm_b.covariances

    n_a = gmm_a.n_components
    n_b = gmm_b.n_components
    cost = np.zeros((n_a, n_b))

    for i in range(n_a):
        for j in range(n_b):
            cost[i, j] = distance_fn(means_a[i], covs_a[i], means_b[j], covs_b[j])

    return cost


def _solve_ot_lp(cost_matrix: np.ndarray, weights_a: np.ndarray, weights_b: np.ndarray) -> float:
    """Solve the optimal transport LP via scipy.optimize.linprog (HiGHS).

    Drops one redundant equality constraint to avoid numerical infeasibility.
    Falls back to product-weights approximation if the LP fails.
    """
    n_a, n_b = cost_matrix.shape
    n_vars = n_a * n_b

    c = cost_matrix.ravel()

    # Equality constraints: row sums = weights_a (n_a constraints)
    #                       col sums = weights_b (n_b constraints, drop last)
    n_constraints = n_a + n_b - 1
    a_eq = np.zeros((n_constraints, n_vars))

    # Row constraints: sum_j W_ij = a_i
    for i in range(n_a):
        for j in range(n_b):
            idx = i * n_b + j
            a_eq[i, idx] = 1.0

    # Column constraints: sum_i W_ij = b_j (drop last column constraint)
    for j in range(n_b - 1):
        row = n_a + j
        for i in range(n_a):
            idx = i * n_b + j
            a_eq[row, idx] = 1.0

    b_eq = np.concatenate([weights_a, weights_b[:-1]])

    bounds = [(0, None)] * n_vars

    result = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        # Fallback: product-weights approximation (upper bound for OT)
        w = np.outer(weights_a, weights_b)
        return float(np.sum(w * cost_matrix))

    return float(result.fun)


def gmm_distance(gmm_a, gmm_b, metric_name: str) -> float:
    """Compute GMM distance via component-coupling LP.

    Parameters
    ----------
    gmm_a, gmm_b : GaussianMixture
        The two GMMs.
    metric_name : str
        One of 'hellinger', 'wasserstein', 'mahalanobis', 'kl'.

    Returns
    -------
    float
        GMM distance.

    Notes
    -----
    Per the algorithm note (Distance_formulae_algo.md), the transport
    ground cost is the *squared* Gaussian distance for Wasserstein-2
    (section 3.2: C_ij = W_2^2) and Mahalanobis (section 4.4:
    C_ik = D_M^2). Wasserstein additionally takes a square root of the
    optimal-transport objective; Mahalanobis does not.
    """
    distance_fn = get_gaussian_distance_fn(metric_name)

    # Single-Gaussian shortcut: closed-form Gaussian distance.
    if gmm_a.is_single_gaussian() and gmm_b.is_single_gaussian():
        mu_a, cov_a = gmm_a.to_single_gaussian()
        mu_b, cov_b = gmm_b.to_single_gaussian()
        value = distance_fn(mu_a, cov_a, mu_b, cov_b)
        if metric_name == "mahalanobis":
            # OT-Mahalanobis uses squared component cost and no square root,
            # so a single component yields D_M^2.
            return value * value
        return value

    cost = _build_cost_matrix(gmm_a, gmm_b, distance_fn)

    # Squared component costs for Wasserstein-2 and Mahalanobis.
    if metric_name in ("wasserstein", "mahalanobis"):
        cost = cost * cost

    result = _solve_ot_lp(cost, gmm_a.weights, gmm_b.weights)

    if metric_name == "wasserstein":
        # Mixture Wasserstein is [min Γ·C]^{1/2} with C = W_2^2.
        result = math.sqrt(result)

    return float(result)