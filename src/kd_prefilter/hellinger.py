"""
Gaussian closed-form Hellinger distance.

For two Gaussians p = N(mu1, S1) and q = N(mu2, S2):

BC(p,q) = |S1|^{1/4} |S2|^{1/4} / |Sbar|^{1/2} * exp(-1/8 * d^T Sbar^{-1} d)
H^2(p,q) = 1 - BC(p,q)
H = sqrt(H^2) in [0, 1]

where Sbar = (S1 + S2) / 2, d = mu2 - mu1
"""

import numpy as np
from scipy.linalg import slogdet, inv
from typing import Tuple, Optional


def _hellinger_gaussian_core(mu1: np.ndarray, S1: np.ndarray,
                             mu2: np.ndarray, S2: np.ndarray) -> float:
    """
    Core computation of Hellinger distance between two Gaussians.
    
    Returns H in [0, 1]. Identical distributions give 0; disjoint give →1.
    """
    d = mu2 - mu1
    Sbar = 0.5 * (S1 + S2)
    
    # Use slogdet for numerical stability
    sign1, logdet1 = slogdet(S1)
    sign2, logdet2 = slogdet(S2)
    sign_bar, logdet_bar = slogdet(Sbar)
    
    # Check for positive definiteness
    if sign1 <= 0 or sign2 <= 0 or sign_bar <= 0:
        # Non-positive definite covariance - return max distance
        return 1.0
    
    # log(BC) = 0.25*(logdet1 + logdet2) - 0.5*logdet_bar - 0.125 * d^T Sbar^{-1} d
    log_bc = 0.25 * (logdet1 + logdet2) - 0.5 * logdet_bar
    
    # Mahalanobis term
    try:
        Sbar_inv = inv(Sbar)
        mahal = d @ Sbar_inv @ d
        log_bc -= 0.125 * mahal
    except np.linalg.LinAlgError:
        return 1.0
    
    # Clamp log_bc to avoid numerical issues
    log_bc = np.clip(log_bc, -700, 0)  # exp(-700) ~ 5e-305
    
    bc = np.exp(log_bc)
    bc = np.clip(bc, 0.0, 1.0)
    
    h2 = 1.0 - bc
    h2 = np.clip(h2, 0.0, 1.0)
    
    return float(np.sqrt(h2))


def hellinger_gaussian(mu1: np.ndarray, S1: np.ndarray,
                       mu2: np.ndarray, S2: np.ndarray) -> float:
    """
    Hellinger distance between two 3D Gaussians.
    
    Parameters
    ----------
    mu1, mu2 : array-like, shape (3,)
        Mean vectors
    S1, S2 : array-like, shape (3, 3)
        Covariance matrices (must be symmetric positive definite)
    
    Returns
    -------
    float
        Hellinger distance in [0, 1]
    """
    mu1 = np.asarray(mu1, dtype=float).ravel()
    mu2 = np.asarray(mu2, dtype=float).ravel()
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    
    return _hellinger_gaussian_core(mu1, S1, mu2, S2)


def hellinger_gaussian_vectorized(mu1: np.ndarray, S1: np.ndarray,
                                  mu2: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Vectorized Hellinger distance for aligned arrays of Gaussians.
    
    Parameters
    ----------
    mu1, mu2 : array-like, shape (n, 3)
        Mean vectors for n pairs
    S1, S2 : array-like, shape (n, 3, 3)
        Covariance matrices for n pairs
    
    Returns
    -------
    ndarray, shape (n,)
        Hellinger distances for each pair
    """
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    
    n = mu1.shape[0]
    results = np.zeros(n)
    
    for i in range(n):
        results[i] = _hellinger_gaussian_core(mu1[i], S1[i], mu2[i], S2[i])
    
    return results


def hellinger_gaussian_pairwise(mus: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """
    Full pairwise Hellinger distance matrix for a set of Gaussians.
    
    Parameters
    ----------
    mus : array-like, shape (n, 3)
        Mean vectors for n objects
    covs : array-like, shape (n, 3, 3)
        Covariance matrices for n objects
    
    Returns
    -------
    ndarray, shape (n, n)
        Pairwise Hellinger distance matrix (symmetric, zero diagonal)
    """
    mus = np.asarray(mus, dtype=float)
    covs = np.asarray(covs, dtype=float)
    n = mus.shape[0]
    
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            h = _hellinger_gaussian_core(mus[i], covs[i], mus[j], covs[j])
            D[i, j] = h
            D[j, i] = h
    
    return D


def hellinger_gaussian_topk(mu_query: np.ndarray, S_query: np.ndarray,
                            mus: np.ndarray, covs: np.ndarray,
                            k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k nearest neighbours by Hellinger distance.
    
    Parameters
    ----------
    mu_query : array-like, shape (3,)
        Query mean vector
    S_query : array-like, shape (3, 3)
        Query covariance matrix
    mus : array-like, shape (n, 3)
        Candidate mean vectors
    covs : array-like, shape (n, 3, 3)
        Candidate covariance matrices
    k : int
        Number of nearest neighbours to return
    
    Returns
    -------
    indices : ndarray, shape (k,)
        Indices of k nearest neighbours (sorted by distance)
    distances : ndarray, shape (k,)
        Corresponding Hellinger distances
    """
    mu_query = np.asarray(mu_query, dtype=float).ravel()
    S_query = np.asarray(S_query, dtype=float)
    mus = np.asarray(mus, dtype=float)
    covs = np.asarray(covs, dtype=float)
    
    n = mus.shape[0]
    k = min(k, n)
    
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = _hellinger_gaussian_core(mu_query, S_query, mus[i], covs[i])
    
    # Get top-k indices
    idx = np.argpartition(distances, k)[:k]
    idx = idx[np.argsort(distances[idx])]
    
    return idx, distances[idx]
