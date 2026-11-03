"""
cKDTree-based initial-KNN provider for hybrid fast path.

Uses scipy.spatial.cKDTree for fast Euclidean KNN queries,
then feeds into the same modified search logic for probabilistic ranking.
"""

import heapq
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def knn_euclidean_scipy(tree: cKDTree, point: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """
    Euclidean KNN using scipy's cKDTree.
    
    Parameters
    ----------
    tree : cKDTree
        Pre-built cKDTree
    point : array-like, shape (3,)
        Query point
    k : int
        Number of neighbours
    
    Returns
    -------
    list of (index, distance) tuples sorted by distance
    """
    distances, indices = tree.query(point, k=k)
    
    # Handle case where k=1 returns scalars
    if k == 1:
        distances = np.array([distances])
        indices = np.array([indices])
    
    return list(zip(indices.tolist(), distances.tolist()))


def _update_best_probabilistic(current_best: List[Tuple[int, float]],
                              candidate_idx: int, candidate_dist: float,
                              k: int) -> List[Tuple[int, float]]:
    """
    Update the best K probabilistic neighbours list.
    
    current_best: list of (index, distance) sorted by distance (ascending)
    Returns updated list
    """
    # Insert maintaining sorted order
    inserted = False
    for i in range(len(current_best)):
        if candidate_dist < current_best[i][1]:
            current_best.insert(i, (candidate_idx, candidate_dist))
            inserted = True
            break
    
    if not inserted and len(current_best) < k:
        current_best.append((candidate_idx, candidate_dist))
    
    # Trim to k elements
    if len(current_best) > k:
        current_best = current_best[:k]
    
    return current_best


def modified_nn_search_hybrid(tree: cKDTree, mu_query: np.ndarray,
                              Sigma_query: np.ndarray,
                              mu_database: np.ndarray, Sigma_database: np.ndarray,
                              hellinger_fn: Callable,
                              k: int,
                              max_extra_nodes: int = 1000) -> Tuple[List[int], List[float], dict]:
    """
    Hybrid modified NN search using cKDTree for initial KNN.
    
    This is a simplified version of Algorithm 3 that:
    1. Uses cKDTree to get initial KNN by Euclidean distance
    2. Ranks by Hellinger distance
    3. Uses cKDTree's ball query to explore nearby regions
    
    Note: This is an approximation - full Algorithm 3 with box pruning
    requires the custom tree structure. This version uses a ball query
    with radius = DE_min to find additional candidates.
    
    Parameters
    ----------
    tree : cKDTree
        Pre-built cKDTree on mean positions
    mu_query : array-like, shape (3,)
        Mean position of query object
    Sigma_query : array-like, shape (3, 3)
        Covariance of query object
    mu_database : ndarray, shape (n, 3)
        Mean positions of all objects
    Sigma_database : ndarray, shape (n, 3, 3)
        Covariances of all objects
    hellinger_fn : callable
        Function computing Hellinger distance
    k : int
        Number of nearest neighbours to find
    max_extra_nodes : int
        Maximum extra candidates to evaluate beyond initial KNN
    
    Returns
    -------
    indices : list of int
        Indices of k nearest neighbours by Hellinger distance
    distances : list of float
        Corresponding Hellinger distances
    stats : dict
        Statistics
    """
    mu_query = np.asarray(mu_query, dtype=float)
    Sigma_query = np.asarray(Sigma_query, dtype=float)
    n = mu_database.shape[0]
    
    stats = {
        'nodes_visited': 0,
        'nodes_opened': 0,
        'nodes_pruned': 0,
        'dh_evals': 0,
        'de_evals': 0,
    }
    
    # Step 1: Get initial KNN using Euclidean distance from cKDTree
    knn_result = knn_euclidean_scipy(tree, mu_query, k)
    
    knn_indices = [idx for idx, _ in knn_result]
    knn_de_dists = [dist for _, dist in knn_result]
    
    stats['de_evals'] += len(knn_indices)
    
    # Compute Hellinger distances for initial neighbours
    knn_dh_dists = []
    for idx in knn_indices:
        dh = hellinger_fn(mu_query, Sigma_query, mu_database[idx], Sigma_database[idx])
        knn_dh_dists.append(dh)
        stats['dh_evals'] += 1
    
    # Current best K neighbours by Hellinger distance
    best_by_dh = list(zip(knn_indices, knn_dh_dists))
    best_by_dh.sort(key=lambda x: x[1])
    
    # Current DE_min
    if len(best_by_dh) >= k:
        de_of_best = []
        for idx, _ in best_by_dh:
            de = np.linalg.norm(mu_query - mu_database[idx])
            de_of_best.append(de)
            stats['de_evals'] += 1
        DE_min = min(de_of_best)
    else:
        DE_min = float('inf')
    
    # Explore additional candidates using ball query
    # Query all points within DE_min radius
    if DE_min < float('inf'):
        extra_indices = tree.query_ball_point(mu_query, DE_min)
        # Remove already evaluated points
        extra_indices = [idx for idx in extra_indices if idx not in knn_indices]
        # Limit extra evaluations
        extra_indices = extra_indices[:max_extra_nodes]
    else:
        extra_indices = []
    
    # Evaluate extra candidates
    for idx in extra_indices:
        if idx not in [i for i, _ in best_by_dh]:
            de = np.linalg.norm(mu_query - mu_database[idx])
            stats['de_evals'] += 1
            
            dh = hellinger_fn(mu_query, Sigma_query, mu_database[idx], Sigma_database[idx])
            stats['dh_evals'] += 1
            
            best_by_dh = _update_best_probabilistic(best_by_dh, idx, dh, k)
            
            # Update DE_min
            if len(best_by_dh) >= k:
                de_of_best = []
                for i, _ in best_by_dh:
                    de_val = np.linalg.norm(mu_query - mu_database[i])
                    de_of_best.append(de_val)
                    stats['de_evals'] += 1
                DE_min = min(de_of_best)
    
    # Extract final results
    indices = [idx for idx, _ in best_by_dh]
    distances = [dist for _, dist in best_by_dh]
    
    return indices, distances, stats


def prefilter_brute_force_euclidean(mu_database: np.ndarray, k: int) -> np.ndarray:
    """
    Brute-force Euclidean KNN for all points (baseline).
    
    Returns array of shape (n, k) with neighbour indices for each point.
    """
    n = mu_database.shape[0]
    result = np.zeros((n, k), dtype=int)
    
    for i in range(n):
        dists = np.linalg.norm(mu_database - mu_database[i], axis=1)
        # Exclude self
        dists[i] = float('inf')
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        result[i] = idx
    
    return result
