"""
Prefilter orchestration module.

Provides the main prefilter(...) function that:
- Dispatches to appropriate Hellinger distance based on object type (Gaussian/GMM)
- Caches pair results to avoid duplicate computations
- Supports multiple backends: custom KD-tree, hybrid (cKDTree), brute force
- Assembles unique pairs (i < j) from neighbour lists
- Collects statistics (DH evals, nodes opened/pruned, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy as np
import itertools

from .hellinger import hellinger_gaussian, hellinger_gaussian_pairwise
from .gmm_hellinger import GMM, hellinger_gmm
from .kdtree_custom import build_kdtree, modified_nn_search
from .kdtree_scipy_adapter import knn_euclidean_scipy, modified_nn_search_hybrid
from .kdtree_scipy_adapter import prefilter_brute_force_euclidean


@dataclass
class ProbabilisticDistance:
    """Container for probabilistic distance computation results."""
    hellinger_distance: float
    euclidean_distance: float
    eval_count: int  # Number of Hellinger evaluations for this pair
    
    def __post_init__(self):
        self.hellinger_distance = float(self.hellinger_distance)
        self.euclidean_distance = float(self.euclidean_distance)
        self.eval_count = int(self.eval_count)


def _build_object_database(objects: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray], GMM]]) -> Tuple[
    np.ndarray,  # mu_database: shape (n, 3)
    np.ndarray,  # Sigma_database: shape (n, 3, 3)
    List[bool],  # is_gaussian: True if object is Gaussian, False if GMM
]:
    """
    Build database arrays from object list.
    
    Objects can be:
    - np.ndarray: mean vector (3,) - treated as zero covariance
    - Tuple[np.ndarray, np.ndarray]: (mean, cov) for Gaussian
    - GMM: Gaussian mixture model
    
    Returns:
    - mu_database: mean positions
    - Sigma_database: covariances (zero for mean-only, actual for Gaussian/GMM)
    - is_gaussian: flags for dispatcher
    """
    n = len(objects)
    mu_database = np.zeros((n, 3))
    Sigma_database = np.zeros((n, 3, 3))
    is_gaussian = []
    
    for i, obj in enumerate(objects):
        if isinstance(obj, GMM):
            mu_database[i] = obj.weighted_mean
            # For GMM, use covariance of weighted mean approximation
            # This is not perfect but provides a single Gaussian approximation
            # For true GMM-GMM distance, we use the GMM hellinger function
            Sigma_database[i] = np.eye(3) * 1e-6  # Small value to avoid zeros
            is_gaussian.append(False)
        elif isinstance(obj, tuple) and len(obj) == 2:
            mu, Sigma = obj
            mu_database[i] = np.asarray(mu)
            Sigma_database[i] = np.asarray(Sigma)
            is_gaussian.append(True)
        elif isinstance(obj, np.ndarray):
            mu_database[i] = np.asarray(obj).ravel()
            # Zero covariance for mean-only objects
            Sigma_database[i] = np.zeros((3, 3))
            is_gaussian.append(True)  # Treat as Gaussian with zero cov
        else:
            raise ValueError(f"Unsupported object type: {type(obj)}")
    
    return mu_database, Sigma_database, is_gaussian


def _hellinger_dispatcher(mu1: np.ndarray, S1: np.ndarray,
                          mu2: np.ndarray, S2: np.ndarray,
                          is_g1: bool, is_g2: bool,
                          obj1: any, obj2: any) -> float:
    """
    Dispatch to appropriate Hellinger distance function.
    
    Parameters
    ----------
    mu1, S1, mu2, S2 : mean and covariance arrays
    is_g1, is_g2 : bool flags indicating if object is Gaussian (True) or GMM (False)
    obj1, obj2 : original objects (for GMM case)
    
    Returns
    -------
    float : Hellinger distance
    """
    # Both Gaussian - use closed form
    if is_g1 and is_g2:
        return hellinger_gaussian(mu1, S1, mu2, S2)
    
    # At least one is GMM - use GMM hellinger
    # Convert mean/cov tuples to GMM objects if needed
    if not is_g1:
        # obj1 should be GMM
        gmm1 = obj1 if isinstance(obj1, GMM) else GMM([1.0], mu1.reshape(1, 3), S1.reshape(1, 3, 3))
    else:
        gmm1 = GMM([1.0], mu1.reshape(1, 3), S1.reshape(1, 3, 3))
    
    if not is_g2:
        # obj2 should be GMM
        gmm2 = obj2 if isinstance(obj2, GMM) else GMM([1.0], mu2.reshape(1, 3), S2.reshape(1, 3, 3))
    else:
        gmm2 = GMM([1.0], mu2.reshape(1, 3), S2.reshape(1, 3, 3))
    
    return hellinger_gmm(gmm1, gmm2, method='lp')


def prefilter(objects: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray], GMM]],
              k: int = 10,
              backend: str = 'hybrid',
              cache_pairs: bool = True) -> Tuple[
                  List[Tuple[int, int]],  # unique pairs (i < j)
                  List[float],           # Hellinger distances for pairs
                  dict                   # statistics
              ]:
    """
    Main prefilter function: KD-tree probabilistic conjunction-screening.
    
    Parameters
    ----------
    objects : list
        List of objects, each can be:
        - np.ndarray: mean vector (3,) 
        - Tuple[np.ndarray, np.ndarray]: (mean, covariance) for Gaussian
        - GMM: Gaussian mixture model object
    k : int
        Number of nearest neighbours to consider per object
    backend : str
        'custom': educational KD-tree implementation
        'hybrid': cKDTree for initial KNN + modified search (approximation)
        'brute': brute force all-pairs (for validation)
    cache_pairs : bool
        Whether to cache Hellinger distance computations
    
    Returns
    -------
    pairs : list of tuples
        Unique pairs (i, j) with i < j
    distances : list of float
        Hellinger distance for each pair
    stats : dict
        Statistics including:
        - 'objects_processed': int
        - 'dh_evals_total': int
        - 'de_evals_total': int
        - 'backend': str
        - 'cache_hits': int (if caching enabled)
        - backend-specific stats
    """
    n = len(objects)
    if n == 0:
        return [], [], {}
    
    # Build database
    mu_database, Sigma_database, is_gaussian = _build_object_database(objects)
    
    # Initialize statistics
    stats = {
        'objects_processed': n,
        'dh_evals_total': 0,
        'de_evals_total': 0,
        'backend': backend,
        'cache_hits': 0 if cache_pairs else None,
    }
    
    # Cache for Hellinger distances
    hellinger_cache = {} if cache_pairs else None
    
    def _cached_hellinger(mu1, S1, mu2, S2, idx1, idx2):
        """Compute Hellinger distance with optional caching."""
        if hellinger_cache is not None:
            # Create cache key (order-independent)
            key = tuple(sorted([idx1, idx2]))
            if key in hellinger_cache:
                stats['cache_hits'] = stats.get('cache_hits', 0) + 1
                return hellinger_cache[key]
        
        # Compute distance
        dist = _hellinger_dispatcher(mu1, S1, mu2, S2,
                                   is_gaussian[idx1], is_gaussian[idx2],
                                   objects[idx1], objects[idx2])
        
        if hellinger_cache is not None:
            hellinger_cache[key] = dist
        
        return dist
    
    # Neighbor lists for each object
    neighbors = [[] for _ in range(n)]  # Each element: list of (neighbor_idx, hellinger_dist)
    
    if backend == 'brute':
        # Brute force: compute all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dist = _cached_hellinger(
                    mu_database[i], Sigma_database[i],
                    mu_database[j], Sigma_database[j],
                    i, j
                )
                neighbors[i].append((j, dist))
                neighbors[j].append((i, dist))
                stats['dh_evals_total'] += 1
        
        stats['de_evals_total'] = n * (n - 1) // 2  # All pairs
    
    elif backend in ['custom', 'hybrid']:
        # Build search structure
        if backend == 'custom':
            # Custom KD-tree
            indices = np.arange(n)
            tree = build_kdtree(mu_database, indices)
            search_fn = modified_nn_search
        else:  # hybrid
            # cKDTree
            tree = cKDTree(mu_database)
            search_fn = modified_nn_search_hybrid
        
        # Process each object
        for i in range(n):
            # Get k nearest neighbours by probabilistic distance
            if backend == 'custom':
                nn_indices, nn_distances, node_stats = search_fn(
                    tree, mu_database[i], Sigma_database[i],
                    mu_database, Sigma_database,
                    lambda mu1, S1, mu2, S2: _cached_hellinger(mu1, S1, mu2, S2, i, j_idx),
                    k
                )
                # Accumulate node stats
                for key in ['nodes_visited', 'nodes_opened', 'nodes_pruned']:
                    stats[key] = stats.get(key, 0) + node_stats.get(key, 0)
            else:  # hybrid
                nn_indices, nn_distances, node_stats = search_fn(
                    tree, mu_database[i], Sigma_database[i],
                    mu_database, Sigma_database,
                    lambda mu1, S1, mu2, S2: _cached_hellinger(mu1, S1, mu2, S2, i, j_idx),
                    k
                )
                # Accumulate node stats
                for key in ['nodes_visited', 'nodes_opened', 'nodes_pruned']:
                    stats[key] = stats.get(key, 0) + node_stats.get(key, 0)
            
            # Store neighbours
            neighbors[i] = list(zip(nn_indices, nn_distances))
            
            # Update eval counts from node stats
            stats['dh_evals_total'] += node_stats.get('dh_evals', 0)
            stats['de_evals_total'] += node_stats.get('de_evals', 0)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'custom', 'hybrid', or 'brute'")
    
    # Assemble unique pairs (i < j) from neighbour lists
    pairs_set = set()
    pair_distances = []
    
    for i in range(n):
        for j_idx, dist in neighbors[i]:
            if i < j_idx:
                pairs_set.add((i, j_idx))
                pair_distances.append(dist)
            elif j_idx < i:
                pairs_set.add((j_idx, i))
                pair_distances.append(dist)
    
    # Convert to sorted lists for deterministic output
    pairs = sorted(list(pairs_set))
    # Reorder distances to match pairs
    pair_dict = {(i, j): dist for (i, j), dist in zip(
        [(i, j_idx) for i in range(n) for j_idx, _ in neighbors[i]],
        [dist for i in range(n) for _, dist in neighbors[i]]
    )}
    distances = [pair_dict[pair] for pair in pairs]
    
    return pairs, distances, stats


def prefilter_brute_force(objects: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray], GMM]]) -> Tuple[
    List[Tuple[int, int]], List[float], dict]:
    """
    Brute force prefilter for validation - computes all pairs.
    
    Equivalent to prefilter(..., backend='brute') but with cleaner interface.
    """
    return prefilter(objects, backend='brute', k=len(objects))  # k=n to get all