"""
Evaluation module for KD-tree prefilter validation.

Provides functions for:
- Recall vs brute force
- KNN sweep
- N-scaling timing experiments
- Nodes pruned fraction
- "Probabilistically-close but mean-far miss" analysis
"""

import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree as SciPyKDTree

from .data_gen import (
    generate_gaussian_catalog,
    generate_gmm_catalog,
    generate_test_case_anisotropic,
    generate_test_case_identical,
)
from .kdtree_custom import build_kdtree, knn_euclidean
from .kdtree_scipy_adapter import cKDTree
from .prefilter import prefilter, prefilter_brute_force


def recall_vs_brute(objects: List,
                    k: int = 10,
                    backend: str = 'custom') -> Dict[str, Any]:
    """
    Compare prefilter candidate pairs against brute force ground truth.
    
    Parameters
    ----------
    objects : list
        List of objects (Gaussians or GMMs)
    k : int
        Number of neighbours per object
    backend : str
        Backend to test ('custom', 'hybrid')
    
    Returns
    -------
    dict with keys:
    - 'pairs_prefilter': set of pairs from prefilter
    - 'pairs_brute': set of pairs from brute force
    - 'recall': float (fraction of brute force pairs found)
    - 'precision': float (fraction of prefilter pairs in brute force)
    - 'stats': dict from prefilter
    """
    # Get prefilter results
    pairs_prefilter, dists_prefilter, stats = prefilter(objects, k=k, backend=backend)
    
    # Get brute force ground truth (all pairs)
    pairs_brute, dists_brute, _ = prefilter_brute_force(objects)
    
    # Convert to sets for comparison
    set_prefilter = set(pairs_prefilter)
    set_brute = set(pairs_brute)
    
    # Recall: what fraction of true pairs did we find?
    # Note: brute force returns ALL pairs, so we compare against top-K by Hellinger
    # Actually, for fair comparison, we should compare the same number of pairs
    # For each object, brute force gives all distances - we take top-k
    
    n = len(objects)
    true_topk_pairs = set()
    
    # Get true top-k for each object
    for i in range(n):
        # Compute all Hellinger distances from i
        all_dists = []
        for j in range(n):
            if i != j:
                all_dists.append((j, dists_brute[pairs_brute.index((min(i,j), max(i,j)))]))
        # Sort and take top-k
        all_dists.sort(key=lambda x: x[1])
        for j, _ in all_dists[:k]:
            true_topk_pairs.add((min(i, j), max(i, j)))
    
    recall = len(set_prefilter & true_topk_pairs) / len(true_topk_pairs) if true_topk_pairs else 1.0
    precision = len(set_prefilter & true_topk_pairs) / len(set_prefilter) if set_prefilter else 0.0
    
    return {
        'pairs_prefilter': set_prefilter,
        'pairs_brute': true_topk_pairs,
        'recall': recall,
        'precision': precision,
        'stats': stats
    }


def knn_sweep(objects: List,
              k_values: List[int] = None,
              backend: str = 'custom') -> Dict[int, Dict]:
    """
    Sweep over KNN values and measure recall/timing.
    
    Parameters
    ----------
    objects : list
        List of objects
    k_values : list of int
        KNN values to test
    backend : str
        Backend to test
    
    Returns
    -------
    dict mapping k -> {recall, precision, time, stats}
    """
    if k_values is None:
        k_values = [5, 10, 20, 30, 50]
    
    results = {}
    for k in k_values:
        start = time.perf_counter()
        result = recall_vs_brute(objects, k=k, backend=backend)
        elapsed = time.perf_counter() - start
        
        results[k] = {
            'recall': result['recall'],
            'precision': result['precision'],
            'time': elapsed,
            'stats': result['stats']
        }
    
    return results


def scaling_experiment(n_values: List[int] = None,
                       k: int = 10,
                       backend: str = 'custom',
                       seed: int = 42) -> Dict[int, Dict]:
    """
    Run timing experiment across different catalog sizes.
    
    Parameters
    ----------
    n_values : list of int
        Catalog sizes to test
    k : int
        KNN value
    backend : str
        Backend to test
    seed : int
        Random seed
    
    Returns
    -------
    dict mapping n -> {time, dh_evals, de_evals, stats}
    """
    if n_values is None:
        n_values = [100, 200, 500, 1000, 2000, 5000]
    
    results = {}
    for n in n_values:
        # Generate catalog
        objects = generate_gaussian_catalog(n, seed=seed)
        
        start = time.perf_counter()
        pairs, dists, stats = prefilter(objects, k=k, backend=backend)
        elapsed = time.perf_counter() - start
        
        results[n] = {
            'time': elapsed,
            'dh_evals': stats.get('dh_evals_total', 0),
            'de_evals': stats.get('de_evals_total', 0),
            'stats': stats
        }
    
    return results


def miss_analysis(objects: List,
                  k: int = 10,
                  backend: str = 'custom') -> Dict[str, Any]:
    """
    Analyze "probabilistically-close but mean-far" misses.
    
    These are pairs where Hellinger distance is small (high PDF overlap)
    but Euclidean distance is large, causing them to be pruned by the
    deterministic box test.
    
    Parameters
    ----------
    objects : list
        List of objects
    k : int
        KNN value
    backend : str
        Backend to test
    
    Returns
    -------
    dict with:
    - 'missed_pairs': list of (i, j, DH, DE) that were missed
    - 'total_misses': int
    - 'miss_rate': float
    """
    # Get prefilter results
    pairs_prefilter, dists_prefilter, stats = prefilter(objects, k=k, backend=backend)
    set_prefilter = set(pairs_prefilter)
    
    # Get brute force ground truth
    pairs_brute, dists_brute, _ = prefilter_brute_force(objects)
    set_brute = set(pairs_brute)
    
    # Find pairs in brute but not in prefilter
    missed = set_brute - set_prefilter
    
    # Analyze misses
    missed_details = []
    n = len(objects)
    
    # Build object database for distance queries
    from .prefilter import _build_object_database
    mu_db, Sigma_db, is_gaussian = _build_object_database(objects)
    
    for i, j in missed:
        # Compute both distances
        from .prefilter import _hellinger_dispatcher
        dh = _hellinger_dispatcher(mu_db[i], Sigma_db[i], mu_db[j], Sigma_db[j],
                                   is_gaussian[i], is_gaussian[j],
                                   objects[i], objects[j])
        de = np.linalg.norm(mu_db[i] - mu_db[j])
        missed_details.append({
            'pair': (i, j),
            'hellinger': dh,
            'euclidean': de
        })
    
    return {
        'missed_pairs': missed_details,
        'total_misses': len(missed),
        'miss_rate': len(missed) / len(set_brute) if set_brute else 0.0
    }


def tree_balance_stats(tree_root) -> Dict[str, Any]:
    """
    Analyze KD-tree balance statistics.
    """
    def _collect(node, depth, stats):
        if node is None:
            return
        stats['depths'].append(depth)
        stats['total_nodes'] += 1
        if node.left is None and node.right is None:
            stats['leaves'] += 1
        _collect(node.left, depth + 1, stats)
        _collect(node.right, depth + 1, stats)
    
    stats = {'depths': [], 'total_nodes': 0, 'leaves': 0}
    _collect(tree_root, 0, stats)
    
    depths = stats['depths']
    return {
        'total_nodes': stats['total_nodes'],
        'leaves': stats['leaves'],
        'max_depth': max(depths) if depths else 0,
        'min_depth': min(depths) if depths else 0,
        'avg_depth': np.mean(depths) if depths else 0,
        'depth_std': np.std(depths) if depths else 0
    }


def euclidean_knn_validation(mu_database: np.ndarray, k: int = 10) -> Dict[str, Any]:
    """
    Validate custom KD-tree Euclidean KNN against cKDTree.
    
    Returns dict with match_rate, timing comparison.
    """
    n = mu_database.shape[0]
    
    # Custom tree
    indices = np.arange(n)
    custom_tree = build_kdtree(mu_database, indices)
    
    # cKDTree
    scipy_tree = SciPyKDTree(mu_database)
    
    match_count = 0
    total_queries = min(100, n)  # Test subset for speed
    
    for i in range(total_queries):
        point = mu_database[i]
        
        # Custom KNN
        custom_result = knn_euclidean(custom_tree, point, k)
        custom_indices = set(idx for idx, _ in custom_result)
        
        # SciPy KNN
        dists, idxs = scipy_tree.query(point, k=k+1)  # +1 to exclude self
        scipy_indices = set(idxs[1:])  # Exclude self
        
        if custom_indices == scipy_indices:
            match_count += 1
    
    return {
        'match_rate': match_count / total_queries,
        'tested_queries': total_queries
    }
