"""
Educational KD-tree implementation with Algorithm 3 modified nearest-neighbour search.

The KD-tree is built using mean positions only. The modified NN search:
1. Gets initial KNN neighbours using Euclidean distance
2. Ranks them by Hellinger (probabilistic) distance
3. Explores remaining nodes using best-bin-first with box-pruning test:
   Open node if Dbox < DE_min, where DE_min is min Euclidean distance 
   among current DH-top-KNN set (inf if fewer than KNN found so far)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Callable
import numpy as np
from heapq import heappush, heappop


@dataclass
class KDTreeNode:
    """KD-tree node."""
    point: np.ndarray          # Data point at this node (mean position)
    index: int                 # Original index of this point
    split_dim: int             # Dimension used for splitting
    split_val: float           # Value used for splitting
    left: Optional['KDTreeNode'] = None   # Left subtree
    right: Optional['KDTreeNode'] = None  # Right subtree
    bbox_min: Optional[np.ndarray] = None # Bounding box min coords
    bbox_max: Optional[np.ndarray] = None # Bounding box max coords


def _compute_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box for a set of points."""
    return np.min(points, axis=0), np.max(points, axis=0)


def build_kdtree(points: np.ndarray, indices: Optional[np.ndarray] = None,
                 depth: int = 0) -> Optional[KDTreeNode]:
    """
    Build KD-tree from points using longest-dimension median split.
    
    Parameters
    ----------
    points : ndarray, shape (n, 3)
        Points to build tree from (mean positions)
    indices : ndarray, shape (n,), optional
        Original indices of points. If None, uses 0..n-1
    depth : int
        Current depth in tree (for recursion)
    
    Returns
    -------
    KDTreeNode or None
        Root of the KD-tree
    """
    n = points.shape[0]
    if n == 0:
        return None
    
    if indices is None:
        indices = np.arange(n)
    
    # Choose split dimension: longest spread
    bbox_min, bbox_max = _compute_bbox(points)
    spreads = bbox_max - bbox_min
    split_dim = int(np.argmax(spreads))
    
    # Sort by split dimension and take median
    sorted_idx = np.argsort(points[:, split_dim])
    mid = n // 2
    split_idx = sorted_idx[mid]
    
    # Create node
    node = KDTreeNode(
        point=points[split_idx],
        index=indices[split_idx],
        split_dim=split_dim,
        split_val=float(points[split_idx, split_dim]),
        bbox_min=bbox_min.copy(),
        bbox_max=bbox_max.copy()
    )
    
    # Recursively build subtrees
    left_points = points[sorted_idx[:mid]]
    left_indices = indices[sorted_idx[:mid]]
    right_points = points[sorted_idx[mid+1:]]
    right_indices = indices[sorted_idx[mid+1:]]
    
    node.left = build_kdtree(left_points, left_indices, depth + 1)
    node.right = build_kdtree(right_points, right_indices, depth + 1)
    
    return node


def dist_point_to_box(point: np.ndarray, bbox_min: np.ndarray,
                      bbox_max: np.ndarray) -> float:
    """
    Euclidean distance from point to bounding box.
    
    Returns 0 if point is inside box, otherwise distance to nearest point on box.
    """
    point = np.asarray(point)
    bbox_min = np.asarray(bbox_min)
    bbox_max = np.asarray(bbox_max)
    
    # Clamp point to box
    clamped = np.maximum(bbox_min, np.minimum(point, bbox_max))
    
    # Distance is zero if inside, otherwise Euclidean distance to clamped point
    diff = point - clamped
    return float(np.sqrt(np.dot(diff, diff)))


def knn_euclidean(tree: KDTreeNode, point: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """
    Classic Euclidean KNN search in KD-tree.
    
    Returns list of (index, distance) tuples sorted by distance.
    """
    # Use heap for K nearest (max-heap via negative distances)
    best = []  # Will store (-distance, index) for max-heap behavior
    
    def _search(node: Optional[KDTreeNode]):
        if node is None:
            return
        
        # Distance to current node
        dist = np.linalg.norm(point - node.point)
        
        # Add to best if we have room or it's better than worst
        if len(best) < k:
            heappush(best, (-dist, node.index))
        elif dist < -best[0][0]:  # Better than current worst
            heappushpop(best, (-dist, node.index))
        
        # Choose which side to search first
        diff = point[node.split_dim] - node.split_val
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)
        
        # Search closer side first
        _search(first)
        
        # Only search farther side if needed
        if len(best) < k or abs(diff) < -best[0][0]:
            _search(second)
    
    _search(tree)
    
    # Convert to (index, distance) and sort by distance
    result = [(-neg_dist, idx) for neg_dist, idx in best]
    result.sort(key=lambda x: x[1])  # Sort by distance
    
    return result


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


def modified_nn_search(tree: KDTreeNode, mu_query: np.ndarray, 
                       Sigma_query: np.ndarray,
                       mu_database: np.ndarray, Sigma_database: np.ndarray,
                       hellinger_fn: Callable,
                       k: int) -> Tuple[List[int], List[float], dict]:
    """
    Modified nearest-neighbour search (Algorithm 3 from paper).
    
    Parameters
    ----------
    tree : KDTreeNode
        Root of KD-tree built on mean positions
    mu_query : array-like, shape (3,)
        Mean position of query object
    Sigma_query : array-like, shape (3, 3)
        Covariance of query object
    mu_database : ndarray, shape (n, 3)
        Mean positions of all objects
    Sigma_database : ndarray, shape (n, 3, 3)
        Covariances of all objects
    hellinger_fn : callable
        Function computing Hellinger distance: hellinger_fn(mu1, S1, mu2, S2) -> float
    k : int
        Number of nearest neighbours to find
    
    Returns
    -------
    indices : list of int
        Indices of k nearest neighbours by Hellinger distance
    distances : list of float
        Corresponding Hellinger distances
    stats : dict
        Statistics: {'nodes_visited': int, 'nodes_opened': int, 'nodes_pruned': int, 
                     'dh_evals': int, 'de_evals': int}
    """
    mu_query = np.asarray(mu_query, dtype=float)
    Sigma_query = np.asarray(Sigma_query, dtype=float)
    n = mu_database.shape[0]
    
    # Statistics
    stats = {
        'nodes_visited': 0,
        'nodes_opened': 0,
        'nodes_pruned': 0,
        'dh_evals': 0,
        'de_evals': 0,
    }
    
    # Step 1: Get initial KNN using Euclidean distance
    knn_result = knn_euclidean(tree, mu_query, k)
    
    # Convert to list of indices and distances
    knn_indices = [idx for idx, _ in knn_result]
    knn_de_dists = [dist for _, dist in knn_result]
    
    stats['de_evals'] += len(knn_indices)
    
    # Compute Hellinger distances for initial neighbours
    knn_dh_dists = []
    for idx in knn_indices:
        dh = hellinger_fn(mu_query, Sigma_query, mu_database[idx], Sigma_database[idx])
        knn_dh_dists.append(dh)
        stats['dh_evals'] += 1
    
    # Current best K neighbours by Hellinger distance (sorted ascending)
    best_by_dh = list(zip(knn_indices, knn_dh_dists))
    best_by_dh.sort(key=lambda x: x[1])  # Sort by Hellinger distance
    
    # Current DE_min: min Euclidean distance among current best DH set
    # If fewer than k found, DE_min = inf (nothing gets pruned)
    if len(best_by_dh) >= k:
        # Get Euclidean distances of current best DH set
        de_of_best = []
        for idx, _ in best_by_dh:
            de = np.linalg.norm(mu_query - mu_database[idx])
            de_of_best.append(de)
            stats['de_evals'] += 1
        DE_min = min(de_of_best)
    else:
        DE_min = float('inf')
    
    # Best-bin-first search using priority queue
    # Priority is Dbox (distance from query mean to node bbox)
    pq = []  # Min-heap of (Dbox, node)
    heappush(pq, (0.0, tree))  # Start with root
    
    while pq:
        Dbox, node = heappop(pq)
        stats['nodes_visited'] += 1
        
        # Pruning test: if Dbox >= DE_min, skip this node
        if Dbox >= DE_min:
            stats['nodes_pruned'] += 1
            continue
        
        # Otherwise, open the node
        stats['nodes_opened'] += 1
        
        # Check if this node contains a data point
        if node.index is not None:
            # Evaluate this object if not already in our best set
            idx = node.index
            if idx not in [i for i, _ in best_by_dh]:
                # Compute distances
                de = np.linalg.norm(mu_query - mu_database[idx])
                stats['de_evals'] += 1
                
                dh = hellinger_fn(mu_query, Sigma_query, mu_database[idx], Sigma_database[idx])
                stats['dh_evals'] += 1
                
                # Update best set
                best_by_dh = _update_best_probabilistic(best_by_dh, idx, dh, k)
                
                # Update DE_min if we have k elements
                if len(best_by_dh) >= k:
                    de_of_best = []
                    for i, _ in best_by_dh:
                        de_val = np.linalg.norm(mu_query - mu_database[i])
                        de_of_best.append(de_val)
                        stats['de_evals'] += 1
                    DE_min = min(de_of_best)
        
        # Add children to priority queue
        for child in [node.left, node.right]:
            if child is not None:
                # Compute distance from query mean to child's bounding box
                Dbox_child = dist_point_to_box(mu_query, child.bbox_min, child.bbox_max)
                heappush(pq, (Dbox_child, child))
    
    # Extract final results
    indices = [idx for idx, _ in best_by_dh]
    distances = [dist for _, dist in best_by_dh]
    
    return indices, distances, stats


def _collect_all_points(node: Optional[KDTreeNode], 
                       points: List[np.ndarray],
                       indices: List[int]):
    """Helper to collect all points from tree (for debugging)."""
    if node is None:
        return
    points.append(node.point)
    indices.append(node.index)
    _collect_all_points(node.left, points, indices)
    _collect_all_points(node.right, points, indices)


def collect_all_points(tree: KDTreeNode) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all points and indices from tree."""
    points = []
    indices = []
    _collect_all_points(tree, points, indices)
    if points:
        return np.array(points), np.array(indices)
    return np.empty((0, 3)), np.array([], dtype=int)