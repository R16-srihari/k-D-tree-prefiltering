"""Balanced K-d tree built on mean positions.

The tree contains no probabilistic-distance logic: it stores mean positions,
bounding boxes, and supports Euclidean KNN search plus box-distance queries.
All probabilistic ranking is handled by the screening layer through the
ProbabilisticDistance interface.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from heapq import heappush, heappop
import numpy as np

from .kd_node import KDTreeNode
from .geometry import dist_point_to_box, dist_point_to_point


class KDTree:
    """A balanced K-d tree built on mean positions using longest-dimension
    median splits.

    Attributes
    ----------
    root : KDTreeNode | None
        Root node of the tree.
    dim : int
        Dimensionality of the points.
    """

    def __init__(self, root: Optional[KDTreeNode], dim: int) -> None:
        self.root = root
        self.dim = dim

    @classmethod
    def build(cls, points: np.ndarray,
              indices: Optional[np.ndarray] = None) -> "KDTree":
        """Build a balanced K-d tree from points.

        Parameters
        ----------
        points : np.ndarray, shape (n, d)
            Points (mean positions) to insert into the tree.
        indices : np.ndarray, shape (n,), optional
            Original indices of the points. Defaults to 0..n-1.

        Returns
        -------
        KDTree
            The constructed tree.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be a 2D array of shape (n, d).")
        n, dim = points.shape
        if indices is None:
            indices = np.arange(n)
        else:
            indices = np.asarray(indices)

        root = _build_recursive(points, indices)
        return cls(root=root, dim=dim)

    @classmethod
    def from_rsos(cls, rsos) -> "KDTree":
        """Build a K-d tree from a sequence of RSO objects.

        Uses each RSO's overall mean position (for a GMM this is the
        mixture weighted mean).

        Parameters
        ----------
        rsos : sequence of RSO

        Returns
        -------
        KDTree
        """
        points = np.stack([r.overall_mean for r in rsos], axis=0)
        indices = np.arange(len(rsos))
        return cls.build(points, indices)

    def knn_euclidean(self, point: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Classic Euclidean K-nearest-neighbour search.

        Parameters
        ----------
        point : np.ndarray, shape (d,)
            Query point.
        k : int
            Number of neighbours to return.

        Returns
        -------
        list of (index, distance)
            Neighbour indices and Euclidean distances, sorted by distance.
        """
        point = np.asarray(point, dtype=float)
        k = max(1, min(k, self._size()))
        best: List[Tuple[float, int]] = []  # max-heap of (-distance, index)

        def _search(node: Optional[KDTreeNode]) -> None:
            if node is None:
                return
            dist = dist_point_to_point(point, node.point)
            if len(best) < k:
                heappush(best, (-dist, node.index))
            elif dist < -best[0][0]:
                heappushpop(best, (-dist, node.index))

            diff = point[node.split_dim] - node.split_val
            first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)
            _search(first)
            if len(best) < k or abs(diff) < -best[0][0]:
                _search(second)

        _search(self.root)
        result = [(-neg_dist, idx) for neg_dist, idx in best]
        result.sort(key=lambda x: x[0])
        return result

    def _size(self) -> int:
        """Count nodes in the tree."""
        count = 0

        def _count(node: Optional[KDTreeNode]) -> None:
            nonlocal count
            if node is None:
                return
            count += 1
            _count(node.left)
            _count(node.right)

        _count(self.root)
        return count


def _compute_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box min/max for a set of points."""
    return np.min(points, axis=0), np.max(points, axis=0)


def _build_recursive(points: np.ndarray, indices: np.ndarray) -> Optional[KDTreeNode]:
    """Recursively build the K-d tree using longest-dimension median split."""
    n = points.shape[0]
    if n == 0:
        return None

    bbox_min, bbox_max = _compute_bbox(points)
    spreads = bbox_max - bbox_min
    split_dim = int(np.argmax(spreads))

    sorted_idx = np.argsort(points[:, split_dim], kind="mergesort")
    mid = n // 2
    split_idx = sorted_idx[mid]

    node = KDTreeNode(
        point=points[split_idx],
        index=int(indices[split_idx]),
        split_dim=split_dim,
        split_val=float(points[split_idx, split_dim]),
        bbox_min=bbox_min.copy(),
        bbox_max=bbox_max.copy(),
    )

    left_points = points[sorted_idx[:mid]]
    left_indices = indices[sorted_idx[:mid]]
    right_points = points[sorted_idx[mid + 1:]]
    right_indices = indices[sorted_idx[mid + 1:]]

    node.left = _build_recursive(left_points, left_indices)
    node.right = _build_recursive(right_points, right_indices)

    return node


__all__ = ["KDTree", "dist_point_to_box", "dist_point_to_point"]
