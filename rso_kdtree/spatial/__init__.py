"""Spatial module backed by scipy.spatial.KDTree.

The tree is built on nominal 3D positions. Node-level AABB instrumentation
(nodes visited/pruned) is not exposed by scipy's KDTree; pruning happens at
the candidate-set level via the D_E,K Euclidean threshold in screening.knn.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def build_kdtree(positions: np.ndarray, leaf_size: int = 10) -> KDTree:
    """Build a scipy KDTree on nominal RSO positions.

    Parameters
    ----------
    positions : np.ndarray, shape (n, 3)
        Nominal 3D positions.
    leaf_size : int
        Points per leaf passed to KDTree.

    Returns
    -------
    scipy.spatial.KDTree
        The constructed tree.
    """
    positions = np.asarray(positions, dtype=float)
    return KDTree(positions, leafsize=leaf_size)


__all__ = ["KDTree", "build_kdtree"]
