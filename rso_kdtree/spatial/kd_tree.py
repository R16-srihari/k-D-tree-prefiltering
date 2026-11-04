"""Custom KD-tree construction with median split on longest spread.

Builds a top-down KD-tree on nominal 3D RSO positions. Each internal node
splits along the dimension with the largest AABB spread at the median
coordinate. Leaf nodes store up to ``leaf_size`` RSOs.
"""

from __future__ import annotations

import numpy as np

from .kd_node import KDNode
from ..models.rso import RSO


def build_kdtree(
    positions: np.ndarray,
    rsos: list[RSO],
    leaf_size: int = 1,
) -> KDNode:
    """Build a custom KD-tree on nominal RSO positions.

    Uses median split along the dimension with the largest spread.
    Every RSO appears in exactly one leaf.

    Parameters
    ----------
    positions : np.ndarray, shape (n, 3)
        Nominal 3D positions.
    rsos : list[RSO]
        RSO objects corresponding to positions (same order).
    leaf_size : int
        Maximum number of RSOs per leaf node. Default 1.

    Returns
    -------
    KDNode
        Root node of the constructed tree.
    """
    positions = np.asarray(positions, dtype=float)
    n = len(positions)

    if n == 0:
        raise ValueError("Cannot build KD-tree with zero positions")

    # Compute root AABB from min/max of all positions
    aabb_lower = positions.min(axis=0)
    aabb_upper = positions.max(axis=0)

    def _build(
        idxs: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        parent: KDNode | None,
    ) -> KDNode:
        node_rsos = [rsos[i] for i in idxs]

        # Leaf condition: few enough RSOs
        if len(idxs) <= leaf_size:
            return KDNode(
                aabb_lower=lower,
                aabb_upper=upper,
                rsos=node_rsos,
                is_leaf=True,
                parent=parent,
            )

        # Choose split axis: dimension with largest spread
        spread = upper - lower
        split_axis = int(np.argmax(spread))

        # Split at median along split axis
        split_values = positions[idxs, split_axis]
        median_val = float(np.median(split_values))

        # Partition indices by median
        left_mask = positions[idxs, split_axis] <= median_val
        right_mask = ~left_mask

        # Handle degenerate case: all values equal on split axis
        if not np.any(left_mask) or not np.any(right_mask):
            mid = len(idxs) // 2
            left_idxs = idxs[:mid]
            right_idxs = idxs[mid:]
        else:
            left_idxs = idxs[left_mask]
            right_idxs = idxs[right_mask]

        # Compute child AABBs
        left_lower = lower.copy()
        left_upper = upper.copy()
        left_upper[split_axis] = median_val

        right_lower = lower.copy()
        right_upper = upper.copy()
        right_lower[split_axis] = median_val

        # Recurse on children
        left_child = _build(left_idxs, left_lower, left_upper, parent=None)
        right_child = _build(right_idxs, right_lower, right_upper, parent=None)

        # Build internal node and wire parent references
        node = KDNode(
            aabb_lower=lower,
            aabb_upper=upper,
            split_axis=split_axis,
            split_value=median_val,
            left=left_child,
            right=right_child,
            is_leaf=False,
            parent=parent,
        )
        left_child.parent = node
        right_child.parent = node

        return node

    return _build(np.arange(n), aabb_lower, aabb_upper, parent=None)
