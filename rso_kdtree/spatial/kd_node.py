"""KD-tree node data model for custom KD-tree construction.

Provides the KDNode class with AABB bounds, split axis/value, parent/child
references, and leaf RSO storage. Used by the custom top-down KD-tree
implementation in kd_tree.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.rso import RSO


class KDNode:
    """A node in the custom KD-tree.

    Attributes
    ----------
    aabb_lower : np.ndarray, shape (3,)
        Lower bound of the axis-aligned bounding box.
    aabb_upper : np.ndarray, shape (3,)
        Upper bound of the axis-aligned bounding box.
    split_axis : int | None
        Dimension used for splitting (internal nodes only).
    split_value : float | None
        Split coordinate along split_axis (internal nodes only).
    parent : KDNode | None
        Parent node, or None for root.
    left : KDNode | None
        Left child, or None for leaf.
    right : KDNode | None
        Right child, or None for leaf.
    rsos : list[RSO]
        RSOs stored at this node (leaf nodes only, up to leaf_size).
    is_leaf : bool
        True if this is a leaf node.
    """

    def __init__(
        self,
        aabb_lower: np.ndarray,
        aabb_upper: np.ndarray,
        split_axis: Optional[int] = None,
        split_value: Optional[float] = None,
        parent: Optional["KDNode"] = None,
        left: Optional["KDNode"] = None,
        right: Optional["KDNode"] = None,
        rsos: Optional[list[RSO]] = None,
        is_leaf: bool = False,
    ) -> None:
        self.aabb_lower = np.asarray(aabb_lower, dtype=float)
        self.aabb_upper = np.asarray(aabb_upper, dtype=float)
        self.split_axis = split_axis
        self.split_value = split_value
        self.parent = parent
        self.left = left
        self.right = right
        self.rsos = rsos if rsos is not None else []
        self.is_leaf = is_leaf

    def __repr__(self) -> str:
        if self.is_leaf:
            return (
                f"KDNode(leaf, rsos={len(self.rsos)}, "
                f"aabb=[{self.aabb_lower}, {self.aabb_upper}])"
            )
        return (
            f"KDNode(axis={self.split_axis}, val={self.split_value:.4f}, "
            f"aabb=[{self.aabb_lower}, {self.aabb_upper}])"
        )
