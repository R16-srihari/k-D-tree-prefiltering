"""Spatial module: custom KD-tree implementation.

Replaces scipy.spatial.KDTree with a custom top-down KD-tree that exposes
node-level AABB bounds for pruning decisions and instrumentation
(nodes visited/pruned/leaves visited) during Alg. 3 traversal.
"""

from __future__ import annotations

from .kd_node import KDNode
from .kd_tree import build_kdtree, build_leaf_map

__all__ = ["KDNode", "build_kdtree", "build_leaf_map"]
