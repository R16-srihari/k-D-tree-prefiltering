"""KNN result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class NeighborResult:
    """A single neighbor result from a KNN query."""

    object_id: str
    probabilistic_distance: float
    nominal_euclidean_distance: float

    def __lt__(self, other: NeighborResult) -> bool:
        if self.probabilistic_distance != other.probabilistic_distance:
            return self.probabilistic_distance < other.probabilistic_distance
        return self.object_id < other.object_id


@dataclass
class KNNResult:
    """Complete KNN result for a single query RSO."""

    query_id: str
    neighbors: list[NeighborResult] = field(default_factory=list)
    metric: str = ""
    component_count: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        # Sort by probabilistic distance, then object_id for deterministic tie-breaking
        self.neighbors.sort()

    @property
    def n_neighbors(self) -> int:
        return len(self.neighbors)

    def to_csv_rows(self) -> list[dict]:
        """Convert to list of CSV row dicts."""
        rows = []
        for rank, neighbor in enumerate(self.neighbors, start=1):
            rows.append({
                "query_id": self.query_id,
                "neighbor_rank": rank,
                "neighbor_id": neighbor.object_id,
                "probabilistic_distance": neighbor.probabilistic_distance,
                "nominal_euclidean_distance": neighbor.nominal_euclidean_distance,
                "metric": self.metric,
                "component_count": self.component_count,
                "seed": self.seed,
            })
        return rows


@dataclass
class KNNSearchStats:
    """Instrumentation statistics for a KNN search."""

    distance_evaluations: int = 0
    nodes_visited: int = 0
    nodes_pruned: int = 0
    leaves_visited: int = 0
    query_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "distance_evaluations": self.distance_evaluations,
            "nodes_visited": self.nodes_visited,
            "nodes_pruned": self.nodes_pruned,
            "leaves_visited": self.leaves_visited,
            "query_time_seconds": self.query_time_seconds,
        }
