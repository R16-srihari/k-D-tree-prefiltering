"""Result containers for screening runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass
class PairResult:
    """A single screened pair with its distances."""

    i: int
    j: int
    euclidean_distance: float
    probabilistic_distance: float


@dataclass
class ScreeningResult:
    """Results from a screening run.

    Attributes
    ----------
    pairs : list of (int, int)
        Unique screened pairs with i < j.
    euclidean_distances : list of float
        Euclidean mean-position distance for each pair.
    probabilistic_distances : list of float
        Probabilistic distance for each pair.
    stats : dict
        Run statistics (nodes visited/opened/pruned, distance evaluations,
        execution time, etc.).
    """

    pairs: List[Tuple[int, int]]
    euclidean_distances: List[float]
    probabilistic_distances: List[float]
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pairs = [(int(i), int(j)) for i, j in self.pairs]
        self.euclidean_distances = [float(d) for d in self.euclidean_distances]
        self.probabilistic_distances = [float(d) for d in self.probabilistic_distances]

    def sorted_by_probabilistic(self) -> List[PairResult]:
        """Return pairs sorted by probabilistic distance (ascending)."""
        results = [
            PairResult(i, j, de, dp)
            for (i, j), de, dp in zip(
                self.pairs,
                self.euclidean_distances,
                self.probabilistic_distances,
            )
        ]
        results.sort(key=lambda r: r.probabilistic_distance)
        return results

    def top_k(self, k: int) -> List[PairResult]:
        """Return the k pairs with smallest probabilistic distance."""
        return self.sorted_by_probabilistic()[:k]

    def __len__(self) -> int:
        return len(self.pairs)
