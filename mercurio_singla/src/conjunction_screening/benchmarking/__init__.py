"""Benchmarking: metrics and scaling experiments.

Provides recall/precision comparison against brute force and timing
experiments across catalog sizes.
"""

from .metrics import recall_precision, topk_recall, evaluate_screening
from .experiments import (
    generate_gaussian_catalog,
    scaling_experiment,
    knn_sweep,
)

__all__ = [
    "recall_precision",
    "topk_recall",
    "evaluate_screening",
    "generate_gaussian_catalog",
    "scaling_experiment",
    "knn_sweep",
]
