"""conjunction_screening — Mercurio–Singla K-d tree conjunction screening.

A modular framework for K-d tree-based conjunction screening with pluggable
probabilistic distances. The K-d tree and screening algorithm depend only on
the :class:`conjunction_screening.distances.ProbabilisticDistance` interface,
never on a concrete metric.

Subpackages
-----------
models : RSO, Gaussian, GaussianMixture uncertainty representations
distances : ProbabilisticDistance interface, HellingerDistance, factory
spatial : KDTree, KDTreeNode, geometry utilities
screening : MercurioSinglaScreening, brute_force_screen, results
benchmarking : metrics, scaling experiments
utils : validation utilities
"""

from .models import Gaussian, GaussianMixture, RSO
from .distances import ProbabilisticDistance, HellingerDistance, create_distance
from .spatial import KDTree, KDTreeNode
from .screening import MercurioSinglaScreening, brute_force_screen, ScreeningResult

__all__ = [
    # models
    "Gaussian",
    "GaussianMixture",
    "RSO",
    # distances
    "ProbabilisticDistance",
    "HellingerDistance",
    "create_distance",
    # spatial
    "KDTree",
    "KDTreeNode",
    # screening
    "MercurioSinglaScreening",
    "brute_force_screen",
    "ScreeningResult",
]

__version__ = "0.1.0"
