"""
KD-tree Probabilistic Conjunction-Screening Prefilter

Implementation of the Mercurio & Singla KD-tree conjunction-screening method:
a KD-tree prefilter built on mean RSO positions, with a modified nearest-neighbour
search that ranks neighbours by Hellinger distance between uncertainty distributions
while retaining Euclidean distance for KD-tree box pruning.
"""

from .data_gen import (
    generate_gaussian_catalog,
    generate_gmm_catalog,
)
from .evaluate import (
    knn_sweep,
    miss_analysis,
    recall_vs_brute,
    scaling_experiment,
)
from .gmm_hellinger import (
    GMM,
    hellinger_gmm,
    hellinger_gmm_approx,
    hellinger_gmm_component,
    hellinger_gmm_lp,
)
from .hellinger import (
    hellinger_gaussian,
    hellinger_gaussian_pairwise,
    hellinger_gaussian_topk,
    hellinger_gaussian_vectorized,
)
from .kdtree_custom import (
    KDTreeNode,
    build_kdtree,
    dist_point_to_box,
    knn_euclidean,
    modified_nn_search,
)
from .kdtree_scipy_adapter import (
    knn_euclidean_scipy,
    modified_nn_search_hybrid,
)
from .prefilter import (
    ProbabilisticDistance,
    prefilter,
    prefilter_brute_force,
)

__all__ = [
    # gmm_hellinger
    "GMM",
    # kdtree_custom
    "KDTreeNode",
    # prefilter
    "ProbabilisticDistance",
    "build_kdtree",
    "dist_point_to_box",
    # data_gen
    "generate_gaussian_catalog",
    "generate_gmm_catalog",
    # hellinger
    "hellinger_gaussian",
    "hellinger_gaussian_pairwise",
    "hellinger_gaussian_topk",
    "hellinger_gaussian_vectorized",
    "hellinger_gmm",
    "hellinger_gmm_approx",
    "hellinger_gmm_component",
    "hellinger_gmm_lp",
    "knn_euclidean",
    # kdtree_scipy_adapter
    "knn_euclidean_scipy",
    "knn_sweep",
    "miss_analysis",
    "modified_nn_search",
    "modified_nn_search_hybrid",
    "prefilter",
    "prefilter_brute_force",
    # evaluate
    "recall_vs_brute",
    "scaling_experiment",
]

__version__ = "0.1.0"
