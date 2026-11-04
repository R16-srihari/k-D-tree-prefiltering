"""Abstract base class for probabilistic distance measures."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProbabilisticDistance(ABC):
    """Common interface for probabilistic distance measures between GMMs.

    The KNN algorithm accepts a distance object and calls ``compute`` to
    rank candidates by probabilistic distance rather than Euclidean distance.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short metric name (e.g. 'hellinger', 'wasserstein')."""

    @abstractmethod
    def compute(self, gmm_a, gmm_b) -> float:
        """Compute the probabilistic distance between two GMMs.

        Parameters
        ----------
        gmm_a, gmm_b : GaussianMixture
            The two uncertainty distributions.

        Returns
        -------
        float
            Probabilistic distance (nonnegative).
        """

    @abstractmethod
    def compute_gaussian(
        self, mean_a, cov_a, mean_b, cov_b
    ) -> float:
        """Compute the distance between two single Gaussians.

        Used as the component-level cost in GMM coupling.
        """


SUPPORTED_METRICS = frozenset({"wasserstein", "mahalanobis", "hellinger", "kl"})