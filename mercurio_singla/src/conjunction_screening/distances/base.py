"""Abstract interface for probabilistic distance measures.

The K-d tree screening algorithm interacts with probabilistic distances
exclusively through this interface. It must not import concrete metric
implementations directly, so that alternative measures can be substituted
without modifying the tree or screening code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProbabilisticDistance(ABC):
    """Abstract base class for probabilistic distances between uncertainty
    distributions.

    A probabilistic distance measures dissimilarity between two full
    probability distributions (e.g. two Gaussians or two GMMs), as opposed
    to the Euclidean mean-position distance used for K-d tree geometry.
    """

    @abstractmethod
    def compute(self, distribution_a, distribution_b) -> float:
        """Compute the probabilistic distance between two distributions.

        Parameters
        ----------
        distribution_a : Gaussian | GaussianMixture
            First uncertainty distribution.
        distribution_b : Gaussian | GaussianMixture
            Second uncertainty distribution.

        Returns
        -------
        float
            A non-negative distance or dissimilarity. For the Hellinger
            distance this lies in [0, 1].
        """
        raise NotImplementedError
