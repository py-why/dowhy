from abc import ABC, abstractmethod

import numpy as np


class AnomalyScorer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fits the anomaly scorer to the given data. Depending on the definition of the scorer, this can imply
        different things, such as fitting a (parametric) distribution to the data or estimating certain properties
        such as mean, variance, median etc. that are used for computing a score.

        :param X: Samples from the underlying data distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
