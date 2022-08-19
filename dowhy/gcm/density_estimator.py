from abc import ABC, abstractmethod

import numpy as np


class DensityEstimator(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def density(self, X: np.ndarray) -> np.ndarray:
        """Returns the density of each input."""
        raise NotImplementedError
