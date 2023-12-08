"""This module contains implementations of different density estimators."""

from typing import Optional

import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

from dowhy.gcm.density_estimator import DensityEstimator
from dowhy.gcm.util.general import shape_into_2d


class GaussianMixtureDensityEstimator(DensityEstimator):
    """Represents a density estimator based on a Gaussian mixture model. The estimator uses the sklearn
    BayesianGaussianMixture model internally.
    """

    def __init__(self, num_components: Optional[int] = None) -> None:
        self._gmm_model = None
        self._num_components = num_components

    def fit(self, X: np.ndarray) -> None:
        if self._num_components is None:
            self._num_components = int(np.ceil(np.sqrt(X.shape[0] / 2)))

        self._gmm_model = BayesianGaussianMixture(n_components=self._num_components, covariance_type="full").fit(
            shape_into_2d(X)
        )

    def density(self, X: np.ndarray) -> np.ndarray:
        if self._gmm_model is None:
            raise RuntimeError("%s has not been fitted!" % self.__class__.__name__)

        # Note, the output of score_samples are log values.
        return np.exp(self._gmm_model.score_samples(shape_into_2d(X)))


class KernelDensityEstimator1D(DensityEstimator):
    """Represents a kernel based density estimator. The estimator uses the sklearn KernelDensity class internally."""

    def __init__(self) -> None:
        self._kde_model = None

    def fit(self, X: np.ndarray) -> None:
        X = shape_into_2d(X)
        self._validate_data(X)

        bandwidth = np.std(X) * np.power(4 / 3 / X.shape[0], 1 / 5)

        self._kde_model = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X.reshape(-1, 1))

    def _validate_data(self, X: np.ndarray) -> None:
        if X.shape[1] > 1:
            raise RuntimeError("%s only supports one dimensional data!" % self.__class__.__name__)

    def density(self, X: np.ndarray) -> np.ndarray:
        if self._kde_model is None:
            raise RuntimeError("%s has not been fitted!" % self.__class__.__name__)

        X = shape_into_2d(X)
        self._validate_data(X)

        # Note, the output of score_samples are log values.
        return np.exp(self._kde_model.score_samples(X.reshape(-1, 1)))
