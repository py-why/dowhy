"""Functions to estimate uncertainties such as entropy, KL divergence etc."""

import numpy as np
from numpy.linalg import det
from scipy.special import digamma
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

from dowhy.gcm.constant import EPS
from dowhy.gcm.util.general import shape_into_2d


def estimate_entropy_using_discretization(X: np.ndarray, bin_width: float = 1) -> float:
    # TODO: Add method for auto select a bin_width/width based on the data. Make sure that the auto selection method is
    #  theoretically sound, i.e. make entropy results from different data comparable.
    X = shape_into_2d(X)

    if X.shape[1] > 1:
        raise RuntimeError(
            "The discrete entropy estimator can only handle one dimensional data, but the input data is "
            "%d dimensional!" % X.shape[1]
        )

    max_value = np.max(X)
    min_value = np.min(X)
    number_of_bins = max(1, int((max_value - min_value) / bin_width))
    num_samples = X.shape[0]
    return -np.sum(
        [
            (i / num_samples * np.log(i / num_samples)) if i > 0 else 0
            for i in np.histogram(X, bins=np.linspace(min_value, max_value, number_of_bins).reshape(-1))[0]
        ]
    )


def estimate_entropy_kmeans(X: np.ndarray) -> float:
    """Related paper:
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi
    Informatsii, 23(2), 9–16.
    """
    X = shape_into_2d(X)

    k = int(np.sqrt(X.shape[0]))

    x_neighbourhood = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = x_neighbourhood.kneighbors(X, k + 1)
    distances = distances[:, -1]

    sum_log_dist = np.sum(np.log(2 * distances + EPS))

    return -digamma(k) + digamma(X.shape[0]) + (X.shape[1] / float(X.shape[0])) * sum_log_dist


def estimate_gaussian_entropy(X: np.ndarray) -> float:
    """Entropy with respect to standardized variables."""
    X = shape_into_2d(X)

    if X.shape[1] > 1:
        return 0.5 * np.log((2 * np.pi * np.e) ** X.shape[1] * estimate_variance(X))
    else:
        return 0.5 * np.log(2 * np.pi * np.e * estimate_variance(X))


def estimate_variance(X: np.ndarray) -> float:
    X = shape_into_2d(X)

    if X.shape[1] > 1:
        result = det(np.cov(X, rowvar=False))
    else:
        result = np.var(X)

    # Extremely small values can somehow result in negative values.
    return max(0.0, result)


def estimate_entropy_of_probabilities(X: np.ndarray) -> float:
    """Estimates the entropy of each probability vector (row wise) in X separately and returns the mean over all
    results.
    """
    return float(np.mean(entropy(X, axis=1)))


def estimate_entropy_discrete(X: np.ndarray) -> float:
    """Estimates the entropy assuming the data in X is discrete.

    :param X: Discrete samples.
    :return: Entropy of X.
    """
    X = shape_into_2d(X)

    _, counts = np.unique(X, return_counts=True, axis=0)
    return entropy(counts)
