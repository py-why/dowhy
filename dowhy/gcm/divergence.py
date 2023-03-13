"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors

from dowhy.gcm.constant import EPS
from dowhy.gcm.util.general import is_categorical, setdiff2d, shape_into_2d


def auto_estimate_kl_divergence(X: np.ndarray, Y: np.ndarray) -> float:
    if is_categorical(X):
        return estimate_kl_divergence_categorical(X, Y)
    elif is_probability_matrix(X):
        return estimate_kl_divergence_of_probabilities(X, Y)
    else:
        return estimate_kl_divergence_continuous(X, Y)


def estimate_kl_divergence_continuous(
    X: np.ndarray, Y: np.ndarray, k: int = 1, remove_common_elements: bool = True
) -> float:
    """Estimates KL-Divergence using k-nearest neighbours (Wang et al., 2009).

    Q. Wang, S. R. Kulkarni, and S. Verdú,
    "Divergence estimation for multidimensional densities via k-nearest-neighbor distances",
    IEEE Transactions on Information Theory, vol. 55, no. 5, pp. 2392-2405, May 2009.

    :param X: (N_1,D) Sample drawn from distribution P_X
    :param Y: (N_2,D) Sample drawn from distribution P_Y
    :param k: Number of neighbors to consider.
    :param remove_common_elements: If true, common values in X and Y are removed. This would otherwise lead to
                                   a KNN distance of zero for these values if k is set to 1, which would cause a
                                   division by zero error.
    return: Estimated value of D(P_X||P_Y).
    """
    X, Y = shape_into_2d(X, Y)

    if X.shape[1] != Y.shape[1]:
        raise RuntimeError(
            "Samples from X and Y need to have the same dimension, but X has dimension %d and Y has "
            "dimension %d." % (X.shape[1], Y.shape[1])
        )

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Making sure that X and Y have no overlapping values, which would lead to a distance of 0 with k=1 and, thus, to
    # a division by zero.
    if remove_common_elements:
        X = setdiff2d(X, Y, assume_unique=True)

    n, m = X.shape[0], Y.shape[0]
    if n == 0:
        return 0

    d = float(X.shape[1])

    x_neighbourhood = NearestNeighbors(n_neighbors=k + 1).fit(X)
    y_neighbourhood = NearestNeighbors(n_neighbors=k).fit(Y)

    distances_x, _ = x_neighbourhood.kneighbors(X, n_neighbors=k + 1)
    distances_y, _ = y_neighbourhood.kneighbors(X, n_neighbors=k)

    rho = distances_x[:, -1]
    nu = distances_y[:, -1]

    result = np.sum((d / n) * np.log(nu / rho)) + np.log(m / (n - 1))

    if ~np.isfinite(result):
        raise RuntimeError(
            "Got a non-finite KL divergence! This can happen if both data sets have overlapping "
            "elements. Since these are normally removed by this method, double check whether the arrays "
            "are numeric."
        )

    if result < 0:
        result = 0

    return result


def estimate_kl_divergence_categorical(X: np.ndarray, Y: np.ndarray) -> float:
    X, Y = shape_into_2d(X, Y)

    if X.shape[1] != Y.shape[1]:
        raise RuntimeError(
            "Samples from X and Y need to have the same dimension, but X has dimension %d and Y has "
            "dimension %d." % (X.shape[1], Y.shape[1])
        )

    all_uniques = np.unique(np.vstack([X, Y]))

    p = np.array([(np.sum(X == i) + EPS) / (X.shape[0] + EPS) for i in all_uniques])
    q = np.array([(np.sum(Y == i) + EPS) / (Y.shape[0] + EPS) for i in all_uniques])

    return float(np.sum(p * np.log(p / q)))


def estimate_kl_divergence_of_probabilities(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimates the Kullback-Leibler divergence between each pair of probability vectors (row wise) in X and Y
    separately and returns the mean over all results."""
    X, Y = shape_into_2d(X, Y)

    if X.shape[1] != Y.shape[1]:
        raise RuntimeError(
            "Samples from X and Y need to have the same dimension, but X has dimension %d and Y has "
            "dimension %d." % (X.shape[1], Y.shape[1])
        )

    return float(np.mean(entropy(X + EPS, Y + EPS, axis=1)))


def is_probability_matrix(X: np.ndarray) -> bool:
    if X.ndim == 1:
        return np.all(np.isclose(np.sum(abs(X.astype(np.float64)), axis=0), 1))
    else:
        return np.all(np.isclose(np.sum(abs(X.astype(np.float64)), axis=1), 1))
