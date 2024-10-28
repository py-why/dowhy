from functools import partial
from typing import Callable, Union

import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from dowhy.gcm.auto import AssignmentQuality, select_model
from dowhy.gcm.constant import EPS
from dowhy.gcm.ml.classification import ClassificationModel, create_logistic_regression_classifier
from dowhy.gcm.util.general import has_categorical, is_categorical, setdiff2d, shape_into_2d


def auto_estimate_kl_divergence(X: np.ndarray, Y: np.ndarray) -> float:
    if is_categorical(X):
        return estimate_kl_divergence_categorical(X, Y)
    elif not has_categorical(X) and is_probability_matrix(X):
        return estimate_kl_divergence_of_probabilities(X, Y)
    else:
        if X.ndim == 2 and X.shape[1] > 1:
            return estimate_kl_divergence_continuous_clf(X, Y)
        else:
            return estimate_kl_divergence_continuous_knn(X, Y)


def estimate_kl_divergence_continuous_knn(
    X: np.ndarray, Y: np.ndarray, k: int = 1, remove_common_elements: bool = True, n_jobs: int = 1
) -> float:
    """Estimates KL-Divergence using k-nearest neighbours (Wang et al., 2009).

    While, in theory, this handles multidimensional inputs, consider using estimate_kl_divergence_continuous_clf
    for data with more than one dimension.

    Q. Wang, S. R. Kulkarni, and S. Verdú,
    "Divergence estimation for multidimensional densities via k-nearest-neighbor distances",
    IEEE Transactions on Information Theory, vol. 55, no. 5, pp. 2392-2405, May 2009.

    :param X: (N_1,D) Sample drawn from distribution P_X
    :param Y: (N_2,D) Sample drawn from distribution P_Y
    :param k: Number of neighbors to consider.
    :param remove_common_elements: If true, common values in X and Y are removed. This would otherwise lead to
                                   a KNN distance of zero for these values if k is set to 1, which would cause a
                                   division by zero error.
    :param n_jobs: Number of parallel jobs used for the nearest neighbors model. -1 means it uses all available cores.
                   Note that in most applications, parallelizing this rather introduces more overhead, leading to a
                   slower runtime.
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
        X = setdiff2d(X, Y, assume_unique=False)
        if X.shape[0] < k + 1:
            # All elements are equal (or at least less than k samples are different)
            return 0

    n, m = X.shape[0], Y.shape[0]
    if n == 0:
        return 0

    d = float(X.shape[1])

    x_neighbourhood = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(X)
    y_neighbourhood = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(Y)

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


def estimate_kl_divergence_continuous_clf(
    samples_P: np.ndarray,
    samples_Q: np.ndarray,
    n_splits: int = 5,
    classifier_model: Union[AssignmentQuality, Callable[[], ClassificationModel]] = partial(
        create_logistic_regression_classifier, max_iter=10000
    ),
    epsilon: float = EPS,
) -> float:
    """Estimates KL-Divergence based on probabilities given by classifier. This is:

        D_f(P || Q) = \int f(p(x)/q(x)) q(x) dx ~= -1/N \sum_x log(p(Y = 1 | x) / (1 - p(Y = 1 | x)))

    Here, the KL divergence can be approximated using the log ratios of probabilities to predict whether a sample
    comes from distribution P or Q.

    :param samples_P: Samples drawn from P. Can have a different number of samples than Q.
    :param samples_Q: Samples drawn from Q. Can have a different number of samples than P.
    :param n_splits: Number of splits of the training and test data. The classifier is trained on the training
                     data and evaluated on the test data to obtain the probabilities.
    :param classifier_model: Used to estimate the probabilities for the log ratio. This can either be a
                             ClassificationModel or an AssignmentQuality. In the latter, a model is automatically
                             selected based on the best performance on a training set.
    :param epsilon: If the probability is either 1 or 0, this value will be used for clipping, i.e., 0 becomes epsilon
                    and 1 becomes 1- epsilon.
    :return: Estimated value of the KL divergence D(P||Q).
    """
    samples_P, samples_Q = shape_into_2d(samples_P, samples_Q)

    if samples_P.shape[1] != samples_Q.shape[1]:
        raise ValueError("X and Y need to have the same number of features!")

    all_probs = []

    splits_p = list(KFold(n_splits=n_splits, shuffle=True).split(samples_P))
    splits_q = list(KFold(n_splits=n_splits, shuffle=True).split(samples_Q))

    if isinstance(classifier_model, AssignmentQuality):
        classifier_model = select_model(
            np.vstack([samples_P, samples_Q]),
            np.concatenate([np.zeros(samples_P.shape[0]), np.ones(samples_Q.shape[0])]).astype(str),
            classifier_model,
        )[0]
    else:
        classifier_model = classifier_model()

    for k in range(n_splits):
        # Balance the classes
        num_samples = min(len(splits_p[k][0]), len(splits_q[k][0]))

        classifier_model.fit(
            np.vstack([samples_P[splits_p[k][0][:num_samples]], samples_Q[splits_q[k][0][:num_samples]]]),
            np.concatenate([np.zeros(num_samples), np.ones(num_samples)]).astype(str),
        )

        probs_P = classifier_model.predict_probabilities(samples_P[splits_p[k][1]])[:, 1]
        probs_P[probs_P == 0] = epsilon
        probs_P[probs_P == 1] = 1 - epsilon
        all_probs.append(probs_P)

    all_probs = np.concatenate(all_probs)
    kl_divergence = -np.mean(np.log(all_probs / (1 - all_probs)))

    if kl_divergence < 0:
        kl_divergence = 0

    return kl_divergence


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
    elif X.shape[1] == 1:
        return False
    else:
        return np.all(np.isclose(np.sum(abs(X.astype(np.float64)), axis=1), 1))
