from collections import Counter
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
    r"""Estimates KL-Divergence based on probabilities given by classifier. This is:

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


# Private sentinel for missing values. A unique object cannot collide with a legitimate category such as "nan".
_MISSING_CATEGORY = object()


def _is_missing(value) -> bool:
    # Missing scalars such as NaN (float, complex, Decimal) and NaT are not equal to themselves, so each occurrence would
    # otherwise become its own category. Only a real boolean result counts: comparisons returning arrays or pd.NA are
    # kept as regular values instead of raising an ambiguous truth value error.
    is_unequal = value != value
    return isinstance(is_unequal, (bool, np.bool_)) and bool(is_unequal)


def _joint_category(row) -> tuple:
    # Map every missing value to one shared category. Other values, including None, "nan" and inf, are kept as is.
    return tuple(_MISSING_CATEGORY if _is_missing(v) else v for v in row)


def _joint_categories_equal(left: tuple, right: tuple) -> bool:
    for left_value, right_value in zip(left, right):
        if left_value is right_value:
            continue
        if isinstance(left_value, (np.datetime64, np.timedelta64)) and type(left_value) is type(right_value):
            ticks = _temporal_sort_keys([left_value, right_value])
            if ticks is not None:
                if ticks[0] != ticks[1]:
                    return False
                continue
        try:
            is_equal = left_value == right_value
        except TypeError:
            # Some unrelated scalars, such as Decimal and timedelta64, cannot be compared.
            return False
        if not isinstance(is_equal, (bool, np.bool_)) or not is_equal:
            return False
    return True


_ATTOSECONDS_PER_UNIT = {
    "W": 7 * 86400 * 10**18,
    "D": 86400 * 10**18,
    "h": 3600 * 10**18,
    "m": 60 * 10**18,
    "s": 10**18,
    "ms": 10**15,
    "us": 10**12,
    "ns": 10**9,
    "ps": 10**6,
    "fs": 10**3,
    "as": 1,
}


def _calendar_month_to_days(months: int) -> int:
    # First day of a proleptic Gregorian month, relative to 1970-01-01. Python integers retain the full NumPy range.
    year, month = 1970 + months // 12, months % 12 + 1
    year -= month <= 2
    era, year_of_era = divmod(year, 400)
    march_month = month + (-3 if month > 2 else 9)
    return (
        era * 146097 + year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + (153 * march_month + 2) // 5 - 719468
    )


def _temporal_sort_keys(values: list):
    metadata = [np.datetime_data(value.dtype) for value in values]
    # Generic durations acquire the other operand's unit during comparison, so retain the equality fallback.
    if any(unit == "generic" for unit, _ in metadata):
        return None
    keys = []
    for value, (unit, step) in zip(values, metadata):
        count = np.asarray(value).view(np.int64).item() * step
        if unit in ("Y", "M"):
            months = count * 12 if unit == "Y" else count
            if isinstance(value, np.timedelta64):
                keys.append(("calendar", months))
                continue
            count, unit = _calendar_month_to_days(months), "D"
        # Arbitrary-precision ticks avoid both overflowing unit casts and NumPy temporal hashing.
        keys.append(("fixed", count * _ATTOSECONDS_PER_UNIT[unit]))
    return keys


def _joint_sort_keys(rows: list):
    keys = [[(value is not _MISSING_CATEGORY, value) for value in row] for row in rows]
    for column_index, column in enumerate(zip(*rows)):
        value_types = {type(value) for value in column if value is not _MISSING_CATEGORY}
        # Mixed types can have non-transitive equality, such as integers and durations in different units.
        if len(value_types) > 1:
            return None
        if value_types & {np.datetime64, np.timedelta64}:
            values = [value for value in column if value is not _MISSING_CATEGORY]
            ticks = _temporal_sort_keys(values)
            if ticks is None:
                return None
            tick_iterator = iter(ticks)
            for index, value in enumerate(column):
                if value is not _MISSING_CATEGORY:
                    keys[index][column_index] = (True, next(tick_iterator))
    return keys


def _sorted_joint_counts(x_rows: list, y_rows: list):
    rows = x_rows + y_rows
    keys = _joint_sort_keys(rows)
    if keys is None:
        return None
    group_ids = [0] * len(rows)
    representative = None
    group = 0
    try:
        for index in sorted(range(len(rows)), key=keys.__getitem__):
            if representative is None:
                representative = index
            elif not _joint_categories_equal(rows[index], rows[representative]):
                # A partial ordering (for example, disjoint sets) cannot safely group equal categories.
                if not keys[representative] < keys[index]:
                    return None
                group += 1
                representative = index
            group_ids[index] = group
    except (TypeError, ValueError):
        # Mixed or non-boolean comparisons may prevent sorting. Use the equality fallback instead.
        return None

    # Count in input order so KL accumulation keeps first-seen order, including categories unique to Y.
    split = len(x_rows)
    return Counter(group_ids[:split]), Counter(group_ids[split:])


def _contains_temporal(values: np.ndarray) -> bool:
    return values.dtype.kind in "mM" or (
        values.dtype.kind == "O" and any(isinstance(value, (np.datetime64, np.timedelta64)) for value in values.flat)
    )


def _joint_counts(X: np.ndarray, Y: np.ndarray):
    # Equal temporal scalars can have different hashes across units. Hashing them also leaks memory in NumPy 2.2.6.
    # Check before hashing any row, including temporal cells in object arrays.
    if not (_contains_temporal(X) or _contains_temporal(Y)):
        try:
            # Stream ordinary hashable rows, retaining only the distinct categories.
            return Counter(map(_joint_category, X)), Counter(map(_joint_category, Y))
        except TypeError:
            # Orderable, unhashable objects were supported by the original np.unique implementation.
            pass

    x_rows = list(map(_joint_category, X))
    y_rows = list(map(_joint_category, Y))
    sorted_counts = _sorted_joint_counts(x_rows, y_rows)
    if sorted_counts is not None:
        return sorted_counts

    # Compare original scalar values without casting to a common temporal unit, which can overflow or lose precision.
    # Share category indices across both samples and retain first-seen order, as on the hashable path.
    categories = []
    counts = []
    for rows in (x_rows, y_rows):
        row_counts = Counter()
        for row in rows:
            index = next(
                (i for i, category in enumerate(categories) if _joint_categories_equal(row, category)), len(categories)
            )
            if index == len(categories):
                categories.append(row)
            row_counts[index] += 1
        counts.append(row_counts)
    return counts


def estimate_kl_divergence_categorical(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimates KL divergence between categorical samples, treating each row as one joint observation.

    Missing values that are not equal to themselves, such as NaN and NaT, are treated as one shared missing category
    within a column.
    """
    X, Y = shape_into_2d(X, Y)

    if X.shape[1] != Y.shape[1]:
        raise RuntimeError(
            "Samples from X and Y need to have the same dimension, but X has dimension %d and Y has "
            "dimension %d." % (X.shape[1], Y.shape[1])
        )

    # Count joint observations rather than pooling categories across columns. Tuples also support object arrays,
    # which are commonly used for categorical data but are not supported by np.unique with axis=0.
    x_counts, y_counts = _joint_counts(X, Y)
    # Keep first-seen order instead of set order, so the summation does not depend on hash randomization.
    all_uniques = list(x_counts) + [i for i in y_counts if i not in x_counts]

    p = np.array([(x_counts[i] + EPS) / (X.shape[0] + EPS) for i in all_uniques])
    q = np.array([(y_counts[i] + EPS) / (Y.shape[0] + EPS) for i in all_uniques])

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
