import random
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder

from dowhy.gcm.util.catboost_encoder import CatBoostEncoder


def shape_into_2d(*args):
    """If necessary, shapes the numpy inputs into 2D matrices.

    Example:
        array([1, 2, 3]) -> array([[1], [2], [3]])
        2 -> array([[2]])

    :param args: The function expects numpy arrays as inputs and returns a reshaped (2D) version of them (if necessary).
    :return: Reshaped versions of the input numpy arrays. For instance, given 1D inputs X, Y and Z, then
             shape_into_2d(X, Y, Z) reshapes them into 2D and returns them. If an input is already 2D, it will not be
             modified and returned as it is.
    """

    def shaping(X: np.ndarray):
        if X.ndim < 2:
            return np.column_stack([X])
        elif X.ndim > 2:
            raise ValueError("Cannot reshape a %dD array into a 2D array!" % X.ndim)

        return X

    result = [shaping(x) for x in args]

    if len(result) == 1:
        return result[0]
    else:
        return result


def set_random_seed(random_seed: int) -> None:
    """Sets random seed in numpy and the random module.

    :param random_seed: Random see for the numpy and random module.
    :return: None
    """
    np.random.seed(random_seed)
    random.seed(random_seed)


def auto_fit_encoders(
    X: np.ndarray, Y: Optional[np.ndarray] = None, catboost_threshold: int = 7
) -> Dict[int, Union[OneHotEncoder, CatBoostEncoder]]:
    if Y is None:
        return fit_one_hot_encoders(X)

    X = shape_into_2d(X)

    total_num_categories = 0
    for column in range(X.shape[1]):
        if is_categorical(X[:, column]):
            total_num_categories += len(np.unique(X[:, column]))

    if total_num_categories > catboost_threshold:
        return fit_catboost_encoders(X, Y)
    else:
        return fit_one_hot_encoders(X)


def auto_apply_encoders(
    X: np.ndarray, encoder_map: Dict[int, Union[OneHotEncoder, CatBoostEncoder]], Y: Optional[np.ndarray] = None
) -> np.ndarray:
    X = shape_into_2d(X)

    if not encoder_map:
        return X

    if isinstance(list(encoder_map.values())[0], OneHotEncoder):
        return apply_one_hot_encoding(X, encoder_map)
    else:
        return apply_catboost_encoding(X, encoder_map, Y)


def fit_one_hot_encoders(X: np.ndarray) -> Dict[int, OneHotEncoder]:
    """Fits one-hot encoders to each categorical column in X. A categorical input needs to be a string, i.e. a
    categorical column consists only of strings.

    :param X: Input data matrix.
    :return: Dictionary that maps a column index to a scikit OneHotEncoder.
    """
    X = shape_into_2d(X)

    one_hot_encoders = {}
    for column in range(X.shape[1]):
        if is_categorical(X[:, column]):
            one_hot_encoders[column] = OneHotEncoder(handle_unknown="ignore")
            one_hot_encoders[column].fit(X[:, column].reshape(-1, 1))

    return one_hot_encoders


def apply_one_hot_encoding(X: np.ndarray, one_hot_encoder_map: Dict[int, OneHotEncoder]) -> np.ndarray:
    X = shape_into_2d(X)

    if not one_hot_encoder_map:
        return X

    one_hot_features = []

    for column in range(X.shape[1]):
        if column in one_hot_encoder_map:
            one_hot_features.append(one_hot_encoder_map[column].transform(X[:, column].reshape(-1, 1)).toarray())
        else:
            one_hot_features.append(X[:, column].reshape(-1, 1))

    return np.hstack(one_hot_features).astype(float)


def fit_catboost_encoders(X: np.ndarray, Y: np.ndarray) -> Dict[int, CatBoostEncoder]:
    X = shape_into_2d(X)

    catboost_encoders = {}
    for column in range(X.shape[1]):
        if is_categorical(X[:, column]):
            catboost_encoders[column] = CatBoostEncoder()
            catboost_encoders[column].fit(X[:, column], Y)

    return catboost_encoders


def apply_catboost_encoding(
    X: np.ndarray, catboost_encoder_map: Dict[int, CatBoostEncoder], Y: Optional[np.ndarray] = None
) -> np.ndarray:
    X = shape_into_2d(X)

    if not catboost_encoder_map:
        return X

    one_hot_features = []

    for column in range(X.shape[1]):
        if column in catboost_encoder_map:
            one_hot_features.append(catboost_encoder_map[column].transform(X[:, column], Y).reshape(-1, 1))
        else:
            one_hot_features.append(X[:, column].reshape(-1, 1))

    return np.hstack(one_hot_features).astype(float)


def is_categorical(X: np.ndarray) -> bool:
    """Checks if all of the given columns are categorical, i.e. either a string or a boolean. Only if all of the
    columns are categorical, this method will return True. Alternatively, consider has_categorical for checking if any
    of the columns is categorical.

    Note: A np matrix with mixed data types might internally convert numeric columns to strings and vice versa. To
    ensure that the given given data keeps the original data type, consider converting/initializing it with the dtype
    'object'. For instance: np.array([[1, 'True', '0', 0.2], [3, 'False', '1', 2.3]], dtype=object)

    :param X: Input array to check if all columns are categorical.
    :return: True if all columns of the input are categorical, False otherwise.
    """
    X = shape_into_2d(X)

    nan_mask = pd.isna(X).any(axis=1)

    status = True
    for column in range(X.shape[1]):
        X = X[~nan_mask]

        if X.shape[0] == 0:
            return False

        status &= isinstance(X[0, column], str) or isinstance(X[0, column], bool) or isinstance(X[0, column], np.bool_)

        if not status:
            break

    if status and nan_mask.any():
        raise ValueError(
            "The target variable appears to be categorical and has missing data. Currently, only missing "
            "data for numerical variables is supported! Consider treating the missing category separately"
        )

    return status


def has_categorical(X: np.ndarray) -> bool:
    """Checks if any of the given columns are categorical, i.e. either a string or a boolean. If any of the columns
    is categorical, this method will return True. Alternatively, consider is_categorical for checking if all columns are
    categorical.

    Note: A np matrix with mixed data types might internally convert numeric columns to strings and vice versa. To
    ensure that the given given data keeps the original data type, consider converting/initializing it with the dtype
    'object'. For instance: np.array([[1, 'True', '0', 0.2], [3, 'False', '1', 2.3]], dtype=object)

    :param X: Input array to check if all columns are categorical.
    :return: True if all columns of the input are categorical, False otherwise.
    """
    X = shape_into_2d(X)

    for column in range(X.shape[1]):
        if is_categorical(X[:, column]):
            return True

    return False


def is_discrete(X: np.ndarray) -> bool:
    """Checks if all values in the given array are discrete.

    :param X: Input array to check.
    :return: True if all values in the input are discrete, False otherwise.
    """
    return np.all(X == np.floor(X))


def setdiff2d(ar1: np.ndarray, ar2: np.ndarray, assume_unique: bool = False) -> np.ndarray:
    """This method generalizes numpy's setdiff1d to 2d, i.e., it compares vectors for arbitrary length. See
    https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html for more details."""
    if ar1.ndim == ar2.ndim != 2:
        raise ValueError("Only support 2D arrays!")

    if ar1.shape[1] != ar2.shape[1]:
        return ar1

    dtype = {"names": ["f{}".format(i) for i in range(ar1.shape[1])], "formats": ar1.shape[1] * [ar1.dtype]}

    if not ar1.flags["C_CONTIGUOUS"]:
        ar1 = np.ascontiguousarray(ar1)
    if not ar2.flags["C_CONTIGUOUS"]:
        ar2 = np.ascontiguousarray(ar2)

    return (
        np.setdiff1d(ar1.view(dtype), ar2.view(dtype), assume_unique=assume_unique)
        .view(ar1.dtype)
        .reshape(-1, ar1.shape[1])
    )


def means_difference(randomized_predictions: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    return np.mean(randomized_predictions).squeeze() - np.mean(baseline_values).squeeze()


def variance_of_deviations(randomized_predictions: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    # Using the negative value here seeing that the Shapley estimation evaluates v(S u {i}) - v(S) for a subset S. In
    # case of variance, we have v(S u {i}) <= v(S), which would result in a negative contribution of players to the
    # target quantity (here, variance).
    return -np.var((randomized_predictions - baseline_values).squeeze())


def variance_of_matching_values(randomized_predictions: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    # Using the negative value here seeing that the Shapley estimation evaluates v(S u {i}) - v(S) for a subset S. In
    # case of variance, we have v(S u {i}) <= v(S), which would result in a negative contribution of players to the
    # target quantity (here, variance).
    return -np.var((randomized_predictions == baseline_values).squeeze())


def geometric_median(x: np.ndarray) -> np.ndarray:
    def distance_function(x_input: np.ndarray) -> np.ndarray:
        return np.sum(np.sqrt(np.sum((x_input - x) ** 2, axis=1)))

    return minimize(distance_function, np.sum(x, axis=0) / x.shape[0]).x
