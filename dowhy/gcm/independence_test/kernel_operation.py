"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

from typing import Optional, List, Callable

import numpy
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import scale

from dowhy.gcm.util.general import shape_into_2d, is_categorical


def apply_rbf_kernel(X: numpy.ndarray,
                     scale_data: bool = True,
                     precision: Optional[float] = None) -> numpy.ndarray:
    """Estimates the RBF (Gaussian) kernel for the given input data.

    :param X: Input data.
    :param scale_data: True if the data should be standardize.
    :param precision: Specific precision matrix for the RBF kernel. If None is given, this is inferred from the data.
    :return: The outcome of applying a RBF (Gaussian) kernel on the data.
    """
    X = shape_into_2d(X)
    if scale_data:
        X = scale(X)

    distance_matrix = euclidean_distances(X, squared=True)

    if precision is None:
        tmp = numpy.sqrt(distance_matrix)
        tmp = tmp - numpy.tril(tmp, -1)
        tmp = tmp.reshape(-1, 1)
        precision = 1 / numpy.median(tmp[tmp > 0])

    return numpy.exp(-precision * distance_matrix)


def apply_delta_kernel(X: numpy.ndarray) -> numpy.ndarray:
    """Applies the delta kernel, i.e. the distance is 1 if two entries are equal and 0 otherwise.

    :param X: Input data.
    :return: The outcome of the delta-kernel, a binary distance matrix.
    """
    X = shape_into_2d(X)
    return numpy.array(list(map(lambda value: value == X, X))).reshape(X.shape[0], X.shape[0]).astype(numpy.float)


def approximate_rbf_kernel_features(X: numpy.ndarray,
                                    num_random_components: int,
                                    scale_data: bool = False,
                                    precision: Optional[float] = None) -> numpy.ndarray:
    """Applies the Nystroem method to create a NxD (D << N) approximated RBF kernel map using a subset of the data,
    where N is the number of samples in X and D the number of components.

    :param X: Input data.
    :param num_random_components: Number of components D for the approximated kernel map.
    :param scale_data: Specific precision matrix for the RBF kernel. If None is given, this is inferred from the data.
    :param precision:
    :return: A NxD approximated RBF kernel map, where N is the number of samples in X and D the number of components.
    """
    X = shape_into_2d(X)
    if scale_data:
        X = scale(X)

    if precision is None:
        tmp = numpy.sqrt(euclidean_distances(X, squared=True))
        tmp = tmp - numpy.tril(tmp, -1)
        tmp = tmp.reshape(-1, 1)
        precision = 1 / numpy.median(tmp[tmp > 0])

    return Nystroem(kernel='rbf', gamma=precision, n_components=num_random_components).fit_transform(X)


def approximate_delta_kernel_features(X: numpy.ndarray, num_random_components: int) -> numpy.ndarray:
    """Applies the Nystroem method to create a NxD (D << N) approximated delta kernel map using a subset of the data,
    where N is the number of samples in X and D the number of components. The delta kernel gives 1 if two entries are
    equal and 0 otherwise.

    :param X: Input data.
    :param num_random_components: Number of components D for the approximated kernel map.
    :param scale_data: Specific precision matrix for the RBF kernel. If None is given, this is inferred from the data.
    :param precision:
    :return: A NxD approximated RBF kernel map, where N is the number of samples in X and D the number of components.
    """
    X = shape_into_2d(X)

    def delta_function(x, y) -> float:
        return float(x == y)

    for i, unique_element in enumerate(numpy.unique(X)):
        X[X == unique_element] = i

    result = Nystroem(kernel=delta_function, n_components=num_random_components).fit_transform(X.astype(int))
    result[result != 0] = 1

    return result


def auto_create_list_of_kernels(X: numpy.ndarray) -> List[Callable[[numpy.ndarray], numpy.ndarray]]:
    X = shape_into_2d(X)

    tmp_list = []
    for i in range(X.shape[1]):
        if not is_categorical(X[:, i]):
            tmp_list.append(apply_rbf_kernel)
        else:
            tmp_list.append(apply_delta_kernel)

    return tmp_list


def auto_create_list_of_kernel_approximations(X: numpy.ndarray) -> List[Callable[[numpy.ndarray, int], numpy.ndarray]]:
    X = shape_into_2d(X)

    tmp_list = []
    for i in range(X.shape[1]):
        if not is_categorical(X[:, i]):
            tmp_list.append(approximate_rbf_kernel_features)
        else:
            tmp_list.append(approximate_delta_kernel_features)

    return tmp_list
