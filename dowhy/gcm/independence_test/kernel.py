"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

from typing import Callable, List, Optional, Union

import numpy as np
import scipy
from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd
from joblib import Parallel, delayed
from sklearn.preprocessing import scale

import dowhy.gcm.config as config
from dowhy.gcm.independence_test.kernel_operation import approximate_rbf_kernel_features
from dowhy.gcm.stats import quantile_based_fwer
from dowhy.gcm.util.general import apply_one_hot_encoding, fit_one_hot_encoders, set_random_seed, shape_into_2d


def kernel_based(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    use_bootstrap: bool = True,
    bootstrap_num_runs: int = 10,
    bootstrap_num_samples_per_run: int = 2000,
    bootstrap_n_jobs: Optional[int] = None,
    p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float] = quantile_based_fwer,
    **kwargs,
) -> float:
    """Prepares the data and uses kernel (conditional) independence test. The independence test estimates a p-value
    for the null hypothesis that X and Y are independent (given Z). Depending whether Z is given, a conditional or
    pairwise independence test is performed.

    Here, we utilize the implementations of the https://github.com/cmu-phil/causal-learn package.

    If Z is given: Using KCI as conditional independence test, i.e. we use https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/KCI/KCI.py#L238.
    If Z is not given: Using KCI as pairwise independence test, i.e. we use https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/KCI/KCI.py#L17.

    Note:
    - The data can be multivariate, i.e. the given input matrices can have multiple columns.
    - Categorical data need to be represented as strings.

    Based on the work:
    - K. Zhang, J. Peters, D. Janzing, B. Schölkopf. *Kernel-based Conditional Independence Test and Application in Causal Discovery*. UAI'11, Pages 804–813, 2011.
    - A. Gretton, K. Fukumizu, C.-H. Teo, L. Song, B. Schölkopf, A. Smola. *A Kernel Statistical Test of Independence*. NIPS 21, 2007.

    For more information about configuring the kernel independence test, see:
    - https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/KCI/KCI.py#L17 (if Z is not given)
    - https://github.com/cmu-phil/causal-learn/blob/main/causallearn/utils/KCI/KCI.py#L238 (if Z is given)

    :param X: Data matrix for observations from X.
    :param Y: Data matrix for observations from Y.
    :param Z: Optional data matrix for observations from Z. This is the conditional variable.
    :param use_bootstrap: If True, the independence tests are performed on multiple subsets of the data and the final
                          p-value is constructed based on the provided p_value_adjust_func function.
    :param bootstrap_num_runs: Number of bootstrap runs (only relevant if use_bootstrap is True).
    :param bootstrap_num_samples_per_run: Number of samples used in a bootstrap run (only relevant if use_bootstrap is
                                          True).
    :param bootstrap_n_jobs: Number of parallel jobs for the bootstrap runs.
    :param p_value_adjust_func: A callable that expects a numpy array of multiple p-values and returns one p-value. This
                                is typically used a family wise error rate control method.
    :return: The p-value for the null hypothesis that X and Y are independent (given Z).
    """
    bootstrap_n_jobs = config.default_n_jobs if bootstrap_n_jobs is None else bootstrap_n_jobs

    X = _remove_constant_columns(X)
    Y = _remove_constant_columns(Y)

    if X.shape[1] == 0 or Y.shape[1] == 0:
        # Either X and/or Y is constant.
        return 1.0

    if Z is not None:
        Z = _remove_constant_columns(Z)
        if Z.shape[1] == 0:
            # If Z is empty, we are in the pairwise setting.
            Z = None

    if "est_width" not in kwargs:
        kwargs["est_width"] = "median"

    def evaluate_kernel_test_on_samples(
        X: np.ndarray, Y: np.ndarray, Z: np.ndarray, parallel_random_seed: int
    ) -> float:
        set_random_seed(parallel_random_seed)

        if Z is None:
            X, Y = _convert_to_numeric(*shape_into_2d(X, Y))
            return KCI_UInd(**kwargs).compute_pvalue(X, Y)[0]
        else:
            X, Y, Z = _convert_to_numeric(*shape_into_2d(X, Y, Z))
            return KCI_CInd(**kwargs).compute_pvalue(X, Y, Z)[0]

    if use_bootstrap and X.shape[0] > bootstrap_num_samples_per_run:
        random_indices = [
            np.random.choice(X.shape[0], min(X.shape[0], bootstrap_num_samples_per_run), replace=False)
            for run in range(bootstrap_num_runs)
        ]

        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(random_indices))
        p_values = Parallel(n_jobs=bootstrap_n_jobs)(
            delayed(evaluate_kernel_test_on_samples)(
                X[indices], Y[indices], Z[indices] if Z is not None else None, random_seed
            )
            for indices, random_seed in zip(random_indices, random_seeds)
        )

        return p_value_adjust_func(p_values)
    else:
        return evaluate_kernel_test_on_samples(X, Y, Z, np.random.randint(np.iinfo(np.int32).max, size=1)[0])


def approx_kernel_based(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    num_random_features_X: int = 50,
    num_random_features_Y: int = 50,
    num_random_features_Z: int = 50,
    num_permutations: int = 100,
    approx_kernel: Callable[[np.ndarray], np.ndarray] = approximate_rbf_kernel_features,
    scale_data: bool = False,
    use_bootstrap: bool = True,
    bootstrap_num_runs: int = 10,
    bootstrap_num_samples: int = 1000,
    bootstrap_n_jobs: Optional[int] = None,
    p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float] = quantile_based_fwer,
) -> float:
    """Implementation of the Randomized Conditional Independence Test. The independence test estimates a p-value
    for the null hypothesis that X and Y are independent (given Z). Depending whether Z is given, a conditional or
    pairwise independence test is performed.

    If Z is given: Using RCIT as conditional independence test.
    If Z is not given: Using RIT as pairwise independence test.

    Note:
    - The data can be multivariate, i.e. the given input matrices can have multiple columns.
    - Categorical data need to be represented as strings.
    - It is possible to apply a different kernel to each column in the matrices. For instance, a RBF kernel for the
      first dimension in X and a delta kernel for the second.

    Based on the work:
        Strobl, Eric V., Kun Zhang, and Shyam Visweswaran.
        Approximate kernel-based conditional independence tests for fast non-parametric causal discovery.
        Journal of Causal Inference 7.1 (2019).

    :param X: Data matrix for observations from X.
    :param Y: Data matrix for observations from Y.
    :param Z: Optional data matrix for observations from Z. This is the conditional variable.
    :param num_random_features_X: Number of features sampled from the approximated kernel map for X.
    :param num_random_features_Y: Number of features sampled from the approximated kernel map for Y.
    :param num_random_features_Z: Number of features sampled from the approximated kernel map for Z.
    :param num_permutations: Number of permutations for estimating the test test statistic.
    :param approx_kernel: The approximated kernel map. The expected input is a n x d numpy array and the output is
                          expected to be a n x k numpy array with k << d. By default, the Nystroem method with a RBF
                          kernel is used.
    :param scale_data: If set to True, the data will be standardized. If set to False, the data is taken as it is.
                       Standardizing the data helps in identifying weak dependencies. If one is only interested in
                       stronger ones, consider setting this to False.
    :param use_bootstrap: If True, the independence tests are performed on multiple subsets of the data and the final
                          p-value is constructed based on the provided p_value_adjust_func function.
    :param bootstrap_num_runs: Number of bootstrap runs (only relevant if use_bootstrap is True).
    :param bootstrap_num_samples: Maximum number of used samples per bootstrap run.
    :param bootstrap_n_jobs: Number of parallel jobs for the bootstrap runs.
    :param p_value_adjust_func: A callable that expects a numpy array of multiple p-values and returns one p-value. This
                                is typically used a family wise error rate control method.
    :return: The p-value for the null hypothesis that X and Y are independent (given Z).
    """
    bootstrap_n_jobs = config.default_n_jobs if bootstrap_n_jobs is None else bootstrap_n_jobs

    X = _remove_constant_columns(X)
    Y = _remove_constant_columns(Y)

    if X.shape[1] == 0 or Y.shape[1] == 0:
        # Either X and/or Y is constant.
        return 1.0

    if Z is not None:
        Z = _remove_constant_columns(Z)
        if Z.shape[1] == 0:
            # If Z is empty, we are in the pairwise setting.
            Z = None

    if not use_bootstrap:
        bootstrap_num_runs = 1
        bootstrap_num_samples = float("inf")
        bootstrap_n_jobs = 1

    if Z is None:
        return _rit(
            X,
            Y,
            num_permutations=num_permutations,
            num_random_features_X=num_random_features_X,
            num_random_features_Y=num_random_features_Y,
            num_runs=bootstrap_num_runs,
            num_max_samples_per_run=bootstrap_num_samples,
            approx_kernel=approx_kernel,
            scale_data=scale_data,
            n_jobs=bootstrap_n_jobs,
            p_value_adjust_func=p_value_adjust_func,
        )
    else:
        return _rcit(
            X,
            Y,
            Z,
            num_permutations=num_permutations,
            num_random_features_X=num_random_features_X,
            num_random_features_Y=num_random_features_Y,
            num_random_features_Z=num_random_features_Z,
            num_runs=bootstrap_num_runs,
            num_max_samples_per_run=bootstrap_num_samples,
            approx_kernel=approx_kernel,
            scale_data=scale_data,
            n_jobs=bootstrap_n_jobs,
            p_value_adjust_func=p_value_adjust_func,
        )


def _rit(
    X: np.ndarray,
    Y: np.ndarray,
    num_random_features_X: int,
    num_random_features_Y: int,
    num_permutations: int,
    num_runs: int,
    num_max_samples_per_run: int,
    approx_kernel: Callable[[np.ndarray, int], np.ndarray],
    scale_data: bool,
    n_jobs: Optional[int],
    p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float],
) -> float:
    """Implementation of the Randomized Independence Test based on the work:
    Strobl, Eric V., Kun Zhang, and Shyam Visweswaran.
    Approximate kernel-based conditional independence tests for fast non-parametric causal discovery.
    Journal of Causal Inference 7.1 (2019).
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    X, Y = _convert_to_numeric(*shape_into_2d(X, Y))

    if scale_data:
        X = scale(X)
        Y = scale(Y)

    def evaluate_rit_on_samples(parallel_random_seed: int):
        set_random_seed(parallel_random_seed)

        if X.shape[0] > num_max_samples_per_run:
            random_indices = np.random.choice(X.shape[0], num_max_samples_per_run, replace=False)
            X_samples = X[random_indices]
            Y_samples = Y[random_indices]
        else:
            X_samples = X
            Y_samples = Y

        random_features_x = scale(approx_kernel(X_samples, num_random_features_X))
        random_features_y = scale(approx_kernel(Y_samples, num_random_features_Y))

        permutation_results_of_statistic = []
        for i in range(num_permutations):
            permutation_results_of_statistic.append(
                _estimate_rit_statistic(
                    random_features_x[
                        np.random.choice(random_features_x.shape[0], random_features_x.shape[0], replace=False)
                    ],
                    random_features_y,
                )
            )

        return 1 - (
            np.sum(_estimate_rit_statistic(random_features_x, random_features_y) > permutation_results_of_statistic)
            / len(permutation_results_of_statistic)
        )

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=num_runs)
    p_values = Parallel(n_jobs=n_jobs)(delayed(evaluate_rit_on_samples)(random_seeds[i]) for i in range(num_runs))

    return p_value_adjust_func(p_values)


def _rcit(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    num_random_features_X: int,
    num_random_features_Y: int,
    num_random_features_Z: int,
    num_permutations: int,
    num_runs: int,
    num_max_samples_per_run: int,
    approx_kernel: Callable[[np.ndarray, int], np.ndarray],
    scale_data: bool,
    n_jobs: Optional[int],
    p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float],
) -> float:
    """
    Implementation of the Randomized Conditional Independence Test based on the work:
        Strobl, Eric V., Kun Zhang, and Shyam Visweswaran.
        Approximate kernel-based conditional independence tests for fast non-parametric causal discovery.
        Journal of Causal Inference 7.1 (2019).
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    X, Y, Z = _convert_to_numeric(*shape_into_2d(X, Y, Z))

    if scale_data:
        X = scale(X)
        Y = scale(Y)
        Z = scale(Z)

    def parallel_job(parallel_random_seed: int):
        set_random_seed(parallel_random_seed)

        if X.shape[0] > num_max_samples_per_run:
            random_indices = np.random.choice(X.shape[0], num_max_samples_per_run, replace=False)
            X_samples = X[random_indices]
            Y_samples = Y[random_indices]
            Z_samples = Z[random_indices]
        else:
            X_samples = X
            Y_samples = Y
            Z_samples = Z

        Y_samples = np.column_stack([Y_samples, Z_samples])
        random_features_x = scale(approx_kernel(X_samples, num_random_features_X))
        random_features_y = scale(approx_kernel(Y_samples, num_random_features_Y))
        random_features_z = scale(approx_kernel(Z_samples, num_random_features_Z))

        cov_zz = _estimate_column_wise_covariances(random_features_z, random_features_z)
        inverse_cov_zz = scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(cov_zz + np.eye(cov_zz.shape[0]) * 10**-10, lower=True), np.eye(cov_zz.shape[0])
        )
        cov_xz = _estimate_column_wise_covariances(random_features_x, random_features_z)
        cov_zy = _estimate_column_wise_covariances(random_features_z, random_features_y)

        z_inverse_cov_zz = random_features_z @ inverse_cov_zz

        residual_x = random_features_x - z_inverse_cov_zz @ cov_xz.T
        residual_y = random_features_y - z_inverse_cov_zz @ cov_zy

        # Estimate test statistic multiple times on different permutations of the data. The p-value is then the
        # probability (i.e. fraction) of obtaining a test statistic that is greater than statistic on the non-permuted
        # data.
        permutation_results_of_statistic = []
        for i in range(num_permutations):
            permutation_results_of_statistic.append(
                _estimate_rit_statistic(
                    residual_x[np.random.choice(residual_x.shape[0], residual_x.shape[0], replace=False)], residual_y
                )
            )

        return 1 - (
            np.sum(_estimate_rit_statistic(residual_x, residual_y) > permutation_results_of_statistic)
            / len(permutation_results_of_statistic)
        )

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=num_runs)
    p_values = Parallel(n_jobs=n_jobs)(delayed(parallel_job)(random_seeds[i]) for i in range(num_runs))

    return p_value_adjust_func(p_values)


def _estimate_rit_statistic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X.shape[0] * np.sum(_estimate_column_wise_covariances(X, Y) ** 2)


def _estimate_column_wise_covariances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.cov(X, Y, rowvar=False)[: X.shape[1], -Y.shape[1] :]


def _convert_to_numeric(*args) -> List[np.ndarray]:
    result = []
    for X in args:
        X = np.array(X)
        for col in range(X.shape[1]):
            if isinstance(X[0, col], bool):
                X[:, col] = X[:, col].astype(str)

        result.append(apply_one_hot_encoding(X, fit_one_hot_encoders(X)))

    return result


def _remove_constant_columns(X: np.ndarray) -> np.ndarray:
    X = shape_into_2d(X)
    return X[:, [np.unique(X[:, i]).shape[0] > 1 for i in range(X.shape[1])]]
