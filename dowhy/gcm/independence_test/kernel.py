"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

from typing import Callable, List, Union, Optional, Tuple

import numpy as np
import scipy
from joblib import Parallel, delayed
from numpy.linalg import pinv, svd, LinAlgError
from scipy.stats import gamma
from sklearn.preprocessing import scale

import dowhy.gcm.config as config
from dowhy.gcm.constant import EPS
from dowhy.gcm.independence_test.kernel_operation import approximate_rbf_kernel_features, \
    apply_rbf_kernel_with_adaptive_precision
from dowhy.gcm.stats import quantile_based_fwer
from dowhy.gcm.util.general import set_random_seed, shape_into_2d, apply_one_hot_encoding, fit_one_hot_encoders


def kernel_based(X: np.ndarray,
                 Y: np.ndarray,
                 Z: Optional[np.ndarray] = None,
                 kernel: Callable[[np.ndarray], np.ndarray] = apply_rbf_kernel_with_adaptive_precision,
                 scale_data: bool = True,
                 use_bootstrap: bool = True,
                 bootstrap_num_runs: int = 20,
                 bootstrap_num_samples_per_run: int = 2000,
                 bootstrap_n_jobs: Optional[int] = None,
                 p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float] = quantile_based_fwer) \
        -> float:
    """Prepares the data and uses kernel (conditional) independence test. The independence test estimates a p-value
    for the null hypothesis that X and Y are independent (given Z). Depending whether Z is given, a conditional or
    pairwise independence test is performed.

    If Z is given: Using KCI as conditional independence test.
    If Z is not given: Using HSIC as pairwise independence test.

    Note:
    - The data can be multivariate, i.e. the given input matrices can have multiple columns.
    - Categorical data need to be represented as strings.
    - It is possible to apply a different kernel to each column in the matrices. For instance, a RBF kernel for the
      first dimension in X and a delta kernel for the second.

    Based on the work:
    - Conditional: K. Zhang, J. Peters, D. Janzing, B. Schölkopf. *Kernel-based Conditional Independence Test and Application in Causal Discovery*. UAI'11, Pages 804–813, 2011.
    - Pairwise: A. Gretton, K. Fukumizu, C.-H. Teo, L. Song, B. Schölkopf, A. Smola. *A Kernel Statistical Test of Independence*. NIPS 21, 2007.

    :param X: Data matrix for observations from X.
    :param Y: Data matrix for observations from Y.
    :param Z: Optional data matrix for observations from Z. This is the conditional variable.
    :param kernel: A kernel for estimating the pairwise similarities between samples. The expected input is a n x d
                   numpy array and the output is expected to be a n x n numpy array. By default, the RBF kernel is used.
    :param scale_data: If set to True, the data will be standardized. If set to False, the data is taken as it is.
                       Standardizing the data helps in identifying weak dependencies. If one is only interested in
                       stronger ones, consider setting this to False.
    :param use_bootstrap: If True, the independence tests are performed on multiple subsets of the data and the final
                          p-value is constructed based on the provided p_value_adjust_func function.
    :param bootstrap_num_runs: Number of bootstrap runs (only relevant if use_bootstrap is True).
    :param bootstrap_num_samples_per_run: Number of samples used in a bootstrap run (only relevant if use_bootstrap is
                                          True).
    :param bootstrap_n_jobs: Number of parallel jobs for the boostrap runs.
    :param p_value_adjust_func: A callable that expects a numpy array of multiple p-values and returns one p-value. This
                                is typically used a family wise error rate control method.
    :return: The p-value for the null hypothesis that X and Y are independent (given Z).
    """
    bootstrap_n_jobs = config.default_n_jobs if bootstrap_n_jobs is None else bootstrap_n_jobs

    def evaluate_kernel_test_on_samples(X: np.ndarray,
                                        Y: np.ndarray,
                                        Z: np.ndarray,
                                        parallel_random_seed: int) -> float:
        set_random_seed(parallel_random_seed)

        try:
            if Z is None:
                return _hsic(X, Y, kernel=kernel, scale_data=scale_data)
            else:
                return _kci(X, Y, Z, kernel=kernel, scale_data=scale_data)
        except LinAlgError:
            # TODO: This is a temporary workaround.
            #       Under some circumstances, the KCI test throws a "numpy.linalg.LinAlgError: SVD did not converge"
            #       error, depending on the data samples. This is related to the utilized algorithms by numpy for SVD.
            #       There is actually a robust version for SVD, but it is not included in numpy.
            #       This can either be addressed by some augmenting the data, using a different SVD implementation or
            #       wait until numpy updates the used algorithm.
            return np.nan

    if use_bootstrap and X.shape[0] > bootstrap_num_samples_per_run:
        random_indices = [np.random.choice(X.shape[0], min(X.shape[0], bootstrap_num_samples_per_run), replace=False)
                          for run in range(bootstrap_num_runs)]

        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(random_indices))
        p_values = Parallel(n_jobs=bootstrap_n_jobs)(
            delayed(evaluate_kernel_test_on_samples)(X[indices],
                                                     Y[indices],
                                                     Z[indices] if Z is not None else None,
                                                     random_seed)
            for indices, random_seed in zip(random_indices, random_seeds))

        return p_value_adjust_func(p_values)
    else:
        return evaluate_kernel_test_on_samples(X, Y, Z, np.random.randint(np.iinfo(np.int32).max, size=1)[0])


def approx_kernel_based(X: np.ndarray,
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
                        p_value_adjust_func:
                        Callable[[Union[np.ndarray, List[float]]], float] = quantile_based_fwer) -> float:
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
    :param bootstrap_n_jobs: Number of parallel jobs for the boostrap runs.
    :param p_value_adjust_func: A callable that expects a numpy array of multiple p-values and returns one p-value. This
                                is typically used a family wise error rate control method.
    :return: The p-value for the null hypothesis that X and Y are independent (given Z).
    """
    bootstrap_n_jobs = config.default_n_jobs if bootstrap_n_jobs is None else bootstrap_n_jobs

    if not use_bootstrap:
        bootstrap_num_runs = 1
        bootstrap_num_samples = float('inf')
        bootstrap_n_jobs = 1

    if Z is None:
        return _rit(X,
                    Y,
                    num_permutations=num_permutations,
                    num_random_features_X=num_random_features_X,
                    num_random_features_Y=num_random_features_Y,
                    num_runs=bootstrap_num_runs,
                    num_max_samples_per_run=bootstrap_num_samples,
                    approx_kernel=approx_kernel,
                    scale_data=scale_data,
                    n_jobs=bootstrap_n_jobs,
                    p_value_adjust_func=p_value_adjust_func)
    else:
        return _rcit(X,
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
                     p_value_adjust_func=p_value_adjust_func)


def _kci(X: np.ndarray,
         Y: np.ndarray,
         Z: np.ndarray,
         kernel: Callable[[np.ndarray], np.ndarray],
         scale_data: bool,
         regularization_param: float = 10 ** -3) -> float:
    """
    Tests the null hypothesis that X and Y are independent given Z using the kernel conditional independence test.

    This is a corrected reimplementation of the KCI method in the CondIndTests R-package. Authors of the original R
    package: Christina Heinze-Deml, Jonas Peters, Asbjoern Marco Sinius Munk

    :return: The p-value for the null hypothesis that X and Y are independent given Z.
    """
    X, Y, Z = _convert_to_numeric(*shape_into_2d(X, Y, Z))

    if X.shape[0] != Y.shape[0] != Z.shape[0]:
        raise RuntimeError('All variables need to have the same number of samples!')

    n = X.shape[0]

    if scale_data:
        X = scale(X)
        Y = scale(Y)
        Z = scale(Z)

    k_x = kernel(X)
    k_y = kernel(Y)
    k_z = kernel(Z)

    k_xz = k_x * k_z

    k_xz = _fast_centering(k_xz)
    k_y = _fast_centering(k_y)
    k_z = _fast_centering(k_z)

    r_z = np.eye(n) - k_z @ pinv(k_z + regularization_param * np.eye(n))

    k_xz_z = r_z @ k_xz @ r_z.T
    k_y_z = r_z @ k_y @ r_z.T

    # Not dividing by n, seeing that the expectation and variance are also not divided by n and n**2, respectively.
    statistic = np.sum(k_xz_z * k_y_z.T)

    # Taking the sum, because due to numerical issues, the matrices might not be symmetric.
    eigen_vec_k_xz_z, eigen_val_k_xz_z, _ = svd((k_xz_z + k_xz_z.T) / 2)
    eigen_vec_k_y_z, eigen_val_k_y_z, _ = svd((k_y_z + k_y_z.T) / 2)

    # Filter out eigenvalues that are too small.
    eigen_val_k_xz_z, eigen_vec_k_xz_z = _filter_out_small_eigen_values_and_vectors(eigen_val_k_xz_z, eigen_vec_k_xz_z)
    eigen_val_k_y_z, eigen_vec_k_y_z = _filter_out_small_eigen_values_and_vectors(eigen_val_k_y_z, eigen_vec_k_y_z)

    if len(eigen_val_k_xz_z) == 1:
        empirical_kernel_map_xz_z = eigen_vec_k_xz_z * np.sqrt(eigen_val_k_xz_z)
    else:
        empirical_kernel_map_xz_z = eigen_vec_k_xz_z @ (np.eye(len(eigen_val_k_xz_z))
                                                        * np.sqrt(eigen_val_k_xz_z)).T

    empirical_kernel_map_xz_z = empirical_kernel_map_xz_z.squeeze()
    empirical_kernel_map_xz_z = empirical_kernel_map_xz_z.reshape(empirical_kernel_map_xz_z.shape[0], -1)

    if len(eigen_val_k_y_z) == 1:
        empirical_kernel_map_y_z = eigen_vec_k_y_z * np.sqrt(eigen_val_k_y_z)
    else:
        empirical_kernel_map_y_z = eigen_vec_k_y_z @ (np.eye(len(eigen_val_k_y_z)) * np.sqrt(eigen_val_k_y_z)).T

    empirical_kernel_map_y_z = empirical_kernel_map_y_z.squeeze()
    empirical_kernel_map_y_z = empirical_kernel_map_y_z.reshape(empirical_kernel_map_y_z.shape[0], -1)

    num_eigen_vec_xz_z = empirical_kernel_map_xz_z.shape[1]
    num_eigen_vec_y_z = empirical_kernel_map_y_z.shape[1]

    size_w = num_eigen_vec_xz_z * num_eigen_vec_y_z

    w = (empirical_kernel_map_y_z[:, None]
         * empirical_kernel_map_xz_z[..., None]).reshape(empirical_kernel_map_y_z.shape[0], -1)

    if size_w > n:
        ww_prod = w @ w.T
    else:
        ww_prod = w.T @ w

    return _estimate_p_value(ww_prod, statistic)


def _fast_centering(k: np.ndarray) -> np.ndarray:
    """Compute centered kernel matrix in time O(n^2).

    The centered kernel matrix is defined as K_c = H @ K @ H, with
    H = identity - 1/ n * ones(n,n). Computing H @ K @ H via matrix multiplication scales with n^3. The
    implementation circumvents this and runs in time n^2.
    :param k: original kernel matrix of size nxn
    :return: centered kernel matrix of size nxn
    """
    n = len(k)
    k_c = (k - 1 / n * np.outer(np.ones(n), np.sum(k, axis=0))
           - 1 / n * np.outer(np.sum(k, axis=1), np.ones(n))
           + 1 / n ** 2 * np.sum(k) * np.ones((n, n)))
    return k_c


def _hsic(X: np.ndarray,
          Y: np.ndarray,
          kernel: Callable[[np.ndarray], np.ndarray],
          scale_data: bool,
          cut_off_value: float = EPS) -> float:
    """
    Estimates the Hilbert-Schmidt Independence Criterion score for a pairwise independence test between variables X
    and Y.

    This is a reimplementation from the original Matlab code provided by the authors.

    :return: The p-value for the null hypothesis that X and Y are independent.
    """
    X, Y = _convert_to_numeric(*shape_into_2d(X, Y))

    if X.shape[0] != Y.shape[0]:
        raise RuntimeError('All variables need to have the same number of samples!')

    if X.shape[0] < 6:
        raise RuntimeError('At least 6 samples are required for the HSIC independence test. Only %d were given.'
                           % X.shape[0])

    n = X.shape[0]

    if scale_data:
        X = scale(X)
        Y = scale(Y)

    k_mat = kernel(X)
    l_mat = kernel(Y)

    k_c = _fast_centering(k_mat)
    l_c = _fast_centering(l_mat)

    #  Test statistic is given as np.trace(K @ H @ L @ H) / n. Below computes without matrix products.
    test_statistic = 1 / n * (np.sum(k_mat * l_mat) - 2 / n * np.sum(k_mat, axis=0) @ np.sum(l_mat, axis=1) +
                              1 / n ** 2 * np.sum(k_mat) * np.sum(l_mat))

    var_hsic = (k_c * l_c) ** 2
    var_hsic = (np.sum(var_hsic) - np.trace(var_hsic)) / n / (n - 1)
    var_hsic = var_hsic * 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    k_mat = k_mat - np.diag(np.diag(k_mat))
    l_mat = l_mat - np.diag(np.diag(l_mat))

    bone = np.ones((n, 1), dtype=float)
    mu_x = (bone.T @ k_mat @ bone) / n / (n - 1)
    mu_y = (bone.T @ l_mat @ bone) / n / (n - 1)

    m_hsic = (1 + mu_x * mu_y - mu_x - mu_y) / n

    var_hsic = max(var_hsic.squeeze(), cut_off_value)
    m_hsic = max(m_hsic.squeeze(), cut_off_value)
    if test_statistic <= cut_off_value:
        test_statistic = 0

    al = m_hsic ** 2 / var_hsic
    bet = var_hsic * n / m_hsic

    p_value = 1 - gamma.cdf(test_statistic, al, scale=bet)

    return p_value


def _filter_out_small_eigen_values_and_vectors(eigen_values: np.ndarray,
                                               eigen_vectors: np.ndarray,
                                               relative_tolerance: float = (10 ** -5)) \
        -> Tuple[np.ndarray, np.ndarray]:
    filtered_indices_xz_z = np.where(eigen_values[eigen_values > max(eigen_values) * relative_tolerance])

    return eigen_values[filtered_indices_xz_z], eigen_vectors[:, filtered_indices_xz_z]


def _estimate_p_value(ww_prod: np.ndarray, statistic: np.ndarray) -> float:
    # Dividing by n not required since we do not divide the test statistical_tools by n.
    mean_approx = np.trace(ww_prod)
    variance_approx = 2 * np.trace(ww_prod @ ww_prod)

    alpha_approx = mean_approx ** 2 / variance_approx
    beta_approx = variance_approx / mean_approx

    return 1 - gamma.cdf(statistic, alpha_approx, scale=beta_approx)


def _rit(X: np.ndarray, Y: np.ndarray,
         num_random_features_X: int,
         num_random_features_Y: int,
         num_permutations: int,
         num_runs: int,
         num_max_samples_per_run: int,
         approx_kernel: Callable[[np.ndarray, int], np.ndarray],
         scale_data: bool,
         n_jobs: Optional[int],
         p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float]) -> float:
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
            permutation_results_of_statistic.append(_estimate_rit_statistic(
                random_features_x[np.random.choice(random_features_x.shape[0],
                                                   random_features_x.shape[0], replace=False)], random_features_y))

        return 1 - (np.sum(_estimate_rit_statistic(random_features_x, random_features_y)
                           > permutation_results_of_statistic) / len(permutation_results_of_statistic))

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=num_runs)
    p_values = Parallel(n_jobs=n_jobs)(delayed(evaluate_rit_on_samples)(random_seeds[i]) for i in range(num_runs))

    return p_value_adjust_func(p_values)


def _rcit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
          num_random_features_X: int,
          num_random_features_Y: int,
          num_random_features_Z: int,
          num_permutations: int,
          num_runs: int,
          num_max_samples_per_run: int,
          approx_kernel: Callable[[np.ndarray, int], np.ndarray],
          scale_data: bool,
          n_jobs: Optional[int],
          p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float]) -> float:
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
            scipy.linalg.cho_factor(cov_zz + np.eye(cov_zz.shape[0]) * 10 ** -10, lower=True),
            np.eye(cov_zz.shape[0]))
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
            permutation_results_of_statistic.append(_estimate_rit_statistic(
                residual_x[np.random.choice(residual_x.shape[0],
                                            residual_x.shape[0], replace=False)], residual_y))

        return 1 - (np.sum(_estimate_rit_statistic(residual_x, residual_y) > permutation_results_of_statistic)
                    / len(permutation_results_of_statistic))

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=num_runs)
    p_values = Parallel(n_jobs=n_jobs)(delayed(parallel_job)(random_seeds[i]) for i in range(num_runs))

    return p_value_adjust_func(p_values)


def _estimate_rit_statistic(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X.shape[0] * np.sum(_estimate_column_wise_covariances(X, Y) ** 2)


def _estimate_column_wise_covariances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.cov(X, Y, rowvar=False)[:X.shape[1], -Y.shape[1]:]


def _convert_to_numeric(*args) -> List[np.ndarray]:
    return [apply_one_hot_encoding(X, fit_one_hot_encoders(X)) for X in args]
