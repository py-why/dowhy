"""Regression based (conditional) independence test. Testing independence via regression, i.e. if a variable has
information about another variable, then they are dependent.
"""

from typing import Callable, List, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

import dowhy.gcm.config as config
from dowhy.gcm.stats import estimate_ftest_pvalue, merge_p_values_average
from dowhy.gcm.util.general import auto_apply_encoders, auto_fit_encoders, set_random_seed, shape_into_2d


def regression_based(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    max_num_components_all_inputs: int = 40,
    k_folds: int = 3,
    p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float] = merge_p_values_average,
    max_samples_per_fold: int = -1,
    n_jobs: Optional[int] = None,
) -> float:
    """The main idea is that if X and Y are dependent, then X should help in predicting Y. If there is no dependency,
    then X should not help. When Z is given, the idea remains the same, but here X and Y are conditionally independent
    given Z if X does not help in predicting Y when knowing Z. This is, X has not additional information about Y given
    Z. In the pairwise case (Z is not given), the performances (in terms of squared error) between predicting Y based
    on X and predicting Y by returning its mean (the best estimator without any inputs) are compared. Note that
    categorical inputs are transformed via encoders.

    Here, we use the :class:`sklearn.kernel_approximation.Nystroem` approach to approximate a kernel map of the inputs that serves as new input features.
    These new features allow to model complex non-linear relationships. In case of categorical data, we first apply an
    encoding and then map it into the kernel feature space. Afterwards, we use linear regression as a prediction
    model based on the non-linear input features. The idea is then to apply a f-test to see if the additional input
    features significantly help in predicting the target or not.

    This test is motivated by Granger causality, the approx_kernel_based test and the following paper:
        K Chalupka, P Perona, F. Eberhardt. *Fast Conditional Independence Test for Vector Variables with Large Sample Sizes*. arXiv:1804.02747, 2018.

    :param X: Input data for X.
    :param Y: Input data for Y.
    :param Z: Input data for Z. The set of variables to (optionally) condition on.
    :param max_num_components_all_inputs: Maximum number of kernel features when combining X and Z. If Z is not given, it will be
                                          replaced with an empty array. If Z is given, half of the number is used to
                                          generate features for Z. Note that the actual number of components is 1/10 of the
                                          number of samples, but at most max_num_components_all_inputs.
    :param num_target_components_factor: The factor indicates how many components are used for the target variable. This is,
                                         num_target_components_factor * dimension of the target many components.
    :param k_folds: Number of folds for training and test set. This equals the number of estimated p-values, which get adjusted by the
                    p_value_adjust_func.
    :param p_value_adjust_func: A callable that expects a numpy array of multiple p-values and returns one p-value. This
                                is typically used a family wise error rate control method.
    :param max_samples_per_fold: Maximum number of samples used per fold for training and testing. If -1, it uses all data.
    :param n_jobs: Number of parallel jobs for the evaluation of the folds.
    :return: The p-value for the null hypothesis that X and Y are independent given Z. If Z is not given,
             then for the hypothesis that X and Y are independent.
    """
    n_jobs = config.default_n_jobs if n_jobs is None else n_jobs

    if max_num_components_all_inputs < 2:
        raise ValueError(
            "Need at least two components for all inputs, but only %d were given!" % max_num_components_all_inputs
        )

    X, Y = shape_into_2d(X, Y)

    if max_samples_per_fold == -1:
        max_samples_per_fold = X.shape[0]

    # Take the lower dimensional variable as target.
    if X.shape[1] < Y.shape[1]:
        X, Y = Y, X

    X = scale(auto_apply_encoders(X, auto_fit_encoders(X), Y))
    Y = scale(auto_apply_encoders(Y, auto_fit_encoders(Y)))

    if Z is not None:
        Z = shape_into_2d(Z)
        Z = scale(auto_apply_encoders(Z, auto_fit_encoders(Z)))

    def estimate_p_value(training_indices, test_indices, parallel_random_seed: int) -> float:
        set_random_seed(parallel_random_seed)
        adaptive_num_components = max(min(max_num_components_all_inputs, training_indices.shape[0] // 3), 2)

        if X.shape[0] > max_samples_per_fold:
            training_indices = training_indices[:max_samples_per_fold]
            test_indices = test_indices[:max_samples_per_fold]

        if Z is not None:
            transformer_input_all = Nystroem(n_components=adaptive_num_components)
            transformer_input_Z = Nystroem(n_components=adaptive_num_components // 2)
            XZ = np.column_stack([X, Z])

            training_all_inputs = transformer_input_all.fit_transform(XZ[training_indices])
            training_Z = transformer_input_Z.fit_transform(Z[training_indices])

            test_all_inputs = transformer_input_all.transform(XZ[test_indices])
            test_Z = transformer_input_Z.transform(Z[test_indices])
        else:
            transformer_input_all = Nystroem(n_components=adaptive_num_components)

            training_all_inputs = transformer_input_all.fit_transform(X[training_indices])
            test_all_inputs = transformer_input_all.transform(X[test_indices])

            training_Z = test_Z = np.array([]).reshape(1, -1)

        training_target = Y[training_indices]
        test_target = Y[test_indices]

        return estimate_ftest_pvalue(
            training_all_inputs,
            training_Z,
            training_target,
            test_all_inputs,
            test_Z,
            test_target,
        )

    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=k_folds)
    p_values = Parallel(n_jobs=n_jobs)(
        delayed(estimate_p_value)(training_indices, test_indices, int(random_seed))
        for (training_indices, test_indices), random_seed in zip(
            KFold(n_splits=k_folds, shuffle=True).split(X), random_seeds
        )
    )

    return p_value_adjust_func(p_values)
