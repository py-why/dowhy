""" Regression based (conditional) independence test. Testing independence via regression, i.e. if a variable has
information about another variable, then they are dependent.
"""
from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import scale

from dowhy.gcm.stats import estimate_ftest_pvalue, quantile_based_fwer
from dowhy.gcm.util.general import apply_one_hot_encoding, fit_one_hot_encoders, shape_into_2d


def regression_based(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    num_components_all_inputs: int = 40,
    num_runs: int = 20,
    p_value_adjust_func: Callable[[Union[np.ndarray, List[float]]], float] = quantile_based_fwer,
    f_test_samples_ratio: Optional[float] = 0.3,
    max_samples_per_run: int = 10000,
) -> float:
    """The main idea is that if X and Y are dependent, then X should help in predicting Y. If there is no dependency,
    then X should not help. When Z is given, the idea remains the same, but here X and Y are conditionally independent
    given Z if X does not help in predicting Y when knowing Z. This is, X has not additional information about Y given
    Z. In the pairwise case (Z is not given), the performances (in terms of squared error) between predicting Y based
    on X and predicting Y by returning its mean (the best estimator without any inputs) are compared. Note that
    categorical inputs are transformed via the sklearn one-hot-encoder.

    Here, we use the :class:`sklearn.kernel_approximation.Nystroem` approach to approximate a kernel map of the inputs that serves as new input features.
    These new features allow to model complex non-linear relationships. In case of categorical data, we first apply an
    one-hot-encoding and then map it into the kernel feature space. Afterwards, we use linear regression as a prediction
    model based on the non-linear input features. The idea is then the same as in Granger causality, where we
    apply a f-test to see if the additional input features significantly help in predicting the target or not.

    Note: As compared to :func:`~dowhy.gcm.kernel_based`, this method is quite fast and provides reasonably well results. However, there
    are types of dependencies that this test cannot detect. For instance, if X determines the variance of Y, then this
    cannot be captured. For these more complex dependencies, consider using the :func:`~dowhy.gcm.kernel_based` independence test instead.

    This test is motivated by Granger causality, the approx_kernel_based test and the following paper:
        K Chalupka, P Perona, F. Eberhardt. *Fast Conditional Independence Test for Vector Variables with Large Sample Sizes*. arXiv:1804.02747, 2018.

    :param X: Input data for X.
    :param Y: Input data for Y.
    :param Z: Input data for Z. The set of variables to (optionally) condition on.
    :param num_components_all_inputs: Number of kernel features when combining X and Z. If Z is not given, it will be
                                      replaced with an empty array. If Z is given, half of the number is used to
                                      generate features for Z.
    :param num_runs: Number of runs. This equals the number of estimated p-values, which get adjusted by the
                     p_value_adjust_func.
    :param p_value_adjust_func: A callable that expects a numpy array of multiple p-values and returns one p-value. This
                                is typically used a family wise error rate control method.
    :param f_test_samples_ratio: Ratio for splitting the data into test and training data sets for calculating
                                 the f-statistic. A ratio of 0.3 means that 30% of the samples are used for the f-test (test samples) and 70% are
                                 used for training the prediction model (training samples). If set to None, training and test data set are the same,
                                 which could help in settings where only a few samples are available.
    :param max_samples_per_run: Maximum number of samples used per run.
    :return: The p-value for the null hypothesis that X and Y are independent given Z. If Z is not given,
             then for the hypothesis that X and Y are independent.
    """
    if num_components_all_inputs < 2:
        raise ValueError(
            "Need at least two components for all inputs, but only %d were given!" % num_components_all_inputs
        )

    X, Y = shape_into_2d(X, Y)

    X = scale(apply_one_hot_encoding(X, fit_one_hot_encoders(X)))
    Y = scale(apply_one_hot_encoding(Y, fit_one_hot_encoders(Y)))

    if Z is not None:
        Z = shape_into_2d(Z)
        Z = scale(apply_one_hot_encoding(Z, fit_one_hot_encoders(Z)))

    org_X = X
    org_Y = Y
    org_Z = Z

    all_p_values = []
    for _ in range(num_runs):
        if X.shape[0] > max_samples_per_run:
            random_indices = np.random.choice(X.shape[0], max_samples_per_run, replace=False)
            X = org_X[random_indices]
            Y = org_Y[random_indices]
            if org_Z is not None:
                Z = org_Z[random_indices]

        if Z is not None:
            all_inputs = Nystroem(n_components=num_components_all_inputs).fit_transform(np.column_stack([X, Z]))
            input_Z = Nystroem(n_components=num_components_all_inputs // 2).fit_transform(Z)
        else:
            all_inputs = Nystroem(n_components=num_components_all_inputs).fit_transform(X)
            input_Z = np.array([]).reshape(1, -1)

        if f_test_samples_ratio is not None:
            num_f_test_samples = int(all_inputs.shape[0] * f_test_samples_ratio)
            indices_random_order = np.random.choice(all_inputs.shape[0], all_inputs.shape[0], replace=False)
            training_indices = indices_random_order[num_f_test_samples:]
            test_indices = indices_random_order[:num_f_test_samples]
        else:
            training_indices = np.arange(0, all_inputs.shape[0])
            test_indices = training_indices

        all_p_values.append(
            estimate_ftest_pvalue(
                all_inputs[training_indices],
                input_Z[training_indices],
                Y[training_indices],
                all_inputs[test_indices],
                input_Z[test_indices],
                Y[test_indices],
            )
        )

    return p_value_adjust_func(all_p_values)
