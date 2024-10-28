from typing import Callable, List, Optional, Union

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

from dowhy.gcm.constant import EPS
from dowhy.gcm.util.general import shape_into_2d


def merge_p_values_average(p_values: Union[np.ndarray, List[float]], randomization: bool = False) -> float:
    """A statistically sound method to merge multiple potentially dependent p-values into one. This is a statistically
    improved (i.e., more powerful) version of the "twice the average" rule, following Theorem 5.3
    (second equation, F_UA) in

    M. Gasparini, R. Wang, and A. Ramdas, *Combining exchangeable p-values*, arXiv 2404.03484, 2024

    Note, if randomization is False, we have u = 1 here. Generally, randomization requires fewer assumptions but leads
    to non-deterministic behavior.

    :param p_values: A list or array of p-values.
    :param randomization: If True, u is taken uniformly randomly from [0, 1] (non-deterministic). If False, u is set
    to 1 (deterministic). Randomization is generally more powerful but provides non-deterministic results.
    :return: A single p-value based on the given p-values.
    """
    if len(p_values) == 0:
        raise ValueError("Given list of p-values is empty!")
    if len(p_values) == 1:
        return p_values[0]

    if np.all(np.isnan(p_values)):
        return float(np.nan)

    if randomization:
        u = float(np.random.uniform(0, 1))
    else:
        u = 1

    p_values = np.array(p_values)
    p_values = p_values[~np.isnan(p_values)]
    p_values.sort()

    K = len(p_values)

    return min(
        1.0, float(np.min([2 * np.mean(p_values[:m]) / (2 - (K * u / m)) for m in range(1, K + 1) if (K * u / m) < 2]))
    )


def merge_p_values_quantile(
    p_values: Union[np.ndarray, List[float]], p_values_scaling: Optional[np.ndarray] = None, quantile: float = 0.5
) -> float:
    """Applies a quantile based approach to merge multiple potentially dependent p-values to one. This is based on the
    approach described in:

    Meinshausen, N., Meier, L. and Buehlmann, P., *p-values for high-dimensional regression*,
    J. Amer. Statist. Assoc.104 1671–1681, 2009

    :param p_values: A list or array of p-values.
    :param p_values_scaling: An optional list of scaling factors for each p-value.
    :param quantile: The quantile used for the p-value adjustment. By default, this is the median (0.5).
    :return: The p-value that lies on the quantile threshold. Note that this is the quantile based on scaled values
             p_values / quantile.
    """

    if quantile <= 0 or abs(quantile - 1) >= 1:
        raise ValueError("The given quantile is %f, but it needs to be on (0, 1]!" % quantile)

    p_values = np.array(p_values)
    if p_values_scaling is None:
        p_values_scaling = np.ones(p_values.shape[0])

    if p_values.shape != p_values_scaling.shape:
        raise ValueError("The p-value scaling array needs to have the same dimension as the given p-values.")

    p_values_scaling = p_values_scaling[~np.isnan(p_values)]
    p_values = p_values[~np.isnan(p_values)]

    p_values = p_values * p_values_scaling
    p_values[p_values > 1] = 1.0

    if p_values.shape[0] == 1:
        return float(p_values[0])
    else:
        return float(min(1.0, np.quantile(p_values / quantile, quantile)))


def merge_p_values_fdr(p_values: Union[np.ndarray, List[float]], fdr_method: str = "fdr_bh") -> float:
    """Merges p-values to represent the global null hypothesis that all hypotheses represented by the p-values are true.

    Here, we first adjust the given p-values based on the provided false discovery rate (FDR) control method, and then
    return the minimum.

    :param p_values: A list or array of p-values.
    :param fdr_method: The false discovery rate control method. For various options, please refer to
                       `this page <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_.
    :return: The minimum p-value after adjusting based on the given FDR method.
    """
    if len(p_values) == 0:
        raise ValueError("Given list of p-values is empty!")

    p_values = np.array(p_values)

    if np.all(np.isnan(p_values)):
        return float(np.nan)

    p_values = p_values[~np.isnan(p_values)]

    # Note: The alpha level doesn't matter here.
    multipletests_result = multipletests(p_values, 0.05, method=fdr_method)
    return min(multipletests_result[1])


def marginal_expectation(
    prediction_method: Callable[[np.ndarray], np.ndarray],
    feature_samples: np.ndarray,
    baseline_samples: np.ndarray,
    baseline_feature_indices: List[int],
    return_averaged_results: bool = True,
    feature_perturbation: str = "randomize_columns_jointly",
    max_batch_size: int = -1,
) -> np.ndarray:
    """Estimates the marginal expectation for samples in baseline_noise_samples when randomizing features that are not
    part of baseline_feature_indices. This is, this function estimates
        y^i = E[Y | do(x^i_s)] := \\int_x_s' E[Y | x^i_s, x_s'] p(x_s') d x_s',
    where x^i_s is the i-th sample from baseline_noise_samples, s denotes the baseline_feature_indices and
    x_s' ~ X_s' denotes the randomized features that are not in s. For an approximation of the integral, the given
    prediction_method is evaluated multiple times for the same x^i_s, but different x_s' ~ X_s'.

    :param prediction_method: Prediction method of interest. This should expect a numpy array as input for making
    predictions.
    :param feature_samples: Samples from the joint distribution. These are used for randomizing the features that are not in
                            baseline_feature_indices.
    :param baseline_samples: Samples for which the marginal expectation should be estimated.
    :param baseline_feature_indices: Column indices of the features in s. These values for these features are remain constant
                                     when estimating the expectation.
    :param return_averaged_results: If set to True, the expectation over all evaluated samples for the i-th
    baseline_noise_samples is returned. If set to False, all corresponding results for the i-th sample are returned.
    :param feature_perturbation: Type of feature permutation:
        'randomize_columns_independently': Each feature not in s is randomly permuted separately.
        'randomize_columns_jointly': All features not in s are jointly permuted. Note that this still represents an
        interventional distribution.
    :param max_batch_size: Maximum batch size for a estimating the predictions. This has a significant influence on the
    overall memory usage. If set to -1, all samples are used in one batch.
    :return: If return_averaged_results is False, a numpy array where the i-th entry belongs to the marginal expectation
    of x^i_s when randomizing the remaining features.
    If return_averaged_results is True, a two dimensional numpy array where the i-th entry contains all
    predictions for x^i_s when randomizing the remaining features.
    """
    feature_samples, baseline_samples = shape_into_2d(feature_samples, baseline_samples)

    batch_size = baseline_samples.shape[0] if max_batch_size == -1 else max_batch_size
    result = [np.nan] * baseline_samples.shape[0]

    # Make copy to avoid manipulating the original matrix.
    feature_samples = np.array(feature_samples)

    features_to_randomize = np.delete(np.arange(0, feature_samples.shape[1]), baseline_feature_indices)

    if feature_perturbation == "randomize_columns_independently":
        feature_samples = permute_features(feature_samples, features_to_randomize, False)
    elif feature_perturbation == "randomize_columns_jointly":
        feature_samples = permute_features(feature_samples, features_to_randomize, True)
    else:
        raise ValueError("Unknown argument %s as feature_perturbation type!" % feature_perturbation)

    # The given prediction method has to be evaluated multiple times on a large amount of different inputs. Typically,
    # the batch evaluation of a prediction model on multiple inputs at the same time is significantly faster
    # than evaluating it on single simples in a for-loop. To make use of this, we try to evaluate as many samples as
    # possible in one batch call of the prediction method. However, this also requires a lot of memory for many samples.
    # To overcome potential memory issues, multiple batch calls are performed, each with at most batch_size many
    # samples. The number of samples that are evaluated is normally
    # baseline_noise_samples.shape[0] * feature_samples.shape[0]. Here, we reduce it to
    # batch_size * feature_samples.shape[0]. If the batch_size would be set 1, then each baseline_noise_samples is
    # evaluated one by one in a for-loop.
    inputs = np.tile(feature_samples, (batch_size, 1))
    for offset in range(0, baseline_samples.shape[0], batch_size):
        # Each batch consist of at most batch_size * feature_samples.shape[0] many samples. If there are multiple
        # batches, the offset indicates the index of the current baseline_noise_samples that has not been evaluated yet.
        if offset + batch_size > baseline_samples.shape[0]:
            # If the batch size would be larger than the remaining amount of samples, it is reduced to only include the
            # remaining baseline_noise_samples.
            adjusted_batch_size = baseline_samples.shape[0] - offset
            inputs = inputs[: adjusted_batch_size * feature_samples.shape[0]]
        else:
            adjusted_batch_size = batch_size

        for index in range(adjusted_batch_size):
            # The inputs consist of batch_size many copies of feature_samples. Here, we set the columns of the features
            # in baseline_feature_indices to their respective values in baseline_noise_samples.
            inputs[
                index * feature_samples.shape[0] : (index + 1) * feature_samples.shape[0], baseline_feature_indices
            ] = baseline_samples[offset + index, baseline_feature_indices]

        # After creating the (potentially large) input data matrix, we can evaluate the prediction method.
        predictions = np.array(prediction_method(inputs))

        for index in range(adjusted_batch_size):
            # Here, offset + index now indicates the sample index in baseline_noise_samples.
            if return_averaged_results:
                # This would average all prediction results obtained for the 'offset + index'-th sample in
                # baseline_noise_samples. This is, y^(offset + index) = E[Y | do(x^(offset + index)_s)].
                result[offset + index] = np.mean(
                    predictions[index * feature_samples.shape[0] : (index + 1) * feature_samples.shape[0]], axis=0
                )
            else:
                # This would return all prediction results obtained for the 'offset + index'-th sample in
                # baseline_noise_samples, i.e. the results are not averaged.
                result[offset + index] = predictions[
                    index * feature_samples.shape[0] : (index + 1) * feature_samples.shape[0]
                ]

    return np.array(result)


def permute_features(
    feature_samples: np.ndarray, features_to_permute: Union[List[int], np.ndarray], randomize_features_jointly: bool
) -> np.ndarray:
    # Making copy to ensure that the original object is not modified.
    feature_samples = np.array(feature_samples)

    if randomize_features_jointly:
        # Permute samples jointly. This still represents an interventional distribution.
        feature_samples[:, features_to_permute] = feature_samples[
            np.random.choice(feature_samples.shape[0], feature_samples.shape[0], replace=False)
        ][:, features_to_permute]
    else:
        # Permute samples independently.
        for feature in features_to_permute:
            np.random.shuffle(feature_samples[:, feature])

    return feature_samples


def estimate_ftest_pvalue(
    X_training_a: np.ndarray,
    X_training_b: np.ndarray,
    Y_training: np.ndarray,
    X_test_a: np.ndarray,
    X_test_b: np.ndarray,
    Y_test: np.ndarray,
) -> float:
    """Estimates the p-value for the null hypothesis that the same regression error with less parameters can be
    achieved. This is, a linear model trained on a data set A with d number of features has the same performance
    (in terms of squared error) relative to the number of features as a model trained on a data set B with k number
    features, where k < d. Here, both data sets need to have the same target values. A small p-value would
    indicate that the model performances are significantly different.

    Note that all given test samples are utilized in the f-test.

    See https://en.wikipedia.org/wiki/F-test#Regression_problems for more details.

    :param X_training_a: Input training samples for model A.
    :param X_training_b: Input training samples for model B. These samples should have less features than samples in X_training_a.
    :param Y_training: Target training values.
    :param X_test_a: Test samples for model A.
    :param X_test_b: Test samples for model B.
    :param Y_test: Test values.
    :return: A p-value on [0, 1].
    """
    X_training_a, X_test_a = shape_into_2d(X_training_a, X_test_a)

    if X_training_b.size > 0:
        X_training_b, X_test_b = shape_into_2d(X_training_b, X_test_b)
    else:
        X_training_b = X_training_b.reshape(0, 0)
        X_test_b = X_test_b.reshape(0, 0)

    if X_training_a.shape[1] <= X_training_b.shape[1]:
        raise ValueError(
            "The data X_training_a should have more dimensions (model parameters) than the data " "X_training_b!"
        )

    ssr_a = np.sum((Y_test - LinearRegression().fit(X_training_a, Y_training).predict(X_test_a)) ** 2)

    if X_training_b.shape[1] > 0:
        ssr_b = np.sum((Y_test - LinearRegression().fit(X_training_b, Y_training).predict(X_test_b)) ** 2)
    else:
        ssr_b = np.sum((Y_test - np.mean(Y_training)) ** 2)

    dof_diff_1 = X_test_a.shape[1] - X_test_b.shape[1]  # p1 - p2
    dof_diff_2 = X_test_a.shape[0] - X_test_a.shape[1] - 1  # n - p2 (parameters include intercept)

    f_statistic = (ssr_b - ssr_a) / dof_diff_1 * dof_diff_2

    if ssr_a < EPS:
        ssr_a = 0
    if ssr_b < EPS:
        ssr_b = 0

    if ssr_a == 0 and ssr_b == 0:
        f_statistic = 0
    elif ssr_a != 0:
        f_statistic /= ssr_a

    return stats.f.sf(f_statistic, dof_diff_1, dof_diff_2)
