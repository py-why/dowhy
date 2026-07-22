import random

import numpy as np
import pytest
from flaky import flaky
from pytest import approx

from dowhy.gcm.shapley import (
    ShapleyApproximationMethods,
    ShapleyConfig,
    _weighted_least_squares_coefficients,
    estimate_shapley_values,
)
from dowhy.gcm.stats import permute_features
from dowhy.gcm.util.general import means_difference


@pytest.fixture
def preserve_random_generator_state():
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(random_state)


def test_given_few_features_when_estimate_shapley_values_with_auto_approx_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(4)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.AUTO, n_jobs=1),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_many_features_when_estimate_shapley_values_with_auto_approx_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.AUTO, n_jobs=1),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_many_features_when_estimate_shapley_values_exact_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT, n_jobs=1),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_many_features_when_estimate_shapley_values_exact_fast_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT_FAST, n_jobs=1),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_many_features_when_estimate_shapley_values_with_subset_sampling_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.SUBSET_SAMPLING),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_many_features_when_estimate_shapley_values_permutation_based_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.PERMUTATION),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_many_features_when_estimate_shapley_values_with_early_stopping_then_returns_correct_linear_shapley_values():
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_values = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
        X.shape[1],
        ShapleyConfig(approximation_method=ShapleyApproximationMethods.EARLY_STOPPING),
    )

    assert coefficients * (X[0, :] - np.mean(X, axis=0)) == approx(shapley_values, abs=0.001)


def test_given_specific_random_seed_when_estimate_shapley_values_with_subset_sampling_then_returns_deterministic_result(
    preserve_random_generator_state,
):
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.SUBSET_SAMPLING)
    assert estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    ) != approx(
        estimate_shapley_values(
            lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
            X.shape[1],
            shapley_config,
        ),
        abs=0,
    )

    np.random.seed(0)
    shapley_values_1 = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    )
    np.random.seed(0)
    shapley_values_2 = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    )

    assert shapley_values_1 == approx(shapley_values_2, abs=0)


@flaky(max_runs=2)
def test_given_specific_random_seed_when_estimate_shapley_values_permutation_based_then_returns_deterministic_result(
    preserve_random_generator_state,
):
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.PERMUTATION)
    assert estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    ) != approx(
        estimate_shapley_values(
            lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
            X.shape[1],
            shapley_config,
        ),
        abs=0,
    )

    np.random.seed(0)
    shapley_values_1 = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    )
    np.random.seed(0)
    shapley_values_2 = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    )

    assert shapley_values_1 == approx(shapley_values_2, abs=0)


def test_given_specific_random_seed_when_estimate_shapley_values_with_early_stopping_then_returns_deterministic_result(
    preserve_random_generator_state,
):
    X, coefficients = _generate_data(15)

    def model(x):
        return np.sum(coefficients * x, axis=1)

    shapley_config = ShapleyConfig(approximation_method=ShapleyApproximationMethods.EARLY_STOPPING)
    assert estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    ) != approx(
        estimate_shapley_values(
            lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model),
            X.shape[1],
            shapley_config,
        ),
        abs=0,
    )

    np.random.seed(0)
    shapley_values_1 = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    )
    np.random.seed(0)
    shapley_values_2 = estimate_shapley_values(
        lambda subset: _set_function_for_aggregated_feature_attribution(subset, X, model), X.shape[1], shapley_config
    )

    assert shapley_values_1 == approx(shapley_values_2, abs=0)


def _generate_data(num_vars):
    return np.random.normal(0, 1, (1000, num_vars)), np.random.choice(20, num_vars) - 10


def _set_function_for_aggregated_feature_attribution(subset, X, model, evaluate_only_one_sample=True):
    tmp = permute_features(X, np.arange(0, X.shape[1])[subset == 0], False)
    tmp[:, subset == 1] = X[0, subset == 1]

    if evaluate_only_one_sample:
        return means_difference(model(tmp), X[0])
    else:
        return np.array([means_difference(model(tmp), x) for x in X])


def test_given_single_and_multi_output_targets_when_solving_weighted_least_squares_then_returns_expected_coefficients():
    num_subsets = 200
    num_players = 6
    rng = np.random.RandomState(0)

    all_subsets = (rng.uniform(size=(num_subsets, num_players)) < 0.5).astype(float)
    all_subsets[0] = 0.0
    all_subsets[1] = 1.0
    # The full and empty subsets act as equality constraints and carry an extreme weight.
    weights = np.ones(num_subsets)
    weights[0] = weights[1] = 10**20

    single_output_coefficients = rng.normal(size=num_players)
    second_output_coefficients = rng.normal(size=num_players)
    intercept = 3.0

    estimated = _weighted_least_squares_coefficients(
        all_subsets, all_subsets @ single_output_coefficients + intercept, weights
    )
    assert estimated.shape == (num_players,)
    assert estimated == approx(single_output_coefficients, abs=1e-8)

    multi_output_targets = np.column_stack(
        [
            all_subsets @ single_output_coefficients + intercept,
            all_subsets @ second_output_coefficients - intercept,
        ]
    )
    estimated_multi_output = _weighted_least_squares_coefficients(all_subsets, multi_output_targets, weights)
    # Matches the (n_outputs, n_features) layout of LinearRegression.coef_
    assert estimated_multi_output.shape == (2, num_players)
    assert estimated_multi_output[0] == approx(single_output_coefficients, abs=1e-8)
    assert estimated_multi_output[1] == approx(second_output_coefficients, abs=1e-8)
