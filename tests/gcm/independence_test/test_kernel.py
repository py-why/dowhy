import random

import numpy as np
import pytest
from _pytest.python_api import approx
from flaky import flaky

from dowhy.gcm.independence_test import approx_kernel_based, kernel_based
from dowhy.gcm.independence_test.kernel import _fast_centering


@flaky(max_runs=5)
def test_given_continuous_conditionally_independent_data_when_perform_kernel_based_test_then_not_reject():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_given_continuous_conditionally_dependent_data_when_perform_kernel_based_test_then_reject():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert 0.05 > kernel_based(x, y, w)


@flaky(max_runs=5)
def test_given_categorical_conditionally_independent_data_when_perform_kernel_based_test_then_not_reject():
    x, y, z = _generate_categorical_data()

    assert kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_given_categorical_conditionally_dependent_data_when_perform_kernel_based_test_then_reject():
    x, y, z = _generate_categorical_data()

    assert kernel_based(x, z, y) < 0.05


@flaky(max_runs=2)
def test_given_random_seed_when_perform_conditional_kernel_based_test_then_return_deterministic_result(
    _preserve_random_generator_state,
):
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(
        x, z, y, bootstrap_num_samples_per_run=5, bootstrap_num_runs=2, p_value_adjust_func=np.mean
    ) != kernel_based(x, z, y, bootstrap_num_samples_per_run=5, bootstrap_num_runs=2, p_value_adjust_func=np.mean)

    np.random.seed(0)
    result_1 = kernel_based(x, z, y, bootstrap_num_samples_per_run=5, bootstrap_num_runs=2, p_value_adjust_func=np.mean)
    np.random.seed(0)
    result_2 = kernel_based(x, z, y, bootstrap_num_samples_per_run=5, bootstrap_num_runs=2, p_value_adjust_func=np.mean)

    assert result_1 == result_2


def test_given_too_few_samples_when_perform_kernel_based_test_then_raise_error():
    with pytest.raises(RuntimeError):
        kernel_based(np.array([1, 2, 3, 4]), np.array([1, 3, 2, 4]))


@flaky(max_runs=5)
def test_given_continuous_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.randn(1000, 1)
    y = np.exp(np.random.rand(1000, 1))

    assert kernel_based(x, y) > 0.05


@flaky(max_runs=5)
def test_given_continuous_dependent_data_when_perform_kernel_based_test_then_reject():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(x, y) < 0.05


@flaky(max_runs=5)
def test_given_categorical_independent_data_when_perform_kernel_based_test_then_not_reject():
    x = np.random.normal(0, 1, 1000)
    y = (np.random.choice(2, 1000) == 1).astype(str)

    assert kernel_based(x, y) > 0.05


@flaky(max_runs=5)
def test_given_categorical_dependent_data_when_perform_kernel_based_test_then_reject():
    x = np.random.normal(0, 1, 1000)
    y = []

    for v in x:
        if v > 0:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y).astype(str)

    assert kernel_based(x, y) < 0.05


@flaky(max_runs=2)
def test_given_random_seed_when_perform_pairwise_kernel_based_test_then_return_deterministic_result(
    _preserve_random_generator_state,
):
    x = np.random.randn(1000, 1)
    y = x + np.random.randn(1000, 1)

    assert kernel_based(
        x, y, bootstrap_num_samples_per_run=10, bootstrap_num_runs=2, p_value_adjust_func=np.mean
    ) != kernel_based(x, y, bootstrap_num_samples_per_run=10, bootstrap_num_runs=2, p_value_adjust_func=np.mean)

    np.random.seed(0)
    result_1 = kernel_based(x, y, bootstrap_num_samples_per_run=10, bootstrap_num_runs=2, p_value_adjust_func=np.mean)
    np.random.seed(0)
    result_2 = kernel_based(x, y, bootstrap_num_samples_per_run=10, bootstrap_num_runs=2, p_value_adjust_func=np.mean)

    assert result_1 == result_2


def test_given_constant_inputs_when_perform_kernel_based_test_then_returns_non_nan_value():
    assert kernel_based(np.random.normal(0, 1, (1000, 2)), np.array([5] * 1000)) != np.nan


@flaky(max_runs=5)
def test_given_continuous_conditionally_independent_data_when_perform_approx_kernel_based_test_then_not_reject():
    z = np.random.randn(5000, 1)
    x = np.exp(z + np.random.rand(5000, 1))
    y = np.exp(z + np.random.rand(5000, 1))

    assert (
        approx_kernel_based(x, y, z, num_random_features_X=10, num_random_features_Y=10, num_random_features_Z=10)
        > 0.05
    )


@flaky(max_runs=5)
def test_given_continuous_conditionally_dependent_data_when_perform_approx_kernel_based_test_then_reject():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert 0.05 > approx_kernel_based(x, y, w)


@flaky(max_runs=5)
def test_given_categorical_conditionally_independent_data_when_perform_approx_kernel_based_test_then_not_reject():
    x, y, z = _generate_categorical_data()

    assert approx_kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_given_categorical_conditionally_dependent_data_when_perform_approx_kernel_based_test_then_reject():
    x, y, z = _generate_categorical_data()

    assert approx_kernel_based(x, z, y) < 0.05


@flaky(max_runs=2)
def test_given_random_seed_when_perform_conditional_approx_kernel_based_test_then_return_deterministic_result(
    _preserve_random_generator_state,
):
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert approx_kernel_based(
        x,
        z,
        y,
        num_random_features_X=1,
        num_random_features_Y=1,
        num_random_features_Z=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    ) != approx_kernel_based(
        x,
        z,
        y,
        num_random_features_X=1,
        num_random_features_Y=1,
        num_random_features_Z=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    )

    np.random.seed(0)
    result_1 = approx_kernel_based(
        x,
        z,
        y,
        num_random_features_X=1,
        num_random_features_Y=1,
        num_random_features_Z=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    )
    np.random.seed(0)
    result_2 = approx_kernel_based(
        x,
        z,
        y,
        num_random_features_X=1,
        num_random_features_Y=1,
        num_random_features_Z=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    )

    assert result_1 == result_2


@flaky(max_runs=5)
def test_given_continuous_independent_data_when_perform_approx_kernel_based_test_then_not_reject():
    x = np.random.randn(1000, 1)
    y = np.exp(np.random.rand(1000, 1))
    assert approx_kernel_based(x, y) > 0.05


@flaky(max_runs=5)
def test_given_continuous_dependent_data_when_perform_approx_kernel_based_test_then_reject():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert approx_kernel_based(x, y) < 0.05


@flaky(max_runs=5)
def test_given_categorical_independent_data_when_perform_approx_kernel_based_test_then_not_reject():
    x = np.random.normal(0, 1, 1000)
    y = np.random.choice(2, 1000).astype(str)
    y[y == "0"] = "Class 1"
    y[y == "1"] = "Class 2"

    assert approx_kernel_based(x, y) > 0.05


@flaky(max_runs=5)
def test_given_categorical_dependent_data_when_perform_approx_kernel_based_test_then_reject():
    x = np.random.normal(0, 1, 1000)
    y = []

    for v in x:
        if v > 0:
            y.append("Class 1")
        else:
            y.append("Class 2")
    y = np.array(y).astype(str)

    assert approx_kernel_based(x, y) < 0.05


@flaky(max_runs=2)
def test_given_random_seed_when_perform_pairwise_approx_kernel_based_test_then_return_deterministic_result(
    _preserve_random_generator_state,
):
    w = np.random.randn(1000, 1)
    x = w + np.random.rand(1000, 1)

    assert approx_kernel_based(
        x,
        w,
        num_random_features_X=1,
        num_random_features_Y=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    ) != approx_kernel_based(
        x,
        w,
        num_random_features_X=1,
        num_random_features_Y=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    )

    np.random.seed(0)
    result_1 = approx_kernel_based(
        x,
        w,
        num_random_features_X=1,
        num_random_features_Y=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    )
    np.random.seed(0)
    result_2 = approx_kernel_based(
        x,
        w,
        num_random_features_X=1,
        num_random_features_Y=1,
        bootstrap_num_samples=5,
        bootstrap_num_runs=10,
        p_value_adjust_func=np.mean,
    )

    assert result_1 == result_2


def test_when_using_fast_centering_then_gives_expected_results():
    X = np.random.normal(0, 1, (100, 100))

    h = np.identity(X.shape[0]) - np.ones((X.shape[0], X.shape[0]), dtype=float) / X.shape[0]

    assert _fast_centering(X) == approx(h @ X @ h)


@flaky(max_runs=3)
def test_given_weak_dependency_when_perform_kernel_based_test_then_returns_expected_result():
    X = np.random.choice(2, (10000, 100))  # Require a lot of data here for the bootstraps.
    Y = np.sum(X * np.random.uniform(-1, 5), axis=1)

    assert kernel_based(X[:, 0], Y) <= 0.05
    assert kernel_based(np.random.choice(2, (10000, 1)), Y) > 0.05


def test_given_constant_data_when_perform_kernel_based_test_then_returns_expected_result():
    assert kernel_based(np.zeros(100), np.random.normal(0, 1, 100)) == 1.0
    assert kernel_based(np.zeros(100), np.random.normal(0, 1, 100), np.random.normal(0, 1, 100)) == 1.0
    assert not np.isnan(kernel_based(np.random.normal(0, 1, 100), np.random.normal(0, 1, 100), np.zeros(100)))


def test_given_constant_data_when_perform_approx_kernel_based_test_then_returns_expected_result():
    assert approx_kernel_based(np.zeros(100), np.random.normal(0, 1, 100)) == 1.0
    assert approx_kernel_based(np.zeros(100), np.random.normal(0, 1, 100), np.random.normal(0, 1, 100)) == 1.0
    assert not np.isnan(approx_kernel_based(np.random.normal(0, 1, 100), np.random.normal(0, 1, 100), np.zeros(100)))


def _generate_categorical_data():
    x = np.random.normal(0, 1, 1000)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + np.random.randn(len(z))
    z = np.array(z).astype(str)
    z[z == "0"] = "Class 1"
    z[z == "1"] = "Class 2"

    return x, y, z


@pytest.fixture
def _preserve_random_generator_state():
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(random_state)
