import random

import numpy as np
import pytest
from _pytest.python_api import approx
from flaky import flaky

from dowhy.gcm.independence_test import kernel_based, approx_kernel_based
from dowhy.gcm.independence_test.kernel import _fast_centering


@pytest.fixture
def preserve_random_generator_state():
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(random_state)


@flaky(max_runs=5)
def test_kernel_based_conditional_independence_test_independent():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_kernel_based_based_conditional_independence_test_dependent():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert 0.05 > kernel_based(x, y, w)


@flaky(max_runs=5)
def test_kernel_based_conditional_independence_test_categorical_independent():
    x, y, z = generate_categorical_data()

    assert kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_kernel_based_conditional_independence_test_categorical_dependent():
    x, y, z = generate_categorical_data()

    assert kernel_based(x, z, y) < 0.05


@flaky(max_runs=2)
def test_kernel_based_conditional_independence_test_with_random_seed(preserve_random_generator_state):
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(x, z, y,
                        bootstrap_num_samples_per_run=5,
                        bootstrap_num_runs=2,
                        p_value_adjust_func=np.mean) \
           != kernel_based(x, z, y,
                           bootstrap_num_samples_per_run=5,
                           bootstrap_num_runs=2,
                           p_value_adjust_func=np.mean)

    np.random.seed(0)
    result_1 = kernel_based(x, z, y,
                            bootstrap_num_samples_per_run=5,
                            bootstrap_num_runs=2,
                            p_value_adjust_func=np.mean)
    np.random.seed(0)
    result_2 = kernel_based(x, z, y,
                            bootstrap_num_samples_per_run=5,
                            bootstrap_num_runs=2,
                            p_value_adjust_func=np.mean)

    assert result_1 == result_2


def test_kernel_based_pairwise_independence_test_raises_error_when_too_few_samples():
    with pytest.raises(RuntimeError):
        kernel_based(np.array([1, 2, 3, 4]),
                     np.array([1, 3, 2, 4]))


@flaky(max_runs=5)
def test_kernel_based_pairwise_independence_test_independent():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(x, w) > 0.05


@flaky(max_runs=5)
def test_kernel_based_pairwise_independence_test_dependent():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert kernel_based(x, y) < 0.05


@flaky(max_runs=5)
def test_kernel_based_pairwise_independence_test_categorical_independent():
    x = np.random.normal(0, 1, 1000)
    y = (np.random.choice(2, 1000) == 1).astype(str)

    assert kernel_based(x, y) > 0.05


@flaky(max_runs=5)
def test_kernel_based_pairwise_independence_test_categorical_dependent():
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
def test_kernel_based_pairwise_independence_test_with_random_seed(preserve_random_generator_state):
    x = np.random.randn(1000, 1)
    y = x + np.random.randn(1000, 1)

    assert kernel_based(x, y,
                        bootstrap_num_samples_per_run=10,
                        bootstrap_num_runs=2,
                        p_value_adjust_func=np.mean) \
           != kernel_based(x, y,
                           bootstrap_num_samples_per_run=10,
                           bootstrap_num_runs=2,
                           p_value_adjust_func=np.mean)

    np.random.seed(0)
    result_1 = kernel_based(x, y, bootstrap_num_samples_per_run=10,
                            bootstrap_num_runs=2,
                            p_value_adjust_func=np.mean)
    np.random.seed(0)
    result_2 = kernel_based(x, y, bootstrap_num_samples_per_run=10,
                            bootstrap_num_runs=2,
                            p_value_adjust_func=np.mean)

    assert result_1 == result_2


def test_kernel_based_pairwise_with_constant():
    assert kernel_based(np.random.normal(0, 1, (1000, 2)), np.array([5] * 1000)) != np.nan


@flaky(max_runs=5)
def test_approx_kernel_based_conditional_independence_test_independent():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert approx_kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_approx_kernel_based_conditional_independence_test_dependent():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert 0.05 > approx_kernel_based(x, y, w)


@flaky(max_runs=5)
def test_approx_kernel_based_conditional_independence_test_categorical_independent():
    x, y, z = generate_categorical_data()

    assert approx_kernel_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_approx_kernel_based_conditional_independence_test_categorical_dependent():
    x, y, z = generate_categorical_data()

    assert approx_kernel_based(x, z, y) < 0.05


@flaky(max_runs=2)
def test_approx_kernel_based_conditional_independence_test_with_random_seed(preserve_random_generator_state):
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert approx_kernel_based(x,
                               z,
                               y,
                               num_random_features_X=1,
                               num_random_features_Y=1,
                               num_random_features_Z=1,
                               bootstrap_num_samples=5,
                               bootstrap_num_runs=10,
                               p_value_adjust_func=np.mean) \
           != approx_kernel_based(x,
                                  z,
                                  y,
                                  num_random_features_X=1,
                                  num_random_features_Y=1,
                                  num_random_features_Z=1,
                                  bootstrap_num_samples=5,
                                  bootstrap_num_runs=10,
                                  p_value_adjust_func=np.mean)

    np.random.seed(0)
    result_1 = approx_kernel_based(x,
                                   z,
                                   y,
                                   num_random_features_X=1,
                                   num_random_features_Y=1,
                                   num_random_features_Z=1,
                                   bootstrap_num_samples=5,
                                   bootstrap_num_runs=10,
                                   p_value_adjust_func=np.mean)
    np.random.seed(0)
    result_2 = approx_kernel_based(x,
                                   z,
                                   y,
                                   num_random_features_X=1,
                                   num_random_features_Y=1,
                                   num_random_features_Z=1,
                                   bootstrap_num_samples=5,
                                   bootstrap_num_runs=10,
                                   p_value_adjust_func=np.mean)

    assert result_1 == result_2


@flaky(max_runs=5)
def test_approx_kernel_based_pairwise_independence_test_independent():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    assert approx_kernel_based(x, w) > 0.05


@flaky(max_runs=5)
def test_approx_kernel_based_pairwise_independence_test_dependent():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert approx_kernel_based(x, y) < 0.05


@flaky(max_runs=5)
def test_approx_kernel_based_pairwise_independence_test_categorical_independent():
    x = np.random.normal(0, 1, 1000)
    y = np.random.choice(2, 1000).astype(str)
    y[y == '0'] = 'Class 1'
    y[y == '1'] = 'Class 2'

    assert approx_kernel_based(x, y) > 0.05


@flaky(max_runs=5)
def test_approx_kernel_based_pairwise_independence_test_categorical_dependent():
    x = np.random.normal(0, 1, 1000)
    y = []

    for v in x:
        if v > 0:
            y.append('Class 1')
        else:
            y.append('Class 2')
    y = np.array(y).astype(str)

    assert approx_kernel_based(x, y) < 0.05


@flaky(max_runs=2)
def test_approx_kernel_based_pairwise_independence_test_with_random_seed(preserve_random_generator_state):
    w = np.random.randn(1000, 1)
    x = w + np.random.rand(1000, 1)

    assert approx_kernel_based(x,
                               w,
                               num_random_features_X=1,
                               num_random_features_Y=1,
                               bootstrap_num_samples=5,
                               bootstrap_num_runs=10,
                               p_value_adjust_func=np.mean) \
           != approx_kernel_based(x,
                                  w,
                                  num_random_features_X=1,
                                  num_random_features_Y=1,
                                  bootstrap_num_samples=5,
                                  bootstrap_num_runs=10,
                                  p_value_adjust_func=np.mean)

    np.random.seed(0)
    result_1 = approx_kernel_based(x,
                                   w,
                                   num_random_features_X=1,
                                   num_random_features_Y=1,
                                   bootstrap_num_samples=5,
                                   bootstrap_num_runs=10,
                                   p_value_adjust_func=np.mean)
    np.random.seed(0)
    result_2 = approx_kernel_based(x,
                                   w,
                                   num_random_features_X=1,
                                   num_random_features_Y=1,
                                   bootstrap_num_samples=5,
                                   bootstrap_num_runs=10,
                                   p_value_adjust_func=np.mean)

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


def generate_categorical_data():
    x = np.random.normal(0, 1, 1000)
    z = []
    for v in x:
        if v > 0:
            z.append(0)
        else:
            z.append(1)
    y = z + np.random.randn(len(z))
    z = np.array(z).astype(str)
    z[z == '0'] = 'Class 1'
    z[z == '1'] = 'Class 2'

    return x, y, z
