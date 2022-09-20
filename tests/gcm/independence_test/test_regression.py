import random

import numpy as np
import pytest
from flaky import flaky

from dowhy.gcm.independence_test import regression_based
from dowhy.gcm.util.general import set_random_seed
from tests.gcm.independence_test.test_kernel import _generate_categorical_data


@pytest.fixture
def preserve_random_generator_state():
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(random_state)


@flaky(max_runs=5)
def test_regression_based_conditional_independence_test_independent():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert regression_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_regression_based_conditional_independence_test_dependent():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert 0.05 > regression_based(x, y, w)


@flaky(max_runs=5)
def test_regression_based_conditional_independence_test_categorical_independent():
    x, y, z = _generate_categorical_data()

    assert regression_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_regression_based_conditional_independence_test_categorical_dependent():
    x, y, z = _generate_categorical_data()

    assert regression_based(x, z, y) < 0.05


@flaky(max_runs=5)
def test_regression_based_pairwise_independence_test_independent():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))

    assert regression_based(x, w) > 0.05


@flaky(max_runs=5)
def test_regression_based_pairwise_independence_test_dependent():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert regression_based(x, y) < 0.05


@flaky(max_runs=5)
def test_regression_based_pairwise_independence_test_categorical_independent():
    x = np.random.normal(0, 1, 1000)
    y = np.random.choice(2, 1000).astype(str)

    assert regression_based(x, y) > 0.05


@flaky(max_runs=5)
def test_regression_based_pairwise_independence_test_categorical_dependent():
    x = np.random.normal(0, 1, 1000)
    y = []

    for v in x:
        if v > 0:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y).astype(str)

    assert regression_based(x, y, num_components_all_inputs=10) < 0.05


def test_regression_based_conditional_independence_parallelization(preserve_random_generator_state):
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    set_random_seed(0)
    p_value_1 = regression_based(x, y, z)
    set_random_seed(0)
    p_value_2 = regression_based(x, y, z)

    assert p_value_1 == p_value_2


@flaky(max_runs=5)
def test_regression_based_conditional_independence_test_categorical_dependent():
    x, y, z = _generate_categorical_data()

    assert regression_based(x, z, y) < 0.05
