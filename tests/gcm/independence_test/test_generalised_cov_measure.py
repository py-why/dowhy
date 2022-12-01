import random

import numpy as np
import pytest
from flaky import flaky

from dowhy.gcm import generalised_cov_based
from tests.gcm.independence_test.test_kernel import _generate_categorical_data


@pytest.fixture
def preserve_random_generator_state():
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(random_state)


@flaky(max_runs=5)
def test_given_conditional_independent_nonlinear_data_when_perform_generalised_cov_based_independence_test_then_returns_expected_result():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert generalised_cov_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_given_dependent_nonlinear_data_when_perform_generalised_cov_based_independence_test_then_returns_expected_result():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert 0.05 > generalised_cov_based(x, y, w)


@flaky(max_runs=5)
def test_given_independent_categorical_Z_data_when_perform_generalised_cov_based_independence_test_then_returns_expected_result():
    x, y, z = _generate_categorical_data()

    assert generalised_cov_based(x, y, z) > 0.05


@flaky(max_runs=5)
def test_given_dependent_categorical_Z_data_when_perform_generalised_cov_based_independence_test_then_returns_expected_result():
    x, y, z = _generate_categorical_data()

    assert generalised_cov_based(x, y, z[np.random.choice(z.shape[0], z.shape[0], replace=False)]) < 0.05


@flaky(max_runs=5)
def test_given_unconditional_independent_nonlinear_data_when_perform_generalised_cov_based_independence_test_then_returns_expected_result():
    z = np.random.randn(1000, 1)
    w = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))

    assert generalised_cov_based(x, w) > 0.05


@flaky(max_runs=5)
def test_given_unconditional_dependent_nonlinear_data_when_perform_generalised_cov_based_independence_test_then_returns_expected_result():
    z = np.random.randn(1000, 1)
    x = np.exp(z + np.random.rand(1000, 1))
    y = np.exp(z + np.random.rand(1000, 1))

    assert generalised_cov_based(x, y) < 0.05


def test_given_categorical_X_or_Y_when_perform_generalised_cov_based_independence_test_then_raise_error():
    with pytest.raises(ValueError):
        generalised_cov_based(np.random.choice(2, 1000, replace=True).astype(str), np.random.normal(0, 1, 1000))

    with pytest.raises(ValueError):
        generalised_cov_based(np.random.normal(0, 1, 1000), np.random.choice(2, 1000, replace=True).astype(str))
