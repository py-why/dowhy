import numpy as np
import pytest

from dowhy.gcm.density_estimators import GaussianMixtureDensityEstimator, KernelDensityEstimator1D


def test_when_fit_and_evaluate_gaussian_mixture_density_estimator_then_behaves_as_expected():
    test_data = np.array([[0, 1], [0, 2], [1, 0], [2, 3]])

    density_estimator_model = GaussianMixtureDensityEstimator()
    with pytest.raises(RuntimeError):
        density_estimator_model.density(test_data)

    density_estimator_model.fit(test_data)
    results = density_estimator_model.density(test_data)

    assert len(results) == 4


def test_when_fit_and_evaluate_gaussian_mixture_density_estimator_with_integer_input_then_behaves_as_expected():
    # sklearn >= 1.9 changed how integer-dtype inputs are handled in BayesianGaussianMixture;
    # ensure results are finite and positive when X has an integer dtype.
    rng = np.random.default_rng(0)
    test_data = rng.integers(0, 10, size=(50, 2))
    assert test_data.dtype == np.int64

    density_estimator_model = GaussianMixtureDensityEstimator()
    density_estimator_model.fit(test_data)
    results = density_estimator_model.density(test_data)

    assert len(results) == 50
    assert np.all(np.isfinite(results))
    assert np.all(results > 0)


def test_when_fit_and_evaluate_kernel_based_density_estimator_1d_then_behaves_as_expected():
    test_data = np.array([[0, 1], [0, 2], [1, 0], [2, 3]])

    density_estimator_model = KernelDensityEstimator1D()
    with pytest.raises(RuntimeError):
        density_estimator_model.density(test_data)

    with pytest.raises(RuntimeError):
        density_estimator_model.fit(test_data)

    test_data = np.array([[0], [2], [1], [3]])
    density_estimator_model.fit(test_data)
    results = density_estimator_model.density(test_data)

    assert len(results) == 4


def test_when_fit_and_evaluate_kernel_based_density_estimator_1d_with_integer_input_then_behaves_as_expected():
    # Ensure KernelDensityEstimator1D also works correctly with integer-dtype input.
    rng = np.random.default_rng(0)
    test_data = rng.integers(0, 20, size=(50, 1))
    assert test_data.dtype == np.int64

    density_estimator_model = KernelDensityEstimator1D()
    density_estimator_model.fit(test_data)
    results = density_estimator_model.density(test_data)

    assert len(results) == 50
    assert np.all(np.isfinite(results))
    assert np.all(results > 0)
