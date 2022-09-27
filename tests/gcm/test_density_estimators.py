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
