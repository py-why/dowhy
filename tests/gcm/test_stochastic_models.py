import numpy as np
import pytest
from flaky import flaky
from pytest import approx
from scipy import stats

from dowhy.gcm import BayesianGaussianMixtureDistribution, EmpiricalDistribution, ScipyDistribution


def test_bayesian_gaussian_mixture_distribution():
    test_data = np.array([[0, 0], [0, 0], [1, 2], [1, 2]])

    approximated_data_distribution_model = BayesianGaussianMixtureDistribution()
    approximated_data_distribution_model.fit(test_data)
    assert approximated_data_distribution_model.draw_samples(5).shape == (5, 2)


def test_bayesian_gaussian_mixture_distribution_runtime_error():
    approximated_data_distribution_model = BayesianGaussianMixtureDistribution()
    with pytest.raises(RuntimeError):
        approximated_data_distribution_model.draw_samples(5)


def test_scipy_fixed_parametric_distribution():
    distribution = ScipyDistribution(stats.norm, loc=0, scale=1)

    assert distribution.parameters["loc"] == 0
    assert distribution.parameters["scale"] == 1


@flaky(max_runs=5)
def test_scipy_fittable_parametric_distribution():
    distribution = ScipyDistribution(stats.norm)

    X = np.random.normal(0, 1, 1000)
    distribution.fit(X)

    assert distribution.parameters["loc"] == approx(0, abs=0.1)
    assert distribution.parameters["scale"] == approx(1, abs=0.1)


@flaky(max_runs=5)
def test_scipy_auto_select_continuous_parametric_distribution():
    distribution = ScipyDistribution()

    X = np.random.normal(0, 1, 1000)
    distribution.fit(X)

    assert np.mean(distribution.draw_samples(1000)) == approx(0, abs=0.1)
    assert np.std(distribution.draw_samples(1000)) == approx(1, abs=0.1)


def test_empirical_distribution():
    X = np.random.normal(0, 1, 1000)

    distribution = EmpiricalDistribution()
    distribution.fit(X)

    X = list(X)

    for val in distribution.draw_samples(1000):
        assert val in X


@flaky(max_runs=5)
def test_fitted_parameters_assigned_correctly_using_normal_distribution():
    distribution = ScipyDistribution(stats.norm)
    distribution.fit(ScipyDistribution(stats.norm, loc=3, scale=2).draw_samples(10000))

    assert distribution.parameters["loc"] == approx(3, abs=0.3)
    assert distribution.parameters["scale"] == approx(2, abs=0.3)


@flaky(max_runs=5)
def test_fitted_parameters_assigned_correctly_using_beta_distribution():
    distribution = ScipyDistribution(stats.beta)
    distribution.fit(ScipyDistribution(stats.beta, a=2, b=0.5).draw_samples(10000))

    assert distribution.parameters["loc"] == approx(0, abs=0.1)
    assert distribution.parameters["scale"] == approx(1, abs=0.1)
    assert distribution.parameters["a"] == approx(2, abs=0.5)
    assert distribution.parameters["b"] == approx(0.5, abs=0.5)
