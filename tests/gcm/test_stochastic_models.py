import numpy as np
import pytest
from flaky import flaky
from pytest import approx
from scipy import stats

from dowhy.gcm import BayesianGaussianMixtureDistribution, EmpiricalDistribution, ScipyDistribution


def test_when_fitting_bayesian_gaussian_mixture_distribution_then_samples_should_be_drawn_correctly():
    test_data = np.array([[0, 0], [0, 0], [1, 2], [1, 2]])

    approximated_data_distribution_model = BayesianGaussianMixtureDistribution()
    approximated_data_distribution_model.fit(test_data)
    assert approximated_data_distribution_model.draw_samples(5).shape == (5, 2)


def test_when_drawing_samples_from_unfitted_bayesian_gaussian_mixture_distribution_then_runtime_error_should_occur():
    approximated_data_distribution_model = BayesianGaussianMixtureDistribution()
    with pytest.raises(RuntimeError):
        approximated_data_distribution_model.draw_samples(5)


@flaky(max_runs=3)
def test_given_two_independent_multimodal_variables_when_drawing_samples_then_they_are_not_spuriously_correlated():
    # sklearn's GaussianMixture.sample returns samples grouped by mixture component (not i.i.d. shuffled). If the samples
    # are returned in that order, two independently drawn multimodal variables end up sorted the same way and appear
    # almost perfectly correlated. draw_samples must shuffle to avoid this.
    def multimodal_data(num_samples):
        return np.concatenate(
            [
                np.random.normal(-5, 0.5, num_samples // 2),
                np.random.normal(5, 0.5, num_samples - num_samples // 2),
            ]
        )

    distribution_a = BayesianGaussianMixtureDistribution()
    distribution_a.fit(multimodal_data(2000))
    distribution_b = BayesianGaussianMixtureDistribution()
    distribution_b.fit(multimodal_data(2000))

    samples_a = distribution_a.draw_samples(2000).reshape(-1)
    samples_b = distribution_b.draw_samples(2000).reshape(-1)

    assert np.corrcoef(samples_a, samples_b)[0, 1] == approx(0, abs=0.15)


def test_when_creating_scipy_distribution_with_fixed_parameters_then_it_should_return_the_correct_parameter_values():
    distribution = ScipyDistribution(stats.norm, loc=0, scale=1)

    assert distribution.parameters["loc"] == 0
    assert distribution.parameters["scale"] == 1


@flaky(max_runs=5)
def test_when_fitting_normal_scipy_distribution_then_it_should_return_correctly_fitted_parameter_values():
    distribution = ScipyDistribution(stats.norm)

    X = np.random.normal(0, 1, 1000)
    distribution.fit(X)

    assert distribution.parameters["loc"] == approx(0, abs=0.1)
    assert distribution.parameters["scale"] == approx(1, abs=0.1)


@flaky(max_runs=5)
def test_given_gaussian_data_when_fitting_scipy_distribution_automatically_then_it_should_return_correctly_fitted_parameter_values():
    distribution = ScipyDistribution()

    X = np.random.normal(0, 1, 5000)
    distribution.fit(X)

    assert np.mean(distribution.draw_samples(1000)) == approx(0, abs=0.2)
    assert np.std(distribution.draw_samples(1000)) == approx(1, abs=0.2)


def test_when_drawing_samples_from_empirical_distribution_then_all_samples_should_be_present_in_the_data():
    X = np.random.normal(0, 1, 1000)

    distribution = EmpiricalDistribution()
    distribution.fit(X)

    X = list(X)

    for val in distribution.draw_samples(1000):
        assert val in X


@flaky(max_runs=5)
def test_when_fitting_scipy_distribution_with_normal_distribution_then_it_should_return_correctly_fitted_parameter_values():
    distribution = ScipyDistribution(stats.norm)
    distribution.fit(ScipyDistribution(stats.norm, loc=3, scale=2).draw_samples(10000))

    assert distribution.parameters["loc"] == approx(3, abs=0.3)
    assert distribution.parameters["scale"] == approx(2, abs=0.3)


@flaky(max_runs=5)
def test_when_fitting_scipy_distribution_with_beta_distribution_then_it_should_return_correctly_fitted_parameter_values():
    distribution = ScipyDistribution(stats.beta)
    distribution.fit(ScipyDistribution(stats.beta, a=2, b=2).draw_samples(10000))

    assert distribution.parameters["loc"] == approx(0, abs=0.1)
    assert distribution.parameters["scale"] == approx(1, abs=0.1)
    assert distribution.parameters["a"] == approx(2, abs=0.5)
    assert distribution.parameters["b"] == approx(2, abs=0.5)
