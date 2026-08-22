import numpy as np
from flaky import flaky
from pytest import approx
from scipy.stats import entropy

from dowhy.gcm.uncertainty import (
    estimate_entropy_discrete,
    estimate_entropy_kmeans,
    estimate_entropy_of_probabilities,
    estimate_entropy_using_discretization,
    estimate_gaussian_entropy,
    estimate_variance,
)


def test_given_gaussian_data_when_estimating_entropy_using_discretization_then_it_should_return_correct_entropy_values():
    X = np.random.normal(0, 1, 10000)
    Y = np.random.normal(0, 3, 10000)

    # With bin_width=1, the discretization has bins of unit width covering the data range. The estimate is close to the
    # theoretical differential entropy of the Gaussian (0.5 * ln(2 * pi * e * sigma^2)), i.e. ~1.42 and ~2.52.
    assert estimate_entropy_using_discretization(X) == approx(1.42, abs=0.2)
    assert estimate_entropy_using_discretization(Y) == approx(2.52, abs=0.2)


def test_given_data_with_range_smaller_than_two_bin_widths_when_estimating_entropy_then_it_is_positive():
    # Regression test: when the data range is smaller than 2 * bin_width the number of bins used to be collapsed to 1
    # (an empty histogram), so the estimated entropy was silently 0 regardless of the actual distribution.
    X = np.random.uniform(0, 1.9, 1000)

    assert estimate_entropy_using_discretization(X, bin_width=1) > 0


def test_given_constant_data_when_estimating_entropy_using_discretization_then_it_is_zero():
    assert estimate_entropy_using_discretization(np.ones(100)) == approx(0.0)


def test_given_gaussian_data_when_estimating_entropy_using_kmeans_then_it_should_return_correct_entropy_values():
    X = np.random.normal(0, 1, 10000)
    Y = np.random.normal(0, 3, 10000)

    assert estimate_entropy_kmeans(X) == approx(1.4, abs=0.2)
    assert estimate_entropy_kmeans(Y) == approx(2.5, abs=0.2)


def test_given_gaussian_data_when_estimating_gaussian_entropy_then_it_should_return_correct_entropy_values():
    X = np.random.normal(0, 1, 10000)
    Y = np.random.normal(0, 3, 10000)

    assert estimate_gaussian_entropy(X) == approx(1.4, abs=0.2)
    assert estimate_gaussian_entropy(Y) == approx(2.5, abs=0.2)


@flaky(max_runs=5)
def test_given_gaussian_data_when_estimating_variance_then_it_should_return_correct_variance_values():
    X = np.random.normal(0, 1, 10000)
    Y = np.random.normal(0, 3, 10000)

    assert estimate_variance(X) == approx(1, abs=0.2)
    assert estimate_variance(Y) == approx(9, abs=0.2)

    X = np.random.normal(0, 1, (1000000, 3))
    Y = np.random.normal(0, 3, (1000000, 3))

    assert estimate_variance(X) == approx(1, abs=1)
    assert estimate_variance(Y) == approx(729, abs=15)


def test_given_probabilities_when_estimating_entropy_of_probabilities_then_it_should_return_correct_entropy_values():
    assert estimate_entropy_of_probabilities(np.array([[0.25, 0.25, 0.25, 0.25]])) == approx(1.38, abs=0.05)
    assert estimate_entropy_of_probabilities(np.array([[1, 0, 0, 0]])) == approx(0, abs=0.05)
    assert estimate_entropy_of_probabilities(np.array([[0.5, 0, 0, 0.5]])) == approx(0.69, abs=0.05)
    assert estimate_entropy_of_probabilities(np.array([[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5]])) == approx(0.69, abs=0.05)
    assert estimate_entropy_of_probabilities(np.array([[1, 0, 0, 0], [1, 0, 0, 0]])) == approx(0, abs=0.05)


def test_given_discrete_data_when_estimating_entropy_discrete_then_it_should_return_correct_entropy_values():
    assert estimate_entropy_discrete(np.random.choice(2, (1000, 1), replace=True)) == approx(entropy([0.5, 0.5]), 0.05)
    assert estimate_entropy_discrete(np.random.choice(2, (1000, 2), replace=True)) == approx(
        entropy([0.25, 0.25, 0.25, 0.25]), 0.05
    )
