import numpy as np
from flaky import flaky
from pytest import approx

from dowhy.gcm.divergence import estimate_kl_divergence_continuous, estimate_kl_divergence_categorical, \
    estimate_kl_divergence_of_probabilities, auto_estimate_kl_divergence


@flaky(max_runs=5)
def test_estimate_kl_divergence_continuous():
    X = np.random.normal(0, 1, 10000)
    Y = np.random.normal(1, 1, 10000)

    assert estimate_kl_divergence_continuous(X, X) == approx(0, abs=0.001)
    assert estimate_kl_divergence_continuous(X, Y) == approx(0.5, abs=0.1)


@flaky(max_runs=5)
def test_estimate_kl_divergence_categorical():
    X = np.random.choice(4, 1000, replace=True, p=[0.25, 0.5, 0.125, 0.125]).astype(str)
    Y = np.random.choice(4, 1000, replace=True, p=[0.5, 0.25, 0.125, 0.125]).astype(str)

    assert estimate_kl_divergence_categorical(X, X) == approx(0)
    assert estimate_kl_divergence_categorical(X, Y) \
           == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.1)


def test_estimate_kl_divergence_of_probabilities():
    assert estimate_kl_divergence_of_probabilities(
        np.array([[0.25, 0.5, 0.125, 0.125], [0.5, 0.25, 0.125, 0.125]]),
        np.array([[0.5, 0.25, 0.125, 0.125], [0.25, 0.5, 0.125, 0.125]])) \
           == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.01)


@flaky(max_runs=5)
def test_auto_estimate_kl_divergence_continuous():
    X = np.random.normal(0, 1, 10000)
    Y = np.random.normal(1, 1, 10000)

    assert auto_estimate_kl_divergence(X, X) == approx(0, abs=0.001)
    assert auto_estimate_kl_divergence(X, Y) == approx(0.5, abs=0.1)


@flaky(max_runs=5)
def test_auto_estimate_kl_divergence_categorical():
    X = np.random.choice(4, 1000, replace=True, p=[0.25, 0.5, 0.125, 0.125]).astype(str)
    Y = np.random.choice(4, 1000, replace=True, p=[0.5, 0.25, 0.125, 0.125]).astype(str)

    assert auto_estimate_kl_divergence(X, X) == approx(0)
    assert auto_estimate_kl_divergence(X, Y) \
           == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.1)


def test_auto_estimate_kl_divergence_probabilities():
    assert auto_estimate_kl_divergence(
        np.array([[0.25, 0.5, 0.125, 0.125], [0.5, 0.25, 0.125, 0.125]]),
        np.array([[0.5, 0.25, 0.125, 0.125], [0.25, 0.5, 0.125, 0.125]])) \
           == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.01)
