import numpy as np
from flaky import flaky
from pytest import approx

from dowhy.gcm.divergence import (
    auto_estimate_kl_divergence,
    estimate_kl_divergence_categorical,
    estimate_kl_divergence_continuous_clf,
    estimate_kl_divergence_continuous_knn,
    estimate_kl_divergence_of_probabilities,
    is_probability_matrix,
)


@flaky(max_runs=3)
def test_given_simple_gaussian_data_when_estimate_kl_divergence_continuous_then_returns_expected_result():
    X = np.random.normal(0, 1, 2000)
    Y = np.random.normal(1, 1, 2000)

    assert estimate_kl_divergence_continuous_knn(X, X) == approx(0, abs=0.001)
    assert estimate_kl_divergence_continuous_knn(X, Y) == approx(0.5, abs=0.15)


@flaky(max_runs=3)
def test_given_simple_categorical_data_estimate_kl_divergence_categorical_then_returns_expected_result():
    X = np.random.choice(4, 1000, replace=True, p=[0.25, 0.5, 0.125, 0.125]).astype(str)
    Y = np.random.choice(4, 1000, replace=True, p=[0.5, 0.25, 0.125, 0.125]).astype(str)

    assert estimate_kl_divergence_categorical(X, X) == approx(0)
    assert estimate_kl_divergence_categorical(X, Y) == approx(
        0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.1
    )


def test_given_probability_vectors_when_estimate_kl_divergence_of_probabilities_then_returns_expected_result():
    assert estimate_kl_divergence_of_probabilities(
        np.array([[0.25, 0.5, 0.125, 0.125], [0.5, 0.25, 0.125, 0.125]]),
        np.array([[0.5, 0.25, 0.125, 0.125], [0.25, 0.5, 0.125, 0.125]]),
    ) == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.01)


@flaky(max_runs=3)
def test_given_simple_gaussian_data_when_auto_estimate_kl_divergence_then_correctly_selects_continuous_version():
    X = np.random.normal(0, 1, 2000)
    Y = np.random.normal(1, 1, 2000)

    assert auto_estimate_kl_divergence(X, X) == approx(0, abs=0.001)
    assert auto_estimate_kl_divergence(X, Y) == approx(0.5, abs=0.15)


@flaky(max_runs=3)
def test_given_categorical_data_when_auto_estimate_kl_divergence_then_correctly_selects_categorical_version():
    X = np.random.choice(4, 1000, replace=True, p=[0.25, 0.5, 0.125, 0.125]).astype(str)
    Y = np.random.choice(4, 1000, replace=True, p=[0.5, 0.25, 0.125, 0.125]).astype(str)

    assert auto_estimate_kl_divergence(X, X) == approx(0)
    assert auto_estimate_kl_divergence(X, Y) == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.1)


def test_given_probability_vectors_when_auto_estimate_kl_divergence_then_correctly_selects_probability_version():
    assert auto_estimate_kl_divergence(
        np.array([[0.25, 0.5, 0.125, 0.125], [0.5, 0.25, 0.125, 0.125]]),
        np.array([[0.5, 0.25, 0.125, 0.125], [0.25, 0.5, 0.125, 0.125]]),
    ) == approx(0.25 * np.log(0.25 / 0.5) + 0.5 * np.log(0.5 / 0.25), abs=0.01)


def test_given_valid_and_invalid_probability_vectors_when_apply_is_probabilities_then_return_expected_results():
    assert is_probability_matrix(np.array([0.5, 0.3, 0.2]))
    assert not is_probability_matrix(np.array([0.1, 0.3, 0.2]))
    assert is_probability_matrix(np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]))
    assert not is_probability_matrix(np.random.normal(0, 1, (5, 3)))


def test_given_numpy_array_with_object_dtype_when_check_is_probability_matrix_then_does_not_raise_error():
    assert not is_probability_matrix(np.array([0, 1, 2], dtype=object))


@flaky(max_runs=3)
def test_given_simple_gaussian_data_with_overlap_when_estimate_kl_divergence_continuous_then_does_not_return_inf():
    X = np.random.normal(0, 1, 2000)
    Y = np.random.normal(1, 1, 2000)

    Y[:10] = X[:10]

    assert estimate_kl_divergence_continuous_knn(X, X) == approx(0, abs=0.001)
    assert estimate_kl_divergence_continuous_knn(X, Y) == approx(0.5, abs=0.15)


@flaky(max_runs=3)
def test_given_simple_gaussian_data_when_estimate_kl_divergence_continuous_clf_then_returns_correct_result():
    X = np.random.normal(0, 1, 2000)
    Y = np.random.normal(1, 1, 2000)

    assert estimate_kl_divergence_continuous_clf(X, X) == approx(0, abs=0.001)
    assert estimate_kl_divergence_continuous_clf(X, Y) == approx(0.5, abs=0.15)


@flaky(max_runs=3)
def test_given_multi_dim_simple_gaussian_data_when_estimate_kl_divergence_continuous_clf_then_returns_correct_result():
    X = np.random.normal(0, 1, (2000, 2))
    Y = np.random.normal(1, 1, (2000, 2))

    assert estimate_kl_divergence_continuous_clf(X, X) == approx(0, abs=0.001)
    assert estimate_kl_divergence_continuous_clf(X, Y) == approx(1, abs=0.2)


@flaky(max_runs=3)
def test_given_multi_dim_gaussian_and_categorical_data_when_estimate_kl_divergence_continuous_clf_then_returns_correct_result():
    X0 = np.random.normal(0, 1, 2000)
    X1 = np.random.choice(3, 2000, replace=True).astype(str)

    X0_other = np.random.normal(1, 1, 2000)
    X1_other = (np.random.choice(4, 2000, replace=True)).astype(str)

    X = np.array([X0, X1], dtype=object).T

    assert estimate_kl_divergence_continuous_clf(X, X) == approx(0, abs=0.001)
    # Only Gaussian component changed
    assert estimate_kl_divergence_continuous_clf(X, np.array([X0_other, X1], dtype=object).T) == approx(0.5, abs=0.15)
    assert estimate_kl_divergence_continuous_clf(X, np.array([X0_other, X1_other], dtype=object).T) == approx(
        0.78, abs=0.15
    )


@flaky(max_runs=3)
def test_given_exponential_data_when_estimate_kl_divergence_continuous_then_returns_correct_result():
    X = np.random.exponential(1, 2000)
    Y = np.random.exponential(0.5, 2000)

    assert estimate_kl_divergence_continuous_clf(X, Y) == approx(0.31, abs=0.1)
    assert estimate_kl_divergence_continuous_knn(X, Y) == approx(0.31, abs=0.1)
