from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx
from scipy import stats
from scipy.stats import norm

from dowhy.gcm import (
    AdditiveNoiseModel,
    ClassifierFCM,
    DiscreteAdditiveNoiseModel,
    EmpiricalDistribution,
    PostNonlinearModel,
    ProbabilisticCausalModel,
    ScipyDistribution,
    StructuralCausalModel,
    draw_samples,
    fit,
)
from dowhy.gcm.auto import assign_causal_mechanisms
from dowhy.gcm.divergence import estimate_kl_divergence_continuous_clf
from dowhy.gcm.ml import (
    SklearnRegressionModel,
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
    create_logistic_regression_classifier,
)
from dowhy.gcm.ml.regression import (
    InvertibleExponentialFunction,
    InvertibleIdentityFunction,
    InvertibleLogarithmicFunction,
)
from dowhy.gcm.util.general import is_discrete


def test_given_linear_data_when_fit_causal_graph_with_linear_anm_then_learns_correct_coefficients():
    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1")]))
    scm.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = scm.causal_mechanism("X0").draw_samples(1000).squeeze()
    test_data = pd.DataFrame({"X0": X0, "X1": X0 * 2 + 2 + np.random.normal(0, 0.1, 1000)})
    fit(scm, test_data)

    assert scm.causal_mechanism("X1").prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.02)
    assert scm.causal_mechanism("X1").prediction_model.sklearn_model.intercept_ == approx(2, abs=0.02)


@flaky(max_runs=3)
def test_given_linear_data_when_draw_samples_from_fitted_anm_then_generates_correct_marginal_distribution():
    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1")]))
    scm.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = scm.causal_mechanism("X0").draw_samples(10000).squeeze()
    test_data = pd.DataFrame({"X0": X0, "X1": X0 * 2 + 2 + np.random.normal(0, 0.1, 10000)})
    fit(scm, test_data)

    generated_samples = scm.causal_mechanism("X1").draw_samples(np.array([1] * 1000))
    assert np.mean(generated_samples) == approx(4, abs=0.05)
    assert np.std(generated_samples) == approx(0.1, abs=0.05)

    generated_samples = scm.causal_mechanism("X1").draw_samples(np.array([2] * 1000))
    assert np.mean(generated_samples) == approx(6, abs=0.05)
    assert np.std(generated_samples) == approx(0.1, abs=0.05)
    assert estimate_kl_divergence_continuous_clf(
        test_data["X1"].to_numpy(), draw_samples(scm, 10000)["X1"].to_numpy()
    ) == approx(0, abs=0.05)


@flaky(max_runs=5)
def test_given_categorical_input_data_when_fit_causal_graph_with_linear_anm_then_learns_correct_coefficients():
    scm = StructuralCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2")]))
    scm.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism("X1", EmpiricalDistribution())
    scm.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    training_data = _generate_data_with_categorical_input()
    fit(scm, data=training_data)

    # Mean from the categorical part is: (-5 + 5+ 10) / 3 = 10/3
    assert scm.causal_mechanism("X2").prediction_model.sklearn_model.coef_ == approx(
        np.array([2, -5 - 10 / 3, 10 - 10 / 3, 5 - 10 / 3]), abs=0.02
    )
    assert scm.causal_mechanism("X2").prediction_model.sklearn_model.intercept_ == approx(10 / 3, abs=0.02)


@flaky(max_runs=3)
def test_given_categorical_input_data_when_draw_from_fitted_causal_graph_with_linear_anm_then_generates_correct_marginal_distribution():
    training_data = _generate_data_with_categorical_input()
    scm = StructuralCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2")]))
    assign_causal_mechanisms(scm, training_data)

    fit(scm, data=training_data)

    assert scm.causal_mechanism("X2").evaluate(np.array([[2, "1"]], dtype=object), np.array([0])) == approx(14)

    test_data = training_data.to_numpy()
    assert scm.causal_mechanism("X2").evaluate(test_data[:, :2], np.array([0] * 1)) == approx(
        test_data[:, 2].astype(float).reshape(-1, 1)
    )

    assert estimate_kl_divergence_continuous_clf(test_data[:, 2], draw_samples(scm, 1000)["X2"].to_numpy()) == approx(
        0, abs=0.05
    )


def test_given_non_string_data_when_try_to_fit_classifier_fcm_then_throws_error():
    scm = StructuralCausalModel(nx.DiGraph([("X0", "X1")]))
    scm.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism("X1", ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = (X0 > np.median(X0)).astype(int)

    with pytest.raises(ValueError):
        fit(scm, data=(pd.DataFrame({"X0": X0, "X1": X1})))


def test_when_draw_from_classifier_fcm_then_returns_string_samples():
    scm = StructuralCausalModel(nx.DiGraph([("X0", "X1")]))
    scm.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism("X1", ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = (X0 > np.median(X0)).astype(str)

    training_data = pd.DataFrame({"X0": X0, "X1": X1})

    fit(scm, training_data)

    for val in scm.causal_mechanism("X1").draw_samples(X0):
        assert isinstance(val[0], str)


@flaky(max_runs=5)
def test_when_fit_classifier_fcm_with_categorical_inputs_then_returns_expected_results():
    scm = StructuralCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2")]))
    scm.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism("X1", EmpiricalDistribution())
    scm.set_causal_mechanism("X2", ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    training_data = _generate_data_with_categorical_input()
    X2 = training_data["X2"].to_numpy()
    training_data["X2"] = (X2 > np.median(X2)).astype(str)
    X2 = training_data["X2"].to_numpy()
    fit(scm, training_data)

    x2_fcm = cast(ClassifierFCM, scm.causal_mechanism("X2"))
    assert x2_fcm.estimate_probabilities(np.array([[2, "1"]], dtype=object)) == approx(np.array([[0, 1]]), abs=0.01)

    test_data = training_data.to_numpy()
    X2[X2 == "True"] = 1
    X2[X2 == "False"] = 0
    assert np.sum(np.argmax(x2_fcm.estimate_probabilities(test_data[:, :2]), axis=1) != X2.astype(int)) < 20

    _, counts = np.unique(x2_fcm.draw_samples(test_data[:, :2]), return_counts=True)
    assert counts / 1000 == approx(np.array([0.5, 0.5]), abs=0.05)


def test_when_clone_additive_noise_models_with_scipy_distribution_then_clone_has_correct_models():
    org_model = AdditiveNoiseModel(create_linear_regressor(), ScipyDistribution(norm))
    clone_1 = org_model.clone()
    clone_2 = org_model.clone()

    org_model.fit(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
    clone_1.fit(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
    clone_2.fit(np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))

    assert isinstance(clone_1, AdditiveNoiseModel)
    assert isinstance(clone_1.prediction_model, SklearnRegressionModel)
    assert isinstance(clone_1.noise_model, ScipyDistribution)
    assert isinstance(clone_2, AdditiveNoiseModel)
    assert isinstance(clone_2.prediction_model, SklearnRegressionModel)
    assert isinstance(clone_2.noise_model, ScipyDistribution)


def test_given_simple_linear_data_when_fit_post_non_linear_sem_with_invertible_identity_then_returns_expected_results():
    X = np.random.normal(0, 1, 1000)
    N = np.random.normal(0, 0.1, 1000)
    Y = 2 * X + N

    sem_ground_truth = PostNonlinearModel(
        create_linear_regressor_with_given_parameters(coefficients=np.array([2])),
        EmpiricalDistribution(),
        InvertibleIdentityFunction(),
    )

    assert sem_ground_truth.estimate_noise(Y, X).reshape(-1) == approx(N)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([0])).squeeze() == 4
    assert sem_ground_truth.evaluate(np.array([2]), np.array([1])).squeeze() == 4 + 1

    sem_fitted = PostNonlinearModel(create_linear_regressor(), EmpiricalDistribution(), InvertibleIdentityFunction())
    sem_fitted.fit(X, Y)

    assert sem_fitted.prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.05)
    assert np.mean(sem_fitted.draw_samples(np.array([2] * 1000))) == approx(4, abs=0.05)


def test_given_exponential_data_when_fit_post_non_linear_sem_with_invertible_exponential_function_then_returns_expected_results():
    X = np.random.normal(0, 1, 1000)
    N = np.random.normal(0, 0.1, 1000)
    Y = np.exp(2 * X + N)

    sem_ground_truth = PostNonlinearModel(
        create_linear_regressor_with_given_parameters(coefficients=np.array([2])),
        EmpiricalDistribution(),
        InvertibleExponentialFunction(),
    )

    assert sem_ground_truth.estimate_noise(Y, X).reshape(-1) == approx(N)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([0])).squeeze() == np.exp(4)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([1])).squeeze() == np.exp(4 + 1)

    sem_fitted = PostNonlinearModel(create_linear_regressor(), EmpiricalDistribution(), InvertibleExponentialFunction())
    sem_fitted.fit(X, Y)

    assert sem_fitted.prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.05)


def test_given_logarithmic_data_when_fit_post_non_linear_sem_with_invertible_logarithmic_function_then_returns_expected_results():
    X = abs(np.random.normal(0, 1, 1000))
    N = abs(np.random.normal(0, 0.1, 1000))
    Y = np.log(2 * X + N)

    sem_ground_truth = PostNonlinearModel(
        create_linear_regressor_with_given_parameters(coefficients=np.array([2])),
        EmpiricalDistribution(),
        InvertibleLogarithmicFunction(),
    )

    assert sem_ground_truth.estimate_noise(Y, X).reshape(-1) == approx(N)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([0])).squeeze() == np.log(4)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([1])).squeeze() == np.log(4 + 1)

    sem_fitted = PostNonlinearModel(create_linear_regressor(), EmpiricalDistribution(), InvertibleLogarithmicFunction())
    sem_fitted.fit(X, Y)

    assert sem_fitted.prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.05)


@flaky(max_runs=3)
def test_given_discrete_target_data_when_fit_discrete_additive_noise_model_then_behaves_as_expected():
    X = np.random.normal(0, 1, (1000, 2))
    X[X > 3] = 3
    X[X < -3] = -3
    Y = np.round(np.sum(X, axis=1))

    danm = DiscreteAdditiveNoiseModel(create_linear_regressor())
    danm.fit(X, Y)

    test_X = np.random.normal(0, 1, (1000, 2))
    test_X[test_X > 3] = 3
    test_X[test_X < -3] = -3
    test_Y = np.round(np.sum(test_X, axis=1)).reshape(-1)

    assert danm.evaluate(test_X, np.zeros(1000)).reshape(-1) == approx(test_Y, abs=3)
    assert danm.evaluate(test_X, np.zeros(1000)).reshape(-1) == approx(test_Y, abs=3)
    assert is_discrete(danm.draw_samples(test_X))

    assert danm.estimate_noise(np.array([0, 1, 2]), np.array([[-1, 1], [0, 0], [0, 1]])).reshape(-1) == approx(
        np.array([0, 1, 1])
    )

    X = np.array([0.1, 10.5, 20, 30.7, 40.3])
    Y = np.floor(X)  # Y has only 0, 10, 20, 30, 40

    danm = DiscreteAdditiveNoiseModel(create_linear_regressor())
    danm.fit(X, Y)

    assert danm.evaluate(np.array([-100, -32.4, 0.4, 4, 9, 11, 30.1, 101.4, 0.9]), np.zeros(9)) == approx(
        [-100, -32, 0, 4, 9, 11, 30, 101, 1]
    )


def _generate_data_with_categorical_input():
    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(3, 1000).astype(str)
    X2 = []

    for i in range(1000):
        tmp_value = 2 * X0[i]

        if X1[i] == "0":
            tmp_value -= 5
        elif X1[i] == "1":
            tmp_value += 10
        else:
            tmp_value += 5

        X2.append(tmp_value)

    return pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})
