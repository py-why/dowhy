from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx
from scipy import stats
from scipy.stats import norm

from dowhy.gcm import ClassifierFCM, PostNonlinearModel, AdditiveNoiseModel, fit, EmpiricalDistribution, \
    ScipyDistribution, ProbabilisticCausalModel, StructuralCausalModel
from dowhy.gcm.ml import create_linear_regressor, SklearnRegressionModel, create_logistic_regression_classifier, \
    create_linear_regressor_with_given_parameters
from dowhy.gcm.ml.regression import InvertibleIdentityFunction, InvertibleExponentialFunction, \
    InvertibleLogarithmicFunction


def test_fit_causal_graph_using_post_nonlinear_models():
    scm = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1')]))
    scm.set_causal_mechanism('X0', ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = scm.causal_mechanism('X0').draw_samples(1000).squeeze()

    test_data = pd.DataFrame({'X0': X0, 'X1': X0 * 2 + 2 + np.random.normal(0, 0.1, 1000)})

    fit(scm, test_data)

    test_parent_samples = np.array([[1], [2]])
    target_values = test_parent_samples * 2 + 2

    mean_samples = np.zeros(target_values.shape)
    for i in range(100):
        mean_samples += scm.causal_mechanism('X1').draw_samples(test_parent_samples)
    mean_samples /= 100

    assert mean_samples == approx(target_values, abs=0.2)


@flaky(max_runs=5)
def test_fit_causal_graph_using_post_nonlinear_models_with_categorical_features():
    scm = StructuralCausalModel(nx.DiGraph([('X0', 'X2'), ('X1', 'X2')]))
    scm.set_causal_mechanism('X0', ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism('X1', EmpiricalDistribution())
    scm.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(3, 1000).astype(str)
    X2 = []

    for i in range(1000):
        tmp_value = 2 * X0[i]

        if X1[i] == '0':
            tmp_value -= 5
        elif X1[i] == '1':
            tmp_value += 10
        else:
            tmp_value += 5

        X2.append(tmp_value)

    training_data = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2})

    fit(scm, data=training_data)

    assert scm.causal_mechanism('X2').evaluate(np.array([[2, '1']], dtype=object), np.array([0])) \
           == approx(14)

    test_data = training_data.to_numpy()

    assert scm.causal_mechanism('X2').evaluate(test_data[:, :2], np.array([0] * 1)) \
           == approx(test_data[:, 2].astype(float).reshape(-1, 1))


def test_fit_causal_graph_using_additive_noise_model():
    scm = StructuralCausalModel(nx.DiGraph([('X0', 'X1')]))
    scm.set_causal_mechanism('X0', ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = scm.causal_mechanism('X0').draw_samples(1000).squeeze()

    test_data = pd.DataFrame({'X0': X0, 'X1': X0 * 2 + 2 + np.random.normal(0, 0.1, 1000)})

    fit(scm, test_data)

    test_parent_samples = np.array([[1], [2]])
    target_values = test_parent_samples * 2 + 2

    mean_samples = np.zeros(target_values.shape)
    for i in range(100):
        mean_samples += scm.causal_mechanism('X1').draw_samples(test_parent_samples)
    mean_samples /= 100

    assert mean_samples == approx(target_values, abs=0.2)


def test_classifier_sem_throws_error_when_non_string_targets():
    scm = StructuralCausalModel(nx.DiGraph([('X0', 'X1')]))
    scm.set_causal_mechanism('X0', ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism('X1', ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = (X0 > np.median(X0)).astype(int)

    with pytest.raises(ValueError):
        fit(scm, data=(pd.DataFrame({'X0': X0, 'X1': X1})))


def test_classifier_sem_produces_strings():
    scm = StructuralCausalModel(nx.DiGraph([('X0', 'X1')]))
    scm.set_causal_mechanism('X0', ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism('X1', ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = (X0 > np.median(X0)).astype(str)

    training_data = pd.DataFrame({'X0': X0, 'X1': X1})

    fit(scm, training_data)

    for val in scm.causal_mechanism('X1').draw_samples(X0):
        assert isinstance(val[0], str)


@flaky(max_runs=5)
def test_classifier_sem_with_categorical_inputs():
    scm = StructuralCausalModel(nx.DiGraph([('X0', 'X2'), ('X1', 'X2')]))
    scm.set_causal_mechanism('X0', ScipyDistribution(stats.norm, loc=0, scale=1))
    scm.set_causal_mechanism('X1', EmpiricalDistribution())
    scm.set_causal_mechanism('X2', ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(3, 1000).astype(str)
    X2 = []

    for i in range(1000):
        tmp_value = 2 * X0[i]

        if X1[i] == '0':
            tmp_value -= 5
        elif X1[i] == '1':
            tmp_value += 10
        else:
            tmp_value += 5

        X2.append(tmp_value)

    X2 = (X2 > np.median(X2)).astype(str)

    training_data = pd.DataFrame({'X0': X0,
                                  'X1': X1,
                                  'X2': X2})

    fit(scm, training_data)

    x2_fcm = cast(ClassifierFCM, scm.causal_mechanism('X2'))
    assert x2_fcm.estimate_probabilities(np.array([[2, '1']], dtype=object)) == approx(np.array([[0, 1]]), abs=0.01)

    test_data = training_data.to_numpy()
    X2[X2 == 'True'] = 1
    X2[X2 == 'False'] = 0
    assert np.sum(np.argmax(x2_fcm.estimate_probabilities(test_data[:, :2]), axis=1) != X2.astype(int)) < 20

    _, counts = np.unique(x2_fcm.draw_samples(test_data[:, :2]), return_counts=True)
    assert counts / 1000 == approx(np.array([0.5, 0.5]), abs=0.05)


def test_clone_sem_model_with_scipy_distribution():
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


def test_post_non_linear_sem_with_invertible_identity():
    X = np.random.normal(0, 1, 1000)
    N = np.random.normal(0, 0.1, 1000)
    Y = 2 * X + N

    sem_ground_truth = PostNonlinearModel(create_linear_regressor_with_given_parameters(coefficients=np.array([2])),
                                          EmpiricalDistribution(),
                                          InvertibleIdentityFunction())

    assert sem_ground_truth.estimate_noise(Y, X).reshape(-1) == approx(N)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([0])).squeeze() == 4
    assert sem_ground_truth.evaluate(np.array([2]), np.array([1])).squeeze() == 4 + 1

    sem_fitted = PostNonlinearModel(create_linear_regressor(),
                                    EmpiricalDistribution(),
                                    InvertibleIdentityFunction())
    sem_fitted.fit(X, Y)

    assert sem_fitted.prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.05)
    assert np.mean(sem_fitted.draw_samples(np.array([2] * 1000))) == approx(4, abs=0.05)


def test_post_non_linear_sem_with_invertible_exponential():
    X = np.random.normal(0, 1, 1000)
    N = np.random.normal(0, 0.1, 1000)
    Y = np.exp(2 * X + N)

    sem_ground_truth = PostNonlinearModel(create_linear_regressor_with_given_parameters(coefficients=np.array([2])),
                                          EmpiricalDistribution(),
                                          InvertibleExponentialFunction())

    assert sem_ground_truth.estimate_noise(Y, X).reshape(-1) == approx(N)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([0])).squeeze() == np.exp(4)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([1])).squeeze() == np.exp(4 + 1)

    sem_fitted = PostNonlinearModel(create_linear_regressor(),
                                    EmpiricalDistribution(),
                                    InvertibleExponentialFunction())
    sem_fitted.fit(X, Y)

    assert sem_fitted.prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.05)


def test_post_non_linear_sem_with_invertible_logarithmic():
    X = abs(np.random.normal(0, 1, 1000))
    N = abs(np.random.normal(0, 0.1, 1000))
    Y = np.log(2 * X + N)

    sem_ground_truth = PostNonlinearModel(create_linear_regressor_with_given_parameters(coefficients=np.array([2])),
                                          EmpiricalDistribution(),
                                          InvertibleLogarithmicFunction())

    assert sem_ground_truth.estimate_noise(Y, X).reshape(-1) == approx(N)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([0])).squeeze() == np.log(4)
    assert sem_ground_truth.evaluate(np.array([2]), np.array([1])).squeeze() == np.log(4 + 1)

    sem_fitted = PostNonlinearModel(create_linear_regressor(),
                                    EmpiricalDistribution(),
                                    InvertibleLogarithmicFunction())
    sem_fitted.fit(X, Y)

    assert sem_fitted.prediction_model.sklearn_model.coef_ == approx(np.array([2]), abs=0.05)
