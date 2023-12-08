import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx
from scipy import stats

from dowhy.gcm import (
    AdditiveNoiseModel,
    ClassifierFCM,
    ProbabilisticCausalModel,
    ScipyDistribution,
    arrow_strength,
    fit,
)
from dowhy.gcm.auto import assign_causal_mechanisms
from dowhy.gcm.divergence import estimate_kl_divergence_continuous_knn
from dowhy.gcm.influence import arrow_strength_of_model
from dowhy.gcm.ml import (
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
    create_logistic_regression_classifier,
)


@pytest.fixture
def preserve_random_generator_state():
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    yield
    np.random.set_state(numpy_state)
    random.setstate(random_state)


@flaky(max_runs=5)
def test_given_kl_divergence_attribution_func_when_estimate_arrow_strength_then_returns_expected_results():
    causal_strengths = arrow_strength(
        _create_causal_model(), "X2", difference_estimation_func=estimate_kl_divergence_continuous_knn
    )

    assert causal_strengths[("X0", "X2")] == approx(1.2, abs=0.2)
    assert causal_strengths[("X1", "X2")] == approx(0.3, abs=0.1)


@flaky(max_runs=5)
def test_given_continuous_data_with_default_attribution_func_when_estimate_arrow_strength_then_returns_expected_results():
    causal_strengths = arrow_strength(_create_causal_model(), "X2")

    # By default, the strength is measure with respect to the variance.
    assert causal_strengths[("X0", "X2")] == approx(9, abs=0.5)
    assert causal_strengths[("X1", "X2")] == approx(1, abs=0.2)


@flaky(max_runs=3)
def test_given_gcm_with_misspecified_mechanism_when_evaluate_arrow_strength_with_observational_data_then_gives_expected_results():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X1", "X2"), ("X0", "X2")]))
    # Here, we misspecified the mechanism on purpose by setting scale to 1 instead of 2.
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X1", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = np.random.normal(0, 2, 2000)  # The standard deviation in the data is actually 2.
    X1 = np.random.normal(0, 1, 2000)

    test_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X0 + X1 + np.random.normal(0, 0.2, X0.shape[0])})
    fit(causal_model, test_data)

    # If we provide the observational data here, we can mitigate the misspecification of the causal mechanism.
    causal_strengths = arrow_strength(
        causal_model, "X2", parent_samples=test_data, difference_estimation_func=lambda x, y: np.var(y) - np.var(x)
    )
    assert causal_strengths[("X0", "X2")] == approx(4, abs=0.5)
    assert causal_strengths[("X1", "X2")] == approx(1, abs=0.1)


@flaky(max_runs=5)
def test_given_categorical_target_node_when_estimate_arrow_strength_of_model_classifier_then_returns_expected_result():
    X = np.random.random((1000, 5))
    Y = []

    for n in X:
        if (n[0] + n[1] + np.random.random(1)) > 1.5:
            Y.append(0)
        else:
            Y.append(1)

    classifier_sem = ClassifierFCM(create_logistic_regression_classifier())
    classifier_sem.fit(X, np.array(Y).astype(str))

    assert arrow_strength_of_model(classifier_sem, X) == approx(np.array([0.3, 0.3, 0, 0, 0]), abs=0.1)


def test_given_fixed_random_seed_when_estimate_arrow_strength_then_return_deterministic_result(
    preserve_random_generator_state,
):
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X1", "X2"), ("X0", "X2")]))
    causal_model.set_causal_mechanism("X1", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.normal(0, 1, 1000)

    test_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": 3 * X0 + X1 + np.random.normal(0, 0.2, X0.shape[0])})
    fit(causal_model, test_data)

    causal_strengths_1 = arrow_strength(causal_model, "X2", max_num_runs=5, n_jobs=-1)
    causal_strengths_2 = arrow_strength(causal_model, "X2", max_num_runs=5, n_jobs=-1)

    assert causal_strengths_1[("X0", "X2")] != causal_strengths_2[("X0", "X2")]
    assert causal_strengths_1[("X1", "X2")] != causal_strengths_2[("X1", "X2")]

    np.random.seed(0)
    causal_strengths_1 = arrow_strength(causal_model, "X2", max_num_runs=5, n_jobs=-1)
    np.random.seed(0)
    causal_strengths_2 = arrow_strength(causal_model, "X2", max_num_runs=5, n_jobs=-1)

    assert causal_strengths_1[("X0", "X2")] == causal_strengths_2[("X0", "X2")]
    assert causal_strengths_1[("X1", "X2")] == causal_strengths_2[("X1", "X2")]


@flaky(max_runs=3)
def test_given_misspecified_graph_when_estimating_direct_arrow_strength_with_observed_data_then_returns_correct_result():
    Z = np.random.normal(0, 1, 1000)
    X0 = Z + np.random.normal(0, 1, 1000)
    X1 = Z + 2 * X0 + np.random.normal(0, 1, 1000)
    X2 = X0 + X1

    data = pd.DataFrame({"Z": Z, "X0": X0, "X1": X1, "X2": X2})

    # Missing connection between X0 and X1.
    # For X0 and X1, we set the ground truth noise to further emphasize the misspecification. The inferred noise of X1
    # would otherwise have a dependency with Z due to the missing connection with X0.
    causal_model_without = ProbabilisticCausalModel(nx.DiGraph([("Z", "X0"), ("Z", "X1"), ("X0", "X2"), ("X1", "X2")]))
    causal_model_without.set_causal_mechanism(
        "X0", AdditiveNoiseModel(create_linear_regressor(), ScipyDistribution(stats.norm, loc=0, scale=1))
    )
    causal_model_without.set_causal_mechanism(
        "X1", AdditiveNoiseModel(create_linear_regressor(), ScipyDistribution(stats.norm, loc=0, scale=1))
    )
    assign_causal_mechanisms(causal_model_without, data)
    fit(causal_model_without, data)

    # Modelling connection between X0 and X1 explicitly.
    causal_model_with = ProbabilisticCausalModel(
        nx.DiGraph([("Z", "X0"), ("Z", "X1"), ("X0", "X1"), ("X0", "X2"), ("X1", "X2")])
    )
    causal_model_with.set_causal_mechanism(
        "X0", AdditiveNoiseModel(create_linear_regressor(), ScipyDistribution(stats.norm, loc=0, scale=1))
    )
    causal_model_with.set_causal_mechanism(
        "X1", AdditiveNoiseModel(create_linear_regressor(), ScipyDistribution(stats.norm, loc=0, scale=1))
    )
    assign_causal_mechanisms(causal_model_with, data, override_models=False)
    fit(causal_model_with, data)

    strength_missing_edge = arrow_strength(causal_model_without, "X2", parent_samples=data)
    strength_with_edge = arrow_strength(causal_model_with, "X2")

    assert strength_missing_edge[("X0", "X2")] == approx(strength_with_edge[("X0", "X2")], abs=0.2)
    assert strength_missing_edge[("X1", "X2")] == approx(strength_with_edge[("X1", "X2")], abs=1)


@flaky(max_runs=3)
def test_given_gcm_with_misspecified_mechanism_when_evaluate_arrow_strength_with_observational_data_then_gives_expected_results():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X1", "X2"), ("X0", "X2")]))
    # Here, we misspecify the mechanism on purpose by setting scale to 1 instead of 2.
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X1", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = np.random.normal(0, 2, 2000)  # The standard deviation in the data is actually 2.
    X1 = np.random.normal(0, 1, 2000)

    test_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X0 + X1 + np.random.normal(0, 0.2, X0.shape[0])})
    fit(causal_model, test_data)

    # If we provide the observational data here, we can mitigate the misspecification of the causal mechanism.
    causal_strengths = arrow_strength(
        causal_model, "X2", parent_samples=test_data, difference_estimation_func=lambda x, y: np.var(y) - np.var(x)
    )
    assert causal_strengths[("X0", "X2")] == approx(4, abs=0.5)
    assert causal_strengths[("X1", "X2")] == approx(1, abs=0.1)


def test_given_less_samples_than_num_conditionals_specified_when_evaluate_arrow_strength_with_observational_data_then_does_not_throw_error():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1")]))
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = np.random.normal(0, 1, 10)
    X1 = np.random.normal(0, 1, 10)

    test_data = pd.DataFrame({"X0": X0, "X1": X1})
    fit(causal_model, test_data)

    arrow_strength(causal_model, "X1", parent_samples=test_data, num_samples_conditional=10000)


def _create_causal_model():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X1", "X2"), ("X0", "X2")]))
    causal_model.set_causal_mechanism("X1", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism(
        "X2",
        AdditiveNoiseModel(
            prediction_model=create_linear_regressor_with_given_parameters([3, 1]),
            noise_model=ScipyDistribution(stats.norm, loc=0, scale=1),
        ),
    )

    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.normal(0, 1, 1000)

    test_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": 3 * X0 + X1 + np.random.normal(0, 1, X0.shape[0])})
    fit(causal_model, test_data)

    return causal_model
