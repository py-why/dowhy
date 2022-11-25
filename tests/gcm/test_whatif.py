import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx

from dowhy.gcm import (
    AdditiveNoiseModel,
    ClassifierFCM,
    EmpiricalDistribution,
    InvertibleStructuralCausalModel,
    ProbabilisticCausalModel,
    auto,
    average_causal_effect,
    counterfactual_samples,
    fit,
    interventional_samples,
)
from dowhy.gcm.ml import create_linear_regressor, create_logistic_regression_classifier


def __create_and_fit_simple_probabilistic_causal_model():
    X0 = np.random.uniform(-1, 1, 10000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 10000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 10000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 10000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism("X3", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(causal_model, original_observations)

    return causal_model, original_observations


@flaky(max_runs=3)
def test_interventional_samples_atomic():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    observed_data = pd.DataFrame({"X0": [0], "X1": [1], "X2": [2], "X3": [3]})

    sample = interventional_samples(causal_model, dict(X2=lambda x: np.array(10)), observed_data).to_numpy()
    sample = sample.squeeze()
    assert sample[0] == 0
    assert sample[1] == 1
    assert sample[2] == 10
    assert sample[3] == approx(5, abs=0.2)


@flaky(max_runs=3)
def test_interventional_samples_conditional():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    observed_data = pd.DataFrame({"X0": [0], "X1": [1], "X2": [2], "X3": [3]})

    sample = interventional_samples(causal_model, dict(X2=lambda x: x + 10), observed_data).to_numpy()
    sample = sample.squeeze()
    assert sample[0] == 0
    assert sample[1] == 1
    assert sample[2] == approx(10, abs=0.5)
    assert sample[3] == approx(5, abs=0.25)


@flaky(max_runs=3)
def test_interventional_samples_atomic_draw():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    samples = interventional_samples(causal_model, dict(X2=lambda x: np.array(10)), num_samples_to_draw=10)
    assert samples["X2"].to_numpy() == approx(10, abs=0)


@flaky(max_runs=3)
def test_interventional_samples_conditional_draw():
    causal_model, training_data = __create_and_fit_simple_probabilistic_causal_model()

    samples = interventional_samples(causal_model, dict(X2=lambda x: x + 10), num_samples_to_draw=10)
    assert samples["X2"].to_numpy() == approx(np.mean(training_data["X2"].to_numpy()) + 10, abs=1)


@flaky(max_runs=3)
def test_interventional_samples_with_categorical_variables_draw():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism("X1", EmpiricalDistribution())
    causal_model.set_causal_mechanism("X2", ClassifierFCM(classifier_model=create_logistic_regression_classifier()))
    causal_model.set_causal_mechanism("X3", ClassifierFCM(classifier_model=create_logistic_regression_classifier()))

    X0 = np.random.normal(0, 1, 5000)
    X1 = np.random.choice(2, 5000).astype(str)

    X2 = []
    for (x0, x1) in zip(X0, X1):
        if x1 == "0":
            X2.append(x0 + 2 > 0)
        else:
            X2.append(x0 - 2 > 0)

    X2 = np.array(X2).astype(str)

    X3 = []
    for x2 in X2:
        if x2 == "True":
            X3.append("False")
        else:
            X3.append("True")

    X3 = np.array(X3).astype(str)

    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    fit(causal_model, training_data)

    intervention_results = interventional_samples(
        causal_model, dict(X0=lambda x: 0, X1=lambda x: "0"), num_samples_to_draw=100
    )

    assert intervention_results["X0"].to_numpy() == approx(np.array([0] * 100))
    assert np.all(intervention_results["X1"].to_numpy() == np.array(["0"] * 100))
    assert np.sum(intervention_results["X2"].to_numpy() == "True") > 95
    assert np.sum(intervention_results["X3"].to_numpy() == "False") > 95


@flaky(max_runs=3)
def test_interventional_samples_atomic_multiple_interventions():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    observed_data = pd.DataFrame({"X0": [0], "X1": [1], "X2": [2], "X3": [3]})

    sample = interventional_samples(causal_model, dict(X0=lambda x: 10, X2=lambda x: x + 5), observed_data).to_numpy()
    sample = sample.squeeze()
    assert sample[0] == 10
    assert sample[1] == approx(20, abs=0.3)
    assert sample[2] == approx(10, abs=0.3)
    assert sample[3] == approx(5, abs=0.3)


def test_interventional_samples_raise_error_all_parameter_none():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    with pytest.raises(ValueError):
        interventional_samples(causal_model, dict(X0=lambda x: 10))


def test_interventional_samples_raise_error_both_parameter_given():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    observed_data = pd.DataFrame({"X0": [0], "X1": [1], "X2": [2], "X3": [3]})

    with pytest.raises(ValueError):
        interventional_samples(
            causal_model, dict(X0=lambda x: 10), observed_data=observed_data, num_samples_to_draw=100
        )


def test_counterfactual_samples_with_observed_samples():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    observed_samples = pd.DataFrame({"X0": [1], "X1": [3], "X2": [3], "X3": [4]})

    sample = counterfactual_samples(causal_model, dict(X2=lambda x: 2), observed_data=observed_samples)
    assert sample["X0"].to_numpy().squeeze() == 1
    assert sample["X1"].to_numpy().squeeze() == 3
    assert sample["X2"].to_numpy().squeeze() == 2
    assert sample["X3"].to_numpy().squeeze() == approx(3.5, abs=0.05)


def test_counterfactual_samples_with_noise_samples():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    noise_samples = pd.DataFrame({"X0": [1], "X1": [2], "X2": [3], "X3": [4]})

    sample = counterfactual_samples(causal_model, dict(X2=lambda x: 2), noise_data=noise_samples)
    assert sample["X0"].to_numpy().squeeze() == 1
    assert sample["X1"].to_numpy().squeeze() == approx(4, abs=0.05)
    assert sample["X2"].to_numpy().squeeze() == 2
    assert sample["X3"].to_numpy().squeeze() == approx(5, abs=0.05)


def test_counterfactual_samples_raises_error_all_parameter_none():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    with pytest.raises(ValueError):
        counterfactual_samples(causal_model, dict(X0=lambda x: 10))


def test_counterfactual_samples_raises_error_both_parameter_given():
    causal_model, _ = __create_and_fit_simple_probabilistic_causal_model()

    with pytest.raises(ValueError):
        counterfactual_samples(
            causal_model,
            dict(X0=lambda x: 10),
            observed_data=pd.DataFrame({"X0": [1], "X1": [3], "X2": [3], "X3": [4]}),
            noise_data=pd.DataFrame({"X0": [1], "X1": [4], "X2": [3], "X3": [4]}),
        )


@flaky(max_runs=3)
def test_given_continuous_target_when_estimate_average_causal_effect_then_return_expected_result():
    T = np.random.choice(2, 1000, replace=True)
    X0 = np.random.normal(0, 0.2, 1000) + T
    X1 = np.random.normal(0, 0.2, 1000) + 0.5 * T
    Y = X0 + X1 + np.random.normal(0, 0.1, 1000)

    data = pd.DataFrame(dict(T=T, X0=X0, X1=X1, Y=Y))

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("T", "X0"), ("T", "X1"), ("X0", "Y"), ("X1", "Y")]))
    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)
    fit(causal_model, data)

    assert average_causal_effect(
        causal_model,
        "Y",
        interventions_alternative={"T": lambda x: 1},
        interventions_reference={"T": lambda x: 0},
        num_samples_to_draw=1000,
    ) == approx(1.5, abs=0.1)


@flaky(max_runs=3)
def test_given_binary_target_when_estimate_average_causal_effect_then_return_expected_result():
    T = np.random.choice(2, 1000, replace=True)
    X0 = np.random.normal(0, 0.1, 1000) + T
    X1 = np.random.normal(0, 0.1, 1000) + 0.5 * T
    Y = ((X0 + X1 + np.random.normal(0, 0.1, 1000)) >= 1.5).astype(str)

    data = pd.DataFrame(dict(T=T, X0=X0, X1=X1, Y=Y))

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("T", "X0"), ("T", "X1"), ("X0", "Y"), ("X1", "Y")]))
    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)
    fit(causal_model, data)

    assert average_causal_effect(
        causal_model,
        "Y",
        interventions_alternative={"T": lambda x: 1},
        interventions_reference={"T": lambda x: 0},
        num_samples_to_draw=1000,
    ) == approx(0.5, abs=0.1)
