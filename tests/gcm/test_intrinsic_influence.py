from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from dowhy.gcm import (
    AdditiveNoiseModel,
    ClassifierFCM,
    EmpiricalDistribution,
    InvertibleStructuralCausalModel,
    ScipyDistribution,
    StructuralCausalModel,
    auto,
    fit,
    intrinsic_causal_influence,
)
from dowhy.gcm._noise import noise_samples_of_ancestors
from dowhy.gcm.influence import intrinsic_causal_influence_sample
from dowhy.gcm.ml import (
    SklearnClassificationModel,
    create_hist_gradient_boost_regressor,
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
)
from dowhy.gcm.shapley import ShapleyApproximationMethods, ShapleyConfig
from dowhy.gcm.uncertainty import estimate_entropy_of_probabilities, estimate_variance
from dowhy.gcm.util.general import apply_one_hot_encoding, fit_one_hot_encoders
from dowhy.graph import node_connected_subgraph_view
from tests.gcm.test_noise import _persist_parents


@flaky(max_runs=3)
def test_given_linear_data_when_estimate_intrinsic_causal_influence_with_variance_then_returns_expected_results():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    X0 = np.random.normal(0, 1, 5000)
    X1 = X0 + np.random.normal(0, 0.001, 5000)
    X2 = X1 + np.random.normal(0, 2, 5000)
    X3 = X2 + np.random.normal(0, 1, 5000)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})
    auto.assign_causal_mechanisms(causal_model, training_data)

    fit(causal_model, training_data)

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx", num_samples_baseline=2000)
    assert iccs["X0"] == approx(1, abs=0.3)
    assert iccs["X1"] == approx(0, abs=0.05)
    assert iccs["X2"] == approx(4, abs=0.5)
    assert iccs["X3"] == approx(1, abs=0.25)
    assert np.sum([iccs[key] for key in iccs]) == approx(estimate_variance(X3), abs=0.5)

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact", num_samples_baseline=2000)
    assert iccs["X0"] == approx(1, abs=0.3)
    assert iccs["X1"] == approx(0, abs=0.05)
    assert iccs["X2"] == approx(4, abs=0.5)
    assert iccs["X3"] == approx(1, abs=0.25)
    assert np.sum([iccs[key] for key in iccs]) == approx(estimate_variance(X3), abs=0.5)


@flaky(max_runs=3)
def test_given_categorical_data_when_estimate_intrinsic_causal_influence_then_returns_expected_results():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    N1 = np.random.normal(0, 0.001, 10000)
    N2 = np.random.normal(0, 2, 10000)
    N3 = np.random.normal(0, 1, 10000)
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + N1
    X2 = X1 + N2
    X3 = ((X2 + N3) >= 0).astype(str)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    log_mdl = LogisticRegression(max_iter=1000)
    log_mdl.fit(np.column_stack([X0, N1, N2, N3]), X3)

    expected_output_full_subset = -estimate_entropy_of_probabilities(
        log_mdl.predict_proba(np.column_stack([X0, N1, N2, N3]))
    )
    expected_output_empty_subset = -estimate_entropy_of_probabilities(np.array([[0.5, 0.5]]))

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.14, abs=0.05)
    assert iccs["X1"] == approx(0.01, abs=0.05)
    assert iccs["X2"] == approx(0.38, abs=0.05)
    assert iccs["X3"] == approx(0.12, abs=0.05)

    assert np.sum([iccs[key] for key in iccs]) == approx(
        expected_output_full_subset - expected_output_empty_subset, abs=0.05
    )

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.14, abs=0.05)
    assert iccs["X1"] == approx(0.01, abs=0.05)
    assert iccs["X2"] == approx(0.38, abs=0.05)
    assert iccs["X3"] == approx(0.12, abs=0.05)

    assert np.sum([iccs[key] for key in iccs]) == approx(-expected_output_empty_subset, abs=0.05)


@flaky(max_runs=3)
def test_given_more_complex_categorical_data_when_estimate_intrinsic_causal_influence_then_returns_expected_results():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))
    N1 = np.random.normal(0, 0.001, 10000)
    N2 = np.random.normal(0, 2, 10000)
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + N1
    X2 = X1 + N2

    X3 = []
    for i in range(10000):
        rand_val = np.random.choice(3, 1).squeeze()
        if rand_val == 0:
            X3.append(X2[i] + np.random.normal(-5, 0.5, 1))
        elif rand_val == 1:
            X3.append(X2[i] + np.random.normal(10, 0.5, 1))
        else:
            X3.append(X2[i] + np.random.normal(5, 0.5, 1))

    X3 = np.array(X3)
    X3 = (X3 > np.median(X3)).astype(str)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3.reshape(-1)})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    data_samples, noise_samples = noise_samples_of_ancestors(
        StructuralCausalModel(node_connected_subgraph_view(causal_model.graph, "X3")), "X3", 100000
    )
    X = apply_one_hot_encoding(noise_samples.to_numpy(), fit_one_hot_encoders(noise_samples.to_numpy()))
    log_mdl = LogisticRegression()
    log_mdl.fit(X, data_samples["X3"].to_numpy())

    expected_output_full_subset = -estimate_entropy_of_probabilities(log_mdl.predict_proba(X))
    expected_output_empty_subset = -estimate_entropy_of_probabilities(np.array([[0.5, 0.5]]))

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="approx", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.05, abs=0.05)
    assert iccs["X1"] == approx(0.03, abs=0.05)
    assert iccs["X2"] == approx(0.1, abs=0.05)
    assert iccs["X3"] == approx(0.5, abs=0.1)

    assert np.sum([iccs[key] for key in iccs]) == approx(
        expected_output_full_subset - expected_output_empty_subset, abs=0.1
    )

    iccs = intrinsic_causal_influence(causal_model, "X3", prediction_model="exact", num_samples_baseline=100)
    assert iccs["X0"] == approx(0.05, abs=0.05)
    assert iccs["X1"] == approx(0.03, abs=0.05)
    assert iccs["X2"] == approx(0.1, abs=0.05)
    assert iccs["X3"] == approx(0.5, abs=0.1)

    assert np.sum([iccs[key] for key in iccs]) == approx(-expected_output_empty_subset, abs=0.1)


@pytest.mark.parametrize("prediction_model", ["approx", "exact"])
def test_given_only_categorical_data_when_estimate_icc_then_does_not_fail(monkeypatch, prediction_model):
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    # Enumerate equally likely noise intervals: X1 copies X0; X2 and X3 copy their parent with probability 3/4.
    # Midpoints avoid ClassifierFCM's probability boundaries and give exact finite-sample conditional probabilities.
    noise_samples = pd.DataFrame(
        product(
            ["False", "True"],
            [0.125, 0.375, 0.625, 0.875],
            [0.125, 0.375, 0.625, 0.875],
            [0.125, 0.375, 0.625, 0.875],
        ),
        columns=["X0", "X1", "X2", "X3"],
    )
    x0 = noise_samples["X0"].to_numpy() == "True"
    x2 = noise_samples["X2"].to_numpy() > np.where(x0, 0.25, 0.75)
    x3 = noise_samples["X3"].to_numpy() > np.where(x2, 0.25, 0.75)
    training_data = pd.DataFrame(
        {
            "X0": x0.astype(str),
            "X1": x0.astype(str),
            "X2": x2.astype(str),
            "X3": x3.astype(str),
        }
    )
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    for node in ["X1", "X2", "X3"]:
        causal_model.set_causal_mechanism(
            node,
            ClassifierFCM(SklearnClassificationModel(DecisionTreeClassifier(random_state=0))),
        )
    fit(causal_model, training_data)

    def balanced_samples(model, target_node, num_samples):
        assert target_node == "X3"
        assert set(model.graph.nodes) == set(training_data.columns)
        assert num_samples % len(noise_samples) == 0
        repeats = num_samples // len(noise_samples)
        return (
            pd.concat([training_data] * repeats, ignore_index=True),
            pd.concat([noise_samples] * repeats, ignore_index=True),
        )

    monkeypatch.setattr("dowhy.gcm.influence.noise_samples_of_ancestors", balanced_samples)
    if prediction_model == "approx":
        # Keep real classifier fitting/prediction, but isolate attribution from automatic model selection.
        monkeypatch.setattr(
            auto,
            "select_model",
            lambda *args: (
                SklearnClassificationModel(DecisionTreeClassifier(random_state=0)),
                [],
            ),
        )

    iccs = intrinsic_causal_influence(
        causal_model,
        "X3",
        prediction_model=prediction_model,
        num_training_samples=len(noise_samples),
        num_samples_randomization=len(noise_samples),
        num_samples_baseline=len(noise_samples),
        shapley_config=ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT, n_jobs=1),
    )

    # The three active noises have conditional entropies H(Y|N0)=h(3/8),
    # H(Y|N2)=(log(2)+h(1/4))/2, H(Y|N3)=log(2)/2,
    # H(Y|N0,N2)=h(1/4), H(Y|N0,N3)=h(1/4)/2, H(Y|N2,N3)=log(2)/4.
    # Averaging the three-noise marginal contributions gives these independent analytic Shapley values.
    entropy = np.log(2)
    h_quarter = stats.entropy([0.25, 0.75])
    h_three_eighths = stats.entropy([0.375, 0.625])
    expected_x0 = 7 * entropy / 12 - h_three_eighths / 3 - h_quarter / 6
    expected_x2 = (1.25 * entropy + h_three_eighths - h_quarter) / 6
    assert iccs == approx(
        {
            "X0": expected_x0,
            "X1": 0,
            "X2": expected_x2,
            "X3": entropy - expected_x0 - expected_x2,
        },
        abs=1e-10,
    )

    assert sum(iccs.values()) == approx(entropy, abs=1e-10)


def test_when_calling_intrinsic_causal_influence_then_the_shape_of_inputs_in_the_attribution_function_should_be_equal():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1")]))
    X0 = np.random.normal(0, 1, 10000)
    X1 = X0 + np.random.normal(0, 0.001, 10000)
    training_data = pd.DataFrame({"X0": X0, "X1": X1})
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model, training_data)

    def my_attr_func(X, Y):
        assert np.all(X.shape == Y.shape)
        return 0

    intrinsic_causal_influence(causal_model, "X1", attribution_func=my_attr_func)


@flaky(max_runs=3)
def test_given_linear_gaussian_data_when_estimate_sample_wise_intrinsic_causal_influence_then_returns_expected_values():
    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism(
        "X1",
        AdditiveNoiseModel(
            create_linear_regressor_with_given_parameters(np.array([2])), ScipyDistribution(stats.norm, loc=0, scale=1)
        ),
    )
    causal_model.set_causal_mechanism(
        "X2",
        AdditiveNoiseModel(
            create_linear_regressor_with_given_parameters(np.array([1])), ScipyDistribution(stats.norm, loc=0, scale=1)
        ),
    )
    causal_model.set_causal_mechanism(
        "X3",
        AdditiveNoiseModel(
            create_linear_regressor_with_given_parameters(np.array([1])), ScipyDistribution(stats.norm, loc=0, scale=1)
        ),
    )
    _persist_parents(causal_model.graph)

    shapley_values = intrinsic_causal_influence_sample(
        causal_model, "X3", pd.DataFrame({"X0": [0, 1], "X1": [0.5, 2.5], "X2": [1.5, 4.5], "X3": [1.5, 5.5]})
    )

    assert shapley_values[0][("X0", "X3")] == approx(0, abs=0.1)
    assert shapley_values[0][("X1", "X3")] == approx(0.5, abs=0.1)
    assert shapley_values[0][("X2", "X3")] == approx(1, abs=0.1)
    assert shapley_values[0][("X3", "X3")] == approx(0, abs=0.1)

    assert shapley_values[1][("X0", "X3")] == approx(2, abs=0.1)
    assert shapley_values[1][("X1", "X3")] == approx(0.5, abs=0.1)
    assert shapley_values[1][("X2", "X3")] == approx(2, abs=0.1)
    assert shapley_values[1][("X3", "X3")] == approx(1, abs=0.1)


@flaky(max_runs=3)
def test_given_linear_gaussian_data_when_estimate_sample_wise_intrinsic_causal_influence_with_a_pre_defined_model_then_returns_expected_values():
    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))

    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism(
        "X1",
        AdditiveNoiseModel(
            create_linear_regressor_with_given_parameters(np.array([2])), ScipyDistribution(stats.norm, loc=0, scale=1)
        ),
    )
    causal_model.set_causal_mechanism(
        "X2",
        AdditiveNoiseModel(
            create_linear_regressor_with_given_parameters(np.array([1])), ScipyDistribution(stats.norm, loc=0, scale=1)
        ),
    )
    causal_model.set_causal_mechanism(
        "X3",
        AdditiveNoiseModel(
            create_linear_regressor_with_given_parameters(np.array([1])), ScipyDistribution(stats.norm, loc=0, scale=1)
        ),
    )
    _persist_parents(causal_model.graph)

    shapley_values = intrinsic_causal_influence_sample(
        causal_model,
        "X3",
        pd.DataFrame({"X0": [0, 1], "X1": [0.5, 2.5], "X2": [1.5, 4.5], "X3": [1.5, 5.5]}),
        prediction_model=create_linear_regressor(),
    )

    assert shapley_values[0][("X0", "X3")] == approx(0, abs=0.15)
    assert shapley_values[0][("X1", "X3")] == approx(0.5, abs=0.15)
    assert shapley_values[0][("X2", "X3")] == approx(1, abs=0.15)
    assert shapley_values[0][("X3", "X3")] == approx(0, abs=0.15)

    assert shapley_values[1][("X0", "X3")] == approx(2, abs=0.15)
    assert shapley_values[1][("X1", "X3")] == approx(0.5, abs=0.15)
    assert shapley_values[1][("X2", "X3")] == approx(2, abs=0.15)
    assert shapley_values[1][("X3", "X3")] == approx(1, abs=0.15)
