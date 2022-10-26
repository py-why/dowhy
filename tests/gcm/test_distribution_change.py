import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    ProbabilisticCausalModel,
    distribution_change,
    distribution_change_of_graphs,
    fit,
)
from dowhy.gcm.auto import AssignmentQuality
from dowhy.gcm.ml import create_linear_regressor
from dowhy.gcm.shapley import ShapleyConfig


@flaky(max_runs=5)
def test_given_two_data_sets_with_different_mechanisms_when_evaluate_distribution_change_then_returns_expected_result():
    original_observations, outlier_observations = _generate_data()

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model)

    results = distribution_change(
        causal_model, original_observations, outlier_observations, "X3", shapley_config=ShapleyConfig(n_jobs=1)
    )

    assert results["X3"] > results["X2"]
    assert results["X2"] > results["X0"]
    assert "X1" not in results
    assert results["X0"] == approx(0, abs=0.15)


@flaky(max_runs=5)
def test_given_two_graphs_fitted_on_data_sets_with_different_mechanisms_when_evaluate_distribution_change_of_graphs_then_returns_expected_result():
    original_observations, outlier_observations = _generate_data()

    causal_model_old = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model_old)
    causal_model_new = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model_new)

    fit(causal_model_old, original_observations)
    fit(causal_model_new, outlier_observations)

    results = distribution_change_of_graphs(causal_model_old, causal_model_new, "X3")

    assert results["X3"] > results["X2"] > results["X0"]
    assert "X1" not in results
    assert results["X0"] == approx(0, abs=0.1)


@flaky(max_runs=5)
def test_when_using_distribution_change_without_fdrc_then_returns_valid_results():
    original_observations, outlier_observations = _generate_data()

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model)

    results = distribution_change(
        causal_model, original_observations, outlier_observations, "X3", mechanism_change_test_fdr_control_method=None
    )

    assert results["X3"] > results["X2"] > results["X0"]
    assert "X1" not in results


@flaky(max_runs=5)
def test_when_using_distribution_change_with_return_additional_info_then_returns_additional_info():
    original_observations, outlier_observations = _generate_data()

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    _assign_causal_mechanisms(causal_model)

    attributions, mechanism_changes, causal_model_old, causal_model_new = distribution_change(
        causal_model,
        original_observations,
        outlier_observations,
        "X3",
        mechanism_change_test_fdr_control_method=None,
        return_additional_info=True,
    )

    assert attributions["X3"] > attributions["X2"] > attributions["X0"]
    assert "X1" not in attributions

    assert not mechanism_changes["X0"]
    assert mechanism_changes["X2"]
    assert mechanism_changes["X3"]

    assert "X1" not in causal_model_old.graph.nodes
    assert "X1" not in causal_model_new.graph.nodes

    for node in causal_model.graph.nodes:
        if node == "X1":
            continue
        assert type(causal_model_old.causal_mechanism(node)) == type(causal_model_new.causal_mechanism(node))


@flaky(max_runs=3)
def test_given_non_linear_data_when_using_distribution_change_with_mean_difference_then_returns_expected_results():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = np.exp(0.5 * X0) + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 4 + np.exp(0.5 * X0) + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))

    attributions = distribution_change(
        causal_model,
        original_observations,
        outlier_observations,
        "X3",
        difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x),
        auto_assignment_quality=AssignmentQuality.GOOD,
    )

    assert attributions["X0"] == approx(0, abs=0.05)
    assert attributions["X2"] == approx(2, abs=0.05)
    assert attributions["X3"] == approx(0, abs=0.05)
    assert "X1" not in attributions


def _assign_causal_mechanisms(causal_model):
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism("X1", AdditiveNoiseModel(create_linear_regressor()))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(create_linear_regressor()))
    causal_model.set_causal_mechanism("X3", AdditiveNoiseModel(create_linear_regressor()))


def _generate_data():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 3 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    return original_observations, outlier_observations
