import networkx as nx
import numpy as np
import pandas as pd
from _pytest.python_api import approx
from flaky import flaky

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    InvertibleStructuralCausalModel,
    StructuralCausalModel,
    fit,
)
from dowhy.gcm._noise import compute_data_from_noise, compute_noise_from_data, get_noise_dependent_function
from dowhy.gcm.auto import assign_causal_mechanisms
from dowhy.gcm.causal_models import PARENTS_DURING_FIT
from dowhy.gcm.ml import (
    create_linear_regressor,
    create_linear_regressor_with_given_parameters,
    create_logistic_regression_classifier,
)
from dowhy.graph import DirectedGraph, get_ordered_predecessors


def test_given_data_with_known_noise_values_when_compute_data_from_noise_then_returns_correct_values():
    N0 = np.random.uniform(-1, 1, 1000)
    N1 = np.random.normal(0, 0.1, 1000)
    N2 = np.random.normal(0, 0.1, 1000)
    N3 = np.random.normal(0, 0.1, 1000)

    X0 = N0
    X1 = 2 * X0 + N1
    X2 = 0.5 * X0 + N2
    X3 = 0.5 * X2 + N3

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    noise_observations = pd.DataFrame({"X0": N0, "X1": N1, "X2": N2, "X3": N3})

    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism(
        "X1", AdditiveNoiseModel(prediction_model=create_linear_regressor_with_given_parameters(np.array([2])))
    )
    causal_model.set_causal_mechanism(
        "X2", AdditiveNoiseModel(prediction_model=create_linear_regressor_with_given_parameters(np.array([0.5])))
    )
    causal_model.set_causal_mechanism(
        "X3", AdditiveNoiseModel(prediction_model=create_linear_regressor_with_given_parameters(np.array([0.5])))
    )

    _persist_parents(causal_model.graph)

    estimated_samples = compute_data_from_noise(causal_model, noise_observations)

    for node in original_observations:
        assert estimated_samples[node].to_numpy() == approx(original_observations[node].to_numpy())


def test_given_data_with_known_noise_values_when_compute_noise_from_data_then_reconstruct_correct_noise_values():
    N0 = np.random.uniform(-1, 1, 1000)
    N1 = np.random.normal(0, 0.1, 1000)
    N2 = np.random.normal(0, 0.1, 1000)
    N3 = np.random.normal(0, 0.1, 1000)

    X0 = N0
    X1 = 2 * X0 + N1
    X2 = 0.5 * X0 + N2
    X3 = 0.5 * X2 + N3

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism(
        "X1", AdditiveNoiseModel(prediction_model=create_linear_regressor_with_given_parameters(np.array([2])))
    )
    causal_model.set_causal_mechanism(
        "X2", AdditiveNoiseModel(prediction_model=create_linear_regressor_with_given_parameters(np.array([0.5])))
    )
    causal_model.set_causal_mechanism(
        "X3", AdditiveNoiseModel(prediction_model=create_linear_regressor_with_given_parameters(np.array([0.5])))
    )

    _persist_parents(causal_model.graph)

    estimated_noise_samples = compute_noise_from_data(causal_model, original_observations)

    assert estimated_noise_samples["X0"].to_numpy() == approx(N0)
    assert estimated_noise_samples["X1"].to_numpy() == approx(N1)
    assert estimated_noise_samples["X2"].to_numpy() == approx(N2)
    assert estimated_noise_samples["X3"].to_numpy() == approx(N3)


@flaky(max_runs=3)
def test_given_continuous_variables_when_get_noise_dependent_function_then_represents_correct_function():
    X0 = np.random.normal(0, 1, 2000)
    X1 = X0 + np.random.normal(0, 0.1, 2000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 2000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 2000)
    data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    assign_causal_mechanisms(causal_model, data)

    fit(causal_model, data)

    fn, parent_order = get_noise_dependent_function(causal_model, "X3")
    input_data = pd.DataFrame(np.array([[0, 0, 0], [0, 0, 2], [1, 0, 0], [1, 2, 0]]), columns=["X0", "X2", "X3"])

    assert set(parent_order) == {"X0", "X2", "X3"}
    assert fn(input_data.to_numpy()) == approx(np.array([0, 2, 0.25, 1.25]), abs=0.1)

    fn, _ = get_noise_dependent_function(causal_model, "X3", approx_prediction_model=create_linear_regressor())
    assert fn(input_data.to_numpy()).reshape(-1) == approx(np.array([0, 2, 0.25, 1.25]), abs=0.1)


@flaky(max_runs=3)
def test_given_continuous_and_categorical_variables_when_get_noise_dependent_function_then_represents_correct_function():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2"), ("X2", "X3")]))

    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.choice(2, 1000).astype(str)

    X2 = []
    for x0, x1 in zip(X0, X1):
        if x1 == "0":
            x = np.random.normal(0, 1)
        else:
            x = np.random.normal(1, 1)

        if x < 0.5:
            X2.append(x0 + 2 > 0)
        else:
            X2.append(x0 - 2 > 0)

    X2 = np.array(X2).astype(str)

    X3 = []
    for x2 in X2:
        if x2 == "True":
            x = np.random.normal(0, 1)
        else:
            x = np.random.normal(1, 1)

        if x < 0.5:
            X3.append("False")
        else:
            X3.append("True")

    X3 = np.array(X3).astype(str)
    data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    assign_causal_mechanisms(causal_model, data)

    fit(causal_model, data)
    fn, parent_order = get_noise_dependent_function(causal_model, "X3")

    assert sorted(parent_order[:2]) == ["X0", "X1"]
    assert parent_order[2:] == ["X2", "X3"]
    assert np.all(
        fn(
            pd.DataFrame({"X0": [0, 0, 0, 0], "X1": ["0", "0", "0", "0"], "X2": [0, 0, 0, 1], "X3": [1, 0, 0.6, 0.6]})[
                parent_order
            ].to_numpy()
        )
        == np.array(["True", "False", "True", "False"])
    )

    fn, parent_order = get_noise_dependent_function(
        causal_model, "X3", approx_prediction_model=create_logistic_regression_classifier()
    )
    assert np.all(
        fn(
            pd.DataFrame({"X0": [0, 0, 0, 0], "X1": ["0", "0", "0", "0"], "X2": [0, 0, 0, 1], "X3": [1, 0, 0.6, 0.6]})[
                parent_order
            ].to_numpy()
        ).reshape(-1)
        == np.array(["True", "False", "True", "False"])
    )


def test_when_get_noise_dependent_function_then_correctly_omits_nodes():
    # Just some random data, since we are only interested in the omitted data.
    data = pd.DataFrame(
        {
            "X0": np.random.normal(0, 1, 10),
            "X1": np.random.normal(0, 1, 10),
            "X2": np.random.normal(0, 1, 10),
            "X3": np.random.normal(0, 1, 10),
            "X4": np.random.normal(0, 1, 10),
            "X5": np.random.normal(0, 1, 10),
            "X6": np.random.normal(0, 1, 10),
            "X7": np.random.normal(0, 1, 10),
        }
    )

    causal_model = StructuralCausalModel(
        nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X3", "X2"), ("X4", "X5"), ("X6", "X5")])
    )
    causal_model.graph.add_node("X7")
    assign_causal_mechanisms(causal_model, data)

    fit(causal_model, data)

    _, parent_order = get_noise_dependent_function(causal_model, "X2")
    assert set(parent_order) == {"X0", "X1", "X2", "X3"}
    assert parent_order.index("X1") > parent_order.index("X0")
    assert parent_order.index("X2") > parent_order.index("X0")
    assert parent_order.index("X2") > parent_order.index("X1")
    assert parent_order.index("X2") > parent_order.index("X3")


def test_given_nodes_names_are_ints_when_calling_noise_dependent_function_then_does_not_raise_key_error_exception():
    causal_model = StructuralCausalModel(nx.DiGraph([(1, 2)]))
    data = pd.DataFrame({1: np.random.normal(0, 1, 10), 2: np.random.normal(0, 1, 10)})
    assign_causal_mechanisms(causal_model, data)

    fit(causal_model, data)

    noise_dependent_function, _ = get_noise_dependent_function(causal_model, 1)

    noise_dependent_function(np.array([[1]]))


def test_given_dataframe_with_object_dtype_using_pandas_v2_when_compute_data_from_noise_then_does_not_raise_error():
    X0 = np.random.choice(2, 10)
    X1 = X0 + np.random.normal(0, 1, 10)

    data = pd.DataFrame({"X0": X0, "X1": X1}, dtype=object)

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1")]))
    assign_causal_mechanisms(causal_model, data)
    fit(causal_model, data)

    # This caused an error before with pandas > 2.0
    compute_noise_from_data(causal_model, data.iloc[0:1])


def _persist_parents(graph: DirectedGraph):
    for node in graph.nodes:
        graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(graph, node)
