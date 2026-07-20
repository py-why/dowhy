import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky
from pytest import approx

from dowhy.gcm import AdditiveNoiseModel, EmpiricalDistribution, ProbabilisticCausalModel, draw_samples, fit
from dowhy.gcm.ml import create_linear_regressor
from dowhy.graph import is_root_node


@flaky(max_runs=2)
def test_given_probabilistic_causal_model_when_samples_are_drawn_then_they_have_correct_mean_and_standard_deviation():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    scm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    scm.set_causal_mechanism("X0", EmpiricalDistribution())
    scm.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    scm.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    scm.set_causal_mechanism("X3", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(scm, original_observations)

    generated_samples = draw_samples(scm, 1000)

    assert np.mean(generated_samples["X0"]) == approx(np.mean(X0), abs=0.1)
    assert np.std(generated_samples["X0"]) == approx(np.std(X0), abs=0.2)
    assert np.mean(generated_samples["X1"]) == approx(np.mean(X1), abs=0.1)
    assert np.std(generated_samples["X1"]) == approx(np.std(X1), abs=0.2)
    assert np.mean(generated_samples["X2"]) == approx(np.mean(X2), abs=0.1)
    assert np.std(generated_samples["X2"]) == approx(np.std(X2), abs=0.2)
    assert np.mean(generated_samples["X3"]) == approx(np.mean(X3), abs=0.1)
    assert np.std(generated_samples["X3"]) == approx(np.std(X3), abs=0.2)


def test_when_trying_to_set_causal_mechanism_of_non_existing_node_then_raises_error():
    with pytest.raises(ValueError):
        ProbabilisticCausalModel().set_causal_mechanism("X0", EmpiricalDistribution())


def test_given_a_directed_graph_when_checking_if_a_node_is_root_then_returns_true_for_root_nodes_and_false_for_non_root_nodes():
    graph = nx.DiGraph([("X", "Z"), ("Y", "Z")])
    assert is_root_node(graph, "X") == True
    assert is_root_node(graph, "Y") == True
    assert is_root_node(graph, "Z") == False


def test_when_set_and_get_causal_model_then_the_set_model_is_returned():
    causal_dag = nx.DiGraph()
    causal_dag.add_node("X0")
    causal_model = ProbabilisticCausalModel(causal_dag)

    mdl = EmpiricalDistribution()

    causal_model.set_causal_mechanism("X0", mdl)

    assert causal_model.causal_mechanism("X0") == mdl


def test_given_dag_with_disconnected_node_when_node_connected_subgraph_view_then_excludes_disconnected_node():
    """node_connected_subgraph_view should return only ancestors of the target node (and the node itself)."""
    from dowhy.graph import node_connected_subgraph_view

    # X0 -> X1 -> X2; D is disconnected
    g = nx.DiGraph([("X0", "X1"), ("X1", "X2")])
    g.add_node("D")

    sub = node_connected_subgraph_view(g, "X2")
    assert set(sub.nodes) == {"X0", "X1", "X2"}
    assert "D" not in sub.nodes


def test_given_dag_when_node_connected_subgraph_view_on_root_then_returns_only_root():
    """For a root node with no ancestors, the view contains only that node."""
    from dowhy.graph import node_connected_subgraph_view

    g = nx.DiGraph([("X0", "X1"), ("X1", "X2")])
    sub = node_connected_subgraph_view(g, "X0")
    assert set(sub.nodes) == {"X0"}


def test_given_dag_when_node_connected_subgraph_view_then_preserves_edges():
    """Edges within the connected subgraph should be preserved."""
    from dowhy.graph import node_connected_subgraph_view

    g = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C"), ("D", "E")])
    sub = node_connected_subgraph_view(g, "C")
    assert set(sub.nodes) == {"A", "B", "C"}
    assert set(sub.edges) == {("A", "B"), ("B", "C"), ("A", "C")}
