"""The backdoor optimization must preserve causal adjustment validity."""

import itertools
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_identifier import auto_identifier


def identify(graph, observed=None):
    return identify_effect_auto(
        graph,
        ["X"],
        ["Y"],
        list(graph) if observed is None else observed,
        EstimandType.NONPARAMETRIC_ATE,
        optimize_backdoor=True,
    )


def valid_backdoor(graph, adjustment):
    """Independent path oracle: exclude descendants and block every backdoor path."""
    adjustment = set(adjustment)
    if adjustment & (nx.descendants(graph, "X") | {"X", "Y"}):
        return False
    activated_colliders = adjustment.copy()
    for node in adjustment:
        activated_colliders.update(nx.ancestors(graph, node))
    for path in nx.all_simple_paths(graph.to_undirected(), "X", "Y"):
        if not graph.has_edge(path[1], "X"):
            continue
        blocked = False
        for previous, node, following in zip(path, path[1:], path[2:]):
            collider = graph.has_edge(previous, node) and graph.has_edge(following, node)
            if (collider and node not in activated_colliders) or (not collider and node in adjustment):
                blocked = True
                break
        if not blocked:
            return False
    return True


@pytest.mark.parametrize("order", [("A", "B", "C", "X", "Y"), ("X", "Y", "A", "B", "C"), ("A", "C", "B", "X", "Y")])
def test_optimized_backdoor_preserves_total_effect(order):
    graph = nx.DiGraph()
    graph.add_nodes_from(order)
    graph.add_edges_from([("A", "C"), ("A", "X"), ("A", "B"), ("B", "Y"), ("C", "B"), ("X", "B")])
    estimand = identify(graph)
    assert estimand.estimands["backdoor"] is not None
    adjustment = estimand.get_backdoor_variables()
    assert valid_backdoor(graph, adjustment)

    # Independent Gaussian SCM with unit-variance noises: C=A+eC, X=A+eX,
    # B=A+C+X+eB, Y=B+eY. Its interventional X -> Y effect is exactly one.
    noises = np.eye(5)
    coefficients = {"A": noises[0]}
    coefficients["C"] = coefficients["A"] + noises[1]
    coefficients["X"] = coefficients["A"] + noises[2]
    coefficients["B"] = coefficients["A"] + coefficients["C"] + coefficients["X"] + noises[3]
    coefficients["Y"] = coefficients["B"] + noises[4]
    regressors = np.stack([coefficients["X"]] + [coefficients[node] for node in adjustment])
    population_effect = np.linalg.solve(regressors @ regressors.T, regressors @ coefficients["Y"])[0]
    assert population_effect == pytest.approx(1.0)


def test_unobserved_optimized_candidate_falls_back_to_observed_adjustment():
    graph = nx.DiGraph([("U", "X"), ("U", "Z1"), ("U", "Z2"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")])
    observed = ["X", "Y", "Z1", "Z2"]
    estimand = identify(graph, observed)
    assert estimand.estimands["backdoor"] is not None
    assert set(estimand.get_backdoor_variables()) == {"Z1", "Z2"}


def test_unobserved_confounding_is_not_identified():
    graph = nx.DiGraph([("U", "X"), ("U", "Y"), ("X", "Y")])
    assert identify(graph, ["X", "Y"]).estimands["backdoor"] is None


def test_valid_optimized_candidate_does_not_use_standard_search():
    graph = nx.DiGraph([("A", "X"), ("A", "Y"), ("X", "Y")])
    with patch.object(auto_identifier, "identify_backdoor", wraps=auto_identifier.identify_backdoor) as standard:
        estimand = identify(graph)
    assert set(estimand.get_backdoor_variables()) == {"A"}
    standard.assert_not_called()


@pytest.mark.parametrize("hidden", [False, True])
def test_small_dags_against_path_oracle(hidden):
    # Every DAG respecting this four-node order: 2^6 = 64 graphs per visibility setting.
    nodes = ["U", "X", "Z", "Y"]
    possible_edges = list(itertools.combinations(nodes, 2))
    observed = [node for node in nodes if not hidden or node != "U"]
    candidates = [node for node in observed if node not in {"X", "Y"}]
    for mask in range(1 << len(possible_edges)):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edge for index, edge in enumerate(possible_edges) if mask & (1 << index))
        if not nx.has_path(graph, "X", "Y"):
            continue
        valid_sets = [
            candidate
            for size in range(len(candidates) + 1)
            for candidate in itertools.combinations(candidates, size)
            if valid_backdoor(graph, candidate)
        ]
        estimand = identify(graph, observed)
        assert (estimand.estimands["backdoor"] is not None) == bool(valid_sets), list(graph.edges())
        if valid_sets:
            adjustment = estimand.get_backdoor_variables()
            assert set(adjustment).issubset(observed)
            assert valid_backdoor(graph, adjustment), (list(graph.edges()), adjustment)
