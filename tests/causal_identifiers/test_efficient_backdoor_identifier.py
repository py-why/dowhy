import copy

import networkx as nx
import pytest

from dowhy.causal_identifier import AutoIdentifier, BackdoorAdjustment, EstimandType
from dowhy.causal_identifier.auto_identifier import EFFICIENT_METHODS
from dowhy.causal_identifier.efficient_backdoor import EfficientBackdoor
from dowhy.graph import build_graph_from_str
from tests.causal_identifiers.example_graphs_efficient import TEST_EFFICIENT_BD_SOLUTIONS


def test_identify_efficient_backdoor_algorithms():
    for example in TEST_EFFICIENT_BD_SOLUTIONS.values():
        for method_name in EFFICIENT_METHODS:
            ident_eff = AutoIdentifier(
                estimand_type=EstimandType.NONPARAMETRIC_ATE,
                backdoor_adjustment=method_name,
                costs=example["costs"],
            )
            method_name_results = method_name.value.replace("-", "_")
            if example[method_name_results] is None:
                with pytest.raises(ValueError):
                    ident_eff.identify_effect(
                        build_graph_from_str(example["graph_str"]),
                        observed_nodes=example["observed_node_names"],
                        action_nodes=["X"],
                        outcome_nodes=["Y"],
                        conditional_node_names=example["conditional_node_names"],
                    )
            else:
                results_eff = ident_eff.identify_effect(
                    build_graph_from_str(example["graph_str"]),
                    observed_nodes=example["observed_node_names"],
                    action_nodes=["X"],
                    outcome_nodes=["Y"],
                    conditional_node_names=example["conditional_node_names"],
                )
                assert set(results_eff.get_backdoor_variables()) == example[method_name_results]


def test_fail_negative_costs_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    mod_costs = copy.deepcopy(example["costs"])
    mod_costs[0][1]["cost"] = 0
    ident_eff = AutoIdentifier(
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT,
        costs=mod_costs,
    )

    with pytest.raises(Exception):
        ident_eff.identify_effect(
            build_graph_from_str(example["graph_str"]),
            observed_nodes=example["observed_node_names"],
            action_nodes=["X"],
            outcome_nodes=["Y"],
            conditional_node_names=example["conditional_node_names"],
        )


def test_fail_unobserved_cond_vars_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    ident_eff = AutoIdentifier(
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT,
        costs=example["costs"],
    )
    mod_cond_names = copy.deepcopy(example["conditional_node_names"])
    mod_cond_names.append("U")
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            build_graph_from_str(example["graph_str"]),
            observed_nodes=example["observed_node_names"],
            action_nodes=["X"],
            outcome_nodes=["Y"],
            conditional_node_names=mod_cond_names,
        )


def test_fail_multivar_treat_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    ident_eff = AutoIdentifier(
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT,
        costs=example["costs"],
    )
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            build_graph_from_str(example["graph_str"]),
            observed_nodes=example["observed_node_names"],
            action_nodes=["X", "K"],
            outcome_nodes=["Y"],
            conditional_node_names=example["conditional_node_names"],
        )


def test_fail_multivar_outcome_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    ident_eff = AutoIdentifier(
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MINCOST_EFFICIENT,
        costs=example["costs"],
    )
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            build_graph_from_str(example["graph_str"]),
            observed_nodes=example["observed_node_names"],
            action_nodes=["U"],
            outcome_nodes=["Y", "F"],
            conditional_node_names=example["conditional_node_names"],
        )


def _make_eb(edges, all_nodes, treatment="X", outcome="Y"):
    """Helper: build an EfficientBackdoor from a list of edges."""
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)
    return EfficientBackdoor(G, [treatment], [outcome], all_nodes)


def test_ancestors_all_chain():
    """ancestors_all on a chain X->Z->Y, W->X includes W."""
    eb = _make_eb([("X", "Z"), ("Z", "Y"), ("W", "X")], ["W", "X", "Z", "Y"])
    assert eb.ancestors_all(["X"]) == {"X", "W"}
    assert eb.ancestors_all(["Y"]) == {"W", "X", "Z", "Y"}


def test_ancestors_all_includes_seed_nodes():
    """ancestors_all always includes the query nodes themselves."""
    eb = _make_eb([("X", "Y")], ["X", "Y"])
    result = eb.ancestors_all(["X", "Y"])
    assert "X" in result and "Y" in result


def test_ancestors_all_multi_source_matches_union_of_singles():
    """Multi-source call matches manual union of per-node results on a larger graph."""
    edges = [("A", "X"), ("B", "X"), ("C", "M"), ("M", "Y"), ("X", "M")]
    nodes = ["A", "B", "C", "M", "X", "Y"]
    eb = _make_eb(edges, nodes)
    multi = eb.ancestors_all(["X", "Y"])
    manual = nx.ancestors(eb.graph, "X") | nx.ancestors(eb.graph, "Y") | {"X", "Y"}
    assert multi == manual


def test_forbidden_chain_dag():
    """forbidden() on X->Z->Y contains X, Z, and Y (Z's descendants include Y)."""
    eb = _make_eb([("X", "Z"), ("Z", "Y")], ["X", "Z", "Y"])
    result = eb.forbidden()
    # Causal vertices = {Z, Y} (nodes on X->Z->Y path, minus X)
    # forbidden = descendants({Z, Y}) ∪ {Z, Y} ∪ {X} = {X, Z, Y}
    assert result == {"X", "Z", "Y"}


def test_forbidden_always_contains_treatment():
    """forbidden() always contains the treatment node."""
    eb = _make_eb([("X", "Y")], ["X", "Y"])
    assert "X" in eb.forbidden()
