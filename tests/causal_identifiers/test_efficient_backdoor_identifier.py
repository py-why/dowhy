from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import CausalIdentifier
from tests.causal_identifiers.example_graphs_efficient import (
    TEST_EFFICIENT_BD_SOLUTIONS,
)
import pytest
import copy


def test_identify_efficient_backdoor_algorithms():
    for example in TEST_EFFICIENT_BD_SOLUTIONS.values():
        G = CausalGraph(
            graph=example["graph_str"],
            treatment_name="X",
            outcome_name="Y",
            observed_node_names=example["observed_node_names"],
        )
        for method_name in CausalIdentifier.EFFICIENT_METHODS:
            ident_eff = CausalIdentifier(
                graph=G, estimand_type="nonparametric-ate", method_name=method_name
            )
            method_name_results = method_name.replace("-", "_")
            if example[method_name_results] is None:
                with pytest.raises(ValueError):
                    ident_eff.identify_effect(
                        costs=example["costs"],
                        conditional_node_names=example["conditional_node_names"],
                    )
            else:
                results_eff = ident_eff.identify_effect(
                    costs=example["costs"],
                    conditional_node_names=example["conditional_node_names"],
                )
                assert (
                    set(results_eff.get_backdoor_variables())
                    == example[method_name_results]
                )


def test_fail_negative_costs_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    G = CausalGraph(
        graph=example["graph_str"],
        treatment_name="X",
        outcome_name="Y",
        observed_node_names=example["observed_node_names"],
    )
    ident_eff = CausalIdentifier(
        graph=G,
        estimand_type="nonparametric-ate",
        method_name="efficient-mincost-adjustment",
    )
    mod_costs = copy.deepcopy(example["costs"])
    mod_costs[0][1]["cost"] = 0
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            costs=mod_costs, conditional_node_names=example["conditional_node_names"],
        )


def test_fail_unobserved_cond_vars_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    G = CausalGraph(
        graph=example["graph_str"],
        treatment_name="X",
        outcome_name="Y",
        observed_node_names=example["observed_node_names"],
    )
    ident_eff = CausalIdentifier(
        graph=G,
        estimand_type="nonparametric-ate",
        method_name="efficient-mincost-adjustment",
    )
    mod_cond_names = copy.deepcopy(example["conditional_node_names"])
    mod_cond_names.append("U")
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            costs=example["costs"], conditional_node_names=mod_cond_names,
        )


def test_fail_multivar_treat_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    G = CausalGraph(
        graph=example["graph_str"],
        treatment_name=["X", "K"],
        outcome_name="Y",
        observed_node_names=example["observed_node_names"],
    )
    ident_eff = CausalIdentifier(
        graph=G,
        estimand_type="nonparametric-ate",
        method_name="efficient-mincost-adjustment",
    )
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            costs=example["costs"],
            conditional_node_names=example["conditional_node_names"],
        )


def test_fail_multivar_outcome_efficient_backdoor_algorithms():
    example = TEST_EFFICIENT_BD_SOLUTIONS["sr22_fig2_example_graph"]
    G = CausalGraph(
        graph=example["graph_str"],
        treatment_name="X",
        outcome_name=["Y", "R"],
        observed_node_names=example["observed_node_names"],
    )
    ident_eff = CausalIdentifier(
        graph=G,
        estimand_type="nonparametric-ate",
        method_name="efficient-mincost-adjustment",
    )
    with pytest.raises(Exception):
        ident_eff.identify_effect(
            costs=example["costs"],
            conditional_node_names=example["conditional_node_names"],
        )
