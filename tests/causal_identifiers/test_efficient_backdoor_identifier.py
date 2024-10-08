import copy

import pytest

from dowhy.causal_identifier import AutoIdentifier, BackdoorAdjustment, EstimandType
from dowhy.causal_identifier.auto_identifier import EFFICIENT_METHODS
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
