from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import CausalIdentifier
from tests.causal_identifiers.example_graphs_efficient import (
    TEST_EFFICIENT_BD_SOLUTIONS,
)
import pytest


def test_identify_efficient_backdoor_algorithms():
    for example in TEST_EFFICIENT_BD_SOLUTIONS.items():
        G = CausalGraph(
            graph=example[1]["graph_str"],
            treatment_name="X",
            outcome_name="Y",
            observed_node_names=example[1]["observed_node_names"],
        )
        for method_name in CausalIdentifier.EFFICIENT_METHODS:
            print(method_name)
            ident_eff = CausalIdentifier(
                graph=G, estimand_type="nonparametric-ate", method_name=method_name
            )
            method_name_results = method_name.replace("-", "_")
            if example[1][method_name_results] is None:
                with pytest.raises(ValueError) as exception:
                    ident_eff.identify_effect(
                        costs=example[1]["costs"],
                        conditional_node_names=example[1]["conditional_node_names"],
                    )
            else:
                results_eff = ident_eff.identify_effect(
                    costs=example[1]["costs"],
                    conditional_node_names=example[1]["conditional_node_names"],
                )
                assert (
                    set(results_eff.get_backdoor_variables())
                    == example[1][method_name_results]
                )
