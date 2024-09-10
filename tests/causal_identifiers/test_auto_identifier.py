from dowhy.causal_identifier import AutoIdentifier
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.graph import build_graph_from_str


class TestAutoIdentification(object):
    def test_auto_identify_consistently_checks_for_directed_paths(self):
        graph = build_graph_from_str("digraph{T->Y;A->Y;A->B;}")
        identifier = AutoIdentifier(
            estimand_type=EstimandType.NONPARAMETRIC_ATE
        )
        identified_estimand = identifier.identify_effect(
            graph,
            action_nodes=["T", "B"],
            outcome_nodes=["Y"],
            observed_nodes=["T", "Y", "A", "B"]
        )
        identified_estimand_swapped_action_order = identifier.identify_effect(
            graph,
            action_nodes=["B", "T"],
            outcome_nodes=["Y"],
            observed_nodes=["T", "Y", "A", "B"]
        )
        backdoor_vars = identified_estimand.get_backdoor_variables()
        backdoor_vars_swapped_action_order = identified_estimand_swapped_action_order.get_backdoor_variables()

        assert len(backdoor_vars) == 0
        assert len(backdoor_vars_swapped_action_order) == 0

