from dowhy.causal_identifier import AutoIdentifier
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.graph import build_graph_from_str


class TestAutoIdentification(object):
    def test_auto_identify_identifies_no_directed_path(self):
        # Test added for issue #1250
        graph = build_graph_from_str("digraph{T->Y;A->Y;A->B;}")
        identifier = AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_ATE)

        assert identifier.identify_effect(
            graph, action_nodes=["T", "B"], outcome_nodes=["Y"], observed_nodes=["T", "Y", "A", "B"]
        ).no_directed_path
        assert identifier.identify_effect(
            graph, action_nodes=["B", "T"], outcome_nodes=["Y"], observed_nodes=["T", "Y", "A", "B"]
        ).no_directed_path
