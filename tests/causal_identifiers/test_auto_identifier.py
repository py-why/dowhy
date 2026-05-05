from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import AutoIdentifier
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.graph import build_graph_from_str


class TestAutoIdentification(object):
    def test_auto_identify_accepts_causal_graph(self):
        # Regression test for issue #1360: passing a CausalGraph directly should give
        # the same result as passing its underlying nx.DiGraph via G._graph.
        causal_graph = CausalGraph(
            treatment_name="T",
            outcome_name="Y",
            common_cause_names=["Z"],
            observed_node_names=["T", "Y", "Z"],
        )
        identifier = AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_ATE)

        result_from_causal_graph = identifier.identify_effect(
            causal_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["T", "Y", "Z"],
        )
        result_from_digraph = identifier.identify_effect(
            causal_graph._graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["T", "Y", "Z"],
        )

        assert not result_from_causal_graph.no_directed_path
        assert str(result_from_causal_graph) == str(result_from_digraph)

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
