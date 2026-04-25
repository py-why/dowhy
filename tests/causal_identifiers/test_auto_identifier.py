from dowhy.causal_identifier import AutoIdentifier
from dowhy.causal_identifier.auto_identifier import identify_mediation
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


class TestIdentifyMediation(object):
    """Tests for identify_mediation (issue #1334: parallel mediators)."""

    def test_single_mediator_identified(self):
        # Simple chain: D -> M -> Y, D -> Y
        graph = build_graph_from_str("digraph{D->M; D->Y; M->Y;}")
        result = identify_mediation(graph, ["D"], ["Y"])
        assert result == ["M"]

    def test_no_mediator_returns_empty(self):
        # Direct effect only: D -> Y
        graph = build_graph_from_str("digraph{D->Y;}")
        result = identify_mediation(graph, ["D"], ["Y"])
        assert result == []

    def test_parallel_mediators_both_identified(self):
        # Graph from issue #1334: D -> M1 -> Y, D -> M2 -> Y, D -> Y
        graph = build_graph_from_str("digraph{D->M1; D->M2; D->Y; M1->Y; M2->Y;}")
        result = identify_mediation(graph, ["D"], ["Y"])
        assert set(result) == {"M1", "M2"}, "Both parallel mediators should be identified; " f"got {result}"

    def test_parallel_mediators_appear_in_nie_estimand(self):
        # End-to-end: NIE estimand should reference both mediators
        graph = build_graph_from_str("digraph{D->M1; D->M2; D->Y; M1->Y; M2->Y;}")
        identifier = AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_NIE)
        estimand = identifier.identify_effect(
            graph,
            action_nodes=["D"],
            outcome_nodes=["Y"],
            observed_nodes=["D", "M1", "M2", "Y"],
        )
        mediator_vars = estimand.mediator_variables
        assert set(mediator_vars) == {"M1", "M2"}, (
            "NIE estimand mediator_variables should include both M1 and M2; " f"got {mediator_vars}"
        )
