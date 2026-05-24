import pandas as pd

from dowhy import CausalModel
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


class TestIdentifyMediation:
    """Tests for identify_mediation with single and parallel mediators (issue #1334)."""

    def test_single_mediator_identified(self):
        """Single mediator case: D -> M -> Y, D -> Y."""
        graph = build_graph_from_str("digraph{D->M;D->Y;M->Y;}")
        mediators = identify_mediation(graph, ["D"], ["Y"])
        assert mediators == ["M"]

    def test_parallel_mediators_both_identified(self):
        """Parallel mediator case from issue #1334: D -> {M1, M2, Y}; M1 -> Y; M2 -> Y.

        Both M1 and M2 must be returned, not just the first one found.
        """
        graph = build_graph_from_str("digraph{D->M1;D->M2;D->Y;M1->Y;M2->Y;}")
        mediators = identify_mediation(graph, ["D"], ["Y"])
        assert sorted(mediators) == ["M1", "M2"]

    def test_no_mediator_returns_empty(self):
        """Graph with no mediating path returns empty list."""
        graph = build_graph_from_str("digraph{D->Y;}")
        mediators = identify_mediation(graph, ["D"], ["Y"])
        assert mediators == []

    def test_parallel_mediators_nie_estimand_includes_both(self):
        """End-to-end: NIE estimand with parallel mediators lists both in mediator_variables."""
        dag = "digraph{D->M1;D->M2;D->Y;M1->Y;M2->Y;}"
        df = pd.DataFrame({"D": [0, 1], "M1": [0.0, 1.0], "M2": [0.0, 1.0], "Y": [0.0, 1.0]})
        model = CausalModel(data=df, treatment="D", outcome="Y", graph=dag)
        estimand = model.identify_effect(
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
            proceed_when_unidentifiable=True,
        )
        assert sorted(estimand.mediator_variables) == ["M1", "M2"]
