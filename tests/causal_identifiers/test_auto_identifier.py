import copy

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

    def test_deepcopy_preserves_all_fields(self):
        """Regression test: __deepcopy__ must copy mediation confounders and no_directed_path."""
        # Build NDE estimand with mediation confounders
        graph = build_graph_from_str("digraph{T->M;T->Y;M->Y;W->T;W->Y;}")
        identifier = AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_NDE)
        estimand = identifier.identify_effect(
            graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["T", "M", "Y", "W"],
        )
        estimand_copy = copy.deepcopy(estimand)

        assert estimand_copy.mediator_variables == estimand.mediator_variables
        assert estimand_copy.mediation_first_stage_confounders == estimand.mediation_first_stage_confounders
        assert estimand_copy.mediation_second_stage_confounders == estimand.mediation_second_stage_confounders
        assert estimand_copy.no_directed_path == estimand.no_directed_path

    def test_deepcopy_preserves_no_directed_path_flag(self):
        """Regression test: deepcopy of an estimand with no directed path preserves the flag."""
        graph = build_graph_from_str("digraph{T->Y;A->Y;A->B;}")
        identifier = AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_ATE)
        estimand = identifier.identify_effect(
            graph, action_nodes=["T", "B"], outcome_nodes=["Y"], observed_nodes=["T", "Y", "A", "B"]
        )
        assert estimand.no_directed_path

        estimand_copy = copy.deepcopy(estimand)
        assert estimand_copy.no_directed_path
