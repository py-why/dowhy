from dowhy.causal_identifier import AutoIdentifier
from dowhy.causal_identifier.auto_identifier import construct_frontdoor_estimand, construct_mediation_estimand
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


class TestConstructEstimands:
    """Tests for estimand helper functions (regression for issue #214)."""

    def test_construct_frontdoor_estimand_multi_char_outcome_not_char_joined(self):
        """Outcome name must appear as a whole word in assumptions, not character-by-character."""
        estimand = construct_frontdoor_estimand(
            treatment_name=["treatment"],
            outcome_name=["behavior"],
            frontdoor_variables_names=["mediator"],
        )
        assumptions = estimand["assumptions"]
        # "Full-mediation" assumption should contain the full word "behavior"
        assert "behavior" in assumptions["Full-mediation"]
        # It should NOT contain character-joined form "b,e,h,a,v,i,o,r"
        assert "b,e,h,a,v,i,o,r" not in assumptions["Full-mediation"]
        # "Second-stage-unconfoundedness" should contain "behavior" as a word
        assert "behavior" in assumptions["Second-stage-unconfoundedness"]

    def test_construct_frontdoor_estimand_single_char_outcome_unchanged(self):
        """Single-character outcome names should still work correctly."""
        estimand = construct_frontdoor_estimand(
            treatment_name=["T"],
            outcome_name=["Y"],
            frontdoor_variables_names=["M"],
        )
        assumptions = estimand["assumptions"]
        assert "Y" in assumptions["Full-mediation"]
        assert "Y" in assumptions["Second-stage-unconfoundedness"]

    def test_construct_mediation_estimand_nie_multi_char_outcome_not_char_joined(self):
        """NIE mediation assumption must use full outcome name, not character-joined."""
        estimand = construct_mediation_estimand(
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
            action_nodes=["treatment"],
            outcome_nodes=["behavior"],
            mediator_nodes=["mediator"],
        )
        assumptions = estimand["assumptions"]
        assert "behavior" in assumptions["Mediation"]
        assert "b,e,h,a,v,i,o,r" not in assumptions["Mediation"]
        assert "behavior" in assumptions["Second-stage-unconfoundedness"]

    def test_construct_mediation_estimand_nde_multi_char_outcome_not_char_joined(self):
        """NDE mediation assumption must use full outcome name, not character-joined."""
        estimand = construct_mediation_estimand(
            estimand_type=EstimandType.NONPARAMETRIC_NDE,
            action_nodes=["treatment"],
            outcome_nodes=["behavior"],
            mediator_nodes=["mediator"],
        )
        assumptions = estimand["assumptions"]
        assert "behavior" in assumptions["Mediation"]
        assert "b,e,h,a,v,i,o,r" not in assumptions["Mediation"]

