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


class TestConstructFrontdoorEstimand:
    """Regression tests for construct_frontdoor_estimand and construct_mediation_estimand.

    Before the fix for issue #214, outcome names longer than one character were
    iterated character-by-character inside `",".join(outcome_name)` after the
    local variable was reassigned from list to str.  This caused multi-char
    outcome names (e.g. "behavior") to appear as "b,e,h,a,v,i,o,r" in the
    assumption text.
    """

    def test_frontdoor_estimand_assumptions_contain_full_outcome_name(self):
        """Assumption strings must use the full outcome name, not individual characters."""
        outcome_name = "long_outcome_variable"
        result = construct_frontdoor_estimand(
            treatment_name=["treatment"],
            outcome_name=[outcome_name],
            frontdoor_variables_names=["mediator"],
        )
        for assumption_text in result["assumptions"].values():
            # The full name must appear somewhere in each assumption that references the outcome
            if outcome_name[0] in assumption_text:  # heuristic: assumption mentions outcome at all
                assert (
                    outcome_name in assumption_text
                ), f"Assumption text does not contain full outcome name '{outcome_name}': {assumption_text!r}"

    def test_frontdoor_full_mediation_assumption_uses_full_outcome_name(self):
        outcome_name = "behavior"
        result = construct_frontdoor_estimand(
            treatment_name=["v0"],
            outcome_name=[outcome_name],
            frontdoor_variables_names=["FD0"],
        )
        text = result["assumptions"]["Full-mediation"]
        assert "behavior" in text, f"Full-mediation assumption missing full outcome name: {text!r}"
        assert "b,e,h,a,v,i,o,r" not in text, f"Full-mediation assumption contains character-joined name: {text!r}"

    def test_mediation_estimand_nie_assumptions_contain_full_outcome_name(self):
        outcome_name = "behavior"
        result = construct_mediation_estimand(
            estimand_type=EstimandType.NONPARAMETRIC_NIE,
            action_nodes=["v0"],
            outcome_nodes=[outcome_name],
            mediator_nodes=["FD0"],
        )
        mediation_text = result["assumptions"]["Mediation"]
        assert "behavior" in mediation_text, f"Mediation assumption missing full outcome name: {mediation_text!r}"
        assert (
            "b,e,h,a,v,i,o,r" not in mediation_text
        ), f"Mediation assumption contains character-joined name: {mediation_text!r}"

    def test_mediation_estimand_nde_assumptions_contain_full_outcome_name(self):
        outcome_name = "behavior"
        result = construct_mediation_estimand(
            estimand_type=EstimandType.NONPARAMETRIC_NDE,
            action_nodes=["v0"],
            outcome_nodes=[outcome_name],
            mediator_nodes=["FD0"],
        )
        second_stage_text = result["assumptions"]["Second-stage-unconfoundedness"]
        assert (
            "behavior" in second_stage_text
        ), f"Second-stage-unconfoundedness missing full outcome name: {second_stage_text!r}"
