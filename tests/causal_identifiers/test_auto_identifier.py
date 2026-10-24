from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import AutoIdentifier
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.graph import build_graph_from_str


class TestAutoIdentification(object):
    def test_auto_identify_accepts_causal_graph(self):
        """Regression test for issue #1360: AutoIdentifier.identify_effect should accept a
        CausalGraph wrapper, not just a plain nx.DiGraph."""
        causal_graph = CausalGraph(
            treatment_name="T",
            outcome_name="Y",
            graph="digraph{T->Y;}",
        )
        identifier = AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_ATE)
        estimand = identifier.identify_effect(
            causal_graph,
            action_nodes=["T"],
            outcome_nodes=["Y"],
            observed_nodes=["T", "Y"],
        )
        assert estimand is not None
        assert not estimand.no_directed_path

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


def test_causal_identifier_protocol_importable_from_top_level():
    """Regression test for issue #831: CausalIdentifier must be importable from dowhy.causal_identifier."""
    from dowhy.causal_identifier import CausalIdentifier  # noqa: F401

    assert CausalIdentifier is not None


def test_auto_identifier_and_id_identifier_conform_to_causal_identifier_protocol():
    """AutoIdentifier and IDIdentifier both implement the CausalIdentifier Protocol."""
    from typing import runtime_checkable

    import pytest

    from dowhy.causal_identifier import AutoIdentifier, CausalIdentifier, IDIdentifier

    # Make it runtime-checkable to allow isinstance checks
    RuntimeCheckableCausalIdentifier = runtime_checkable(CausalIdentifier)
    assert isinstance(AutoIdentifier(estimand_type=EstimandType.NONPARAMETRIC_ATE), RuntimeCheckableCausalIdentifier)
    assert isinstance(IDIdentifier(), RuntimeCheckableCausalIdentifier)


def test_find_valid_adjustment_sets_respects_max_iterations(monkeypatch):
    import networkx as nx

    from dowhy.causal_identifier.auto_identifier import (
        BackdoorAdjustment,
        find_valid_adjustment_sets,
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(["T", "Y", "A", "B", "C", "D", "E"])

    num_checks = 0

    def always_invalid(*args, **kwargs):
        nonlocal num_checks
        num_checks += 1
        return {"is_dseparated": False}

    monkeypatch.setattr(
        "dowhy.causal_identifier.auto_identifier.check_valid_backdoor_set",
        always_invalid,
    )

    find_valid_adjustment_sets(
        graph=graph,
        action_nodes=["T"],
        outcome_nodes=["Y"],
        observed_nodes=["T", "Y"],
        backdoor_paths=None,
        bdoor_graph=graph,
        dseparation_algo="default",
        backdoor_sets=[],
        filt_eligible_variables=["A", "B", "C", "D", "E"],
        backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MIN,
        max_iterations=3,
    )

    assert num_checks == 3
