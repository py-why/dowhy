import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel
from dowhy.causal_identifier import AutoIdentifier, BackdoorAdjustment, auto_identifier
from dowhy.causal_identifier.auto_identifier import identify_frontdoor
from dowhy.causal_identifier.identify_effect import EstimandType
from dowhy.graph import build_graph_from_str

from .base import IdentificationTestFrontdoorGraphSolution, example_frontdoor_graph_solution


class TestFrontdoorIdentification(object):
    def test_identify_frontdoor_functional_api(
        self, example_frontdoor_graph_solution: IdentificationTestFrontdoorGraphSolution
    ):
        graph = example_frontdoor_graph_solution.graph
        expected_sets = example_frontdoor_graph_solution.valid_frontdoor_sets
        invalid_sets = example_frontdoor_graph_solution.invalid_frontdoor_sets
        frontdoor_set = identify_frontdoor(
            graph,
            observed_nodes=example_frontdoor_graph_solution.observed_nodes,
            action_nodes=example_frontdoor_graph_solution.action_nodes,
            outcome_nodes=example_frontdoor_graph_solution.outcome_nodes,
        )

        assert (
            (len(frontdoor_set) == 0) and (len(expected_sets) == 0)
        ) or (  # No adjustments exist and that's expected.
            set(frontdoor_set) in expected_sets and set(frontdoor_set) not in invalid_sets
        )

    def test_identify_frontdoor_causal_model(
        self, example_frontdoor_graph_solution: IdentificationTestFrontdoorGraphSolution
    ):
        graph = example_frontdoor_graph_solution.graph
        expected_sets = example_frontdoor_graph_solution.valid_frontdoor_sets
        invalid_sets = example_frontdoor_graph_solution.invalid_frontdoor_sets
        observed_nodes = example_frontdoor_graph_solution.observed_nodes
        # Building the causal model
        num_samples = 10
        df = pd.DataFrame(np.random.random((num_samples, len(observed_nodes))), columns=observed_nodes)
        model = CausalModel(
            data=df,
            treatment=example_frontdoor_graph_solution.action_nodes,
            outcome=example_frontdoor_graph_solution.outcome_nodes,
            graph=graph,
        )
        estimand = model.identify_effect()
        frontdoor_set = estimand.frontdoor_variables
        assert (
            (len(frontdoor_set) == 0) and (len(expected_sets) == 0)
        ) or (  # No adjustments exist and that's expected.
            (set(frontdoor_set) in expected_sets) and (set(frontdoor_set) not in invalid_sets)
        )

    def test_identify_frontdoor_returns_smallest_set(self):
        # {M1}, {M2} and {M1, M2} are all valid frontdoor sets here. TwoStageRegression only supports a single
        # frontdoor variable, so returning the larger set made frontdoor estimation fail.
        graph = build_graph_from_str("digraph {Z->X; Z->Y; X->M1; M1->M2; M2->Y}")
        frontdoor_set = identify_frontdoor(
            graph, action_nodes=["X"], outcome_nodes=["Y"], observed_nodes=["X", "M1", "M2", "Y"]
        )

        assert frontdoor_set == ["M1"]

    def test_identify_frontdoor_does_not_search_subsets_when_none_can_be_valid(self, monkeypatch):
        # Test added for issue #1818. The direct X->Y edge cannot be intercepted, so no frontdoor set exists. This
        # used to be found by checking all 2^12 - 1 subsets of the mediators.
        mediators = [f"M{i}" for i in range(12)]
        graph = build_graph_from_str("digraph {X->Y; " + "->".join(["X", *mediators, "Y"]) + "}")

        num_checks = 0
        check_valid_frontdoor_set = auto_identifier.check_valid_frontdoor_set

        def counting_check(*args, **kwargs):
            nonlocal num_checks
            num_checks += 1
            return check_valid_frontdoor_set(*args, **kwargs)

        monkeypatch.setattr(auto_identifier, "check_valid_frontdoor_set", counting_check)

        frontdoor_set = identify_frontdoor(
            graph, action_nodes=["X"], outcome_nodes=["Y"], observed_nodes=["X", *mediators, "Y"]
        )

        assert frontdoor_set == []
        assert num_checks == 1

    def test_identify_frontdoor_respects_max_iterations(self, monkeypatch):
        # Every set that intercepts the causal paths contains B, and B<-C->Y is not blocked by X, so no frontdoor set
        # exists. Without a limit this graph needs every subset of the candidates to be checked.
        graph = build_graph_from_str("digraph {X->P0->P1->P2->P3->A->B->D->Y; X->D; C->B; C->Y}")
        monkeypatch.setattr(auto_identifier, "MAX_FRONTDOOR_ITERATIONS", 10)

        num_checks = 0
        check_valid_frontdoor_set = auto_identifier.check_valid_frontdoor_set

        def counting_check(*args, **kwargs):
            nonlocal num_checks
            num_checks += 1
            return check_valid_frontdoor_set(*args, **kwargs)

        monkeypatch.setattr(auto_identifier, "check_valid_frontdoor_set", counting_check)

        frontdoor_set = identify_frontdoor(
            graph, action_nodes=["X"], outcome_nodes=["Y"], observed_nodes=list(graph.nodes)
        )

        assert frontdoor_set == []
        # One check for the early Cond 1 test, then at most 10 candidate sets.
        assert num_checks <= 11
