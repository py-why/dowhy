import pytest

from dowhy.graph import build_graph_from_str

from .example_graphs import (
    TEST_FRONTDOOR_GRAPH_SOLUTIONS,
    TEST_GRAPH_SOLUTIONS,
    TEST_GRAPH_SOLUTIONS_COMPLETE_ADJUSTMENT,
)


class IdentificationTestGraphSolution(object):
    def __init__(
        self,
        graph_str,
        observed_variables,
        biased_sets,
        minimal_adjustment_sets,
        maximal_adjustment_sets,
        direct_maximal_adjustment_sets=None,
    ):
        self.graph = build_graph_from_str(graph_str)
        self.action_nodes = ["X"]
        self.outcome_nodes = ["Y"]
        self.observed_nodes = observed_variables
        self.biased_sets = biased_sets
        self.minimal_adjustment_sets = minimal_adjustment_sets
        self.maximal_adjustment_sets = maximal_adjustment_sets
        self.direct_maximal_adjustment_sets = (
            maximal_adjustment_sets if direct_maximal_adjustment_sets is None else direct_maximal_adjustment_sets
        )


class IdentificationTestFrontdoorGraphSolution(object):
    def __init__(
        self,
        graph_str,
        observed_variables,
        valid_frontdoor_sets,
        invalid_frontdoor_sets,
        action_nodes=None,
        outcome_nodes=None,
    ):
        if outcome_nodes is None:
            outcome_nodes = ["Y"]
        if action_nodes is None:
            action_nodes = ["X"]
        self.graph = build_graph_from_str(graph_str)
        self.action_nodes = action_nodes
        self.outcome_nodes = outcome_nodes
        self.observed_nodes = observed_variables
        self.valid_frontdoor_sets = valid_frontdoor_sets
        self.invalid_frontdoor_sets = invalid_frontdoor_sets


class IdentificationTestGeneralCovariateAdjustmentGraphSolution(object):
    def __init__(
        self,
        graph_str,
        observed_variables,
        action_nodes,
        outcome_nodes,
        minimal_adjustment_sets,
        exhaustive_adjustment_sets=None,
    ):
        self.graph = build_graph_from_str(graph_str)
        self.action_nodes = action_nodes
        self.outcome_nodes = outcome_nodes
        self.observed_nodes = observed_variables
        self.minimal_adjustment_sets = minimal_adjustment_sets
        self.exhaustive_adjustment_sets = exhaustive_adjustment_sets


@pytest.fixture(params=TEST_GRAPH_SOLUTIONS.keys())
def example_graph_solution(request):
    return IdentificationTestGraphSolution(**TEST_GRAPH_SOLUTIONS[request.param])


@pytest.fixture(params=TEST_FRONTDOOR_GRAPH_SOLUTIONS.keys())
def example_frontdoor_graph_solution(request):
    return IdentificationTestFrontdoorGraphSolution(**TEST_FRONTDOOR_GRAPH_SOLUTIONS[request.param])


@pytest.fixture(params=TEST_GRAPH_SOLUTIONS_COMPLETE_ADJUSTMENT.keys())
def example_complete_adjustment_graph_solution(request):
    return IdentificationTestGeneralCovariateAdjustmentGraphSolution(
        **TEST_GRAPH_SOLUTIONS_COMPLETE_ADJUSTMENT[request.param]
    )
