import pytest
from dowhy.causal_graph import CausalGraph

from .example_graphs import TEST_GRAPH_SOLUTIONS


class IdentificationTestGraphSolution(object):

    def __init__(self, graph_str, observed_variables, biased_sets, minimal_adjustment_sets, maximal_adjustment_sets):
        self.graph = CausalGraph("X", "Y", graph_str, observed_node_names=observed_variables)
        self.graph_str = graph_str
        self.observed_variables = observed_variables
        self.biased_sets = biased_sets
        self.minimal_adjustment_sets = minimal_adjustment_sets
        self.maximal_adjustment_sets = maximal_adjustment_sets


@pytest.fixture(params=TEST_GRAPH_SOLUTIONS.keys())
def example_graph_solution(request):
    return IdentificationTestGraphSolution(**TEST_GRAPH_SOLUTIONS[request.param])
