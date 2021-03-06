import pytest
from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifier import CausalIdentifier

# TODO: What does num_paths_blocked_by_observed_nodes mean?
# TODO: Test no duplicates in sets
# TODO: Test no duplicate sets.

@pytest.fixture
def pearl_backdoor_example_graph():
    # TODO: Add reference to the book example.
    graph_str = """graph[directed 1 node[id "Z1" label "Z1"]  
                node[id "Z2" label "Z2"]
                node[id "Z3" label "Z3"]
                node[id "Z4" label "Z4"]
                node[id "Z5" label "Z5"]
                node[id "Z6" label "Z6"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]      
                edge[source "Z1" target "Z3"]
                edge[source "Z1" target "Z4"]
                edge[source "Z2" target "Z4"]
                edge[source "Z2" target "Z5"]
                edge[source "Z3" target "X"]
                edge[source "Z4" target "X"]
                edge[source "Z4" target "Y"]
                edge[source "Z5" target "Y"]
                edge[source "Z6" target "Y"]
                edge[source "X" target "Z6"]]    
                """

    graph = CausalGraph(
            "X",
            "Y",
            graph_str,
            observed_node_names=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "X", "Y"]
    )
    biased_backdoor_sets = [("Z4",), ("Z6",), ("Z5",), ("Z2",), ("Z1",), ("Z3",), ("Z1", "Z3"), ("Z2", "Z5"), ("Z1", "Z2")]
    minimal_sufficient_adjustment_sets = [("Z1", "Z4"), ("Z2", "Z4"), ("Z3", "Z4"), ("Z5", "Z4")] 

    return graph, biased_backdoor_sets, minimal_sufficient_adjustment_sets

@pytest.fixture
def simple_selection_bias_graph():
    graph_str = """graph[directed 1 node[id "Z1" label "Z1"]  
                    node[id "X" label "X"]
                    node[id "Y" label "Y"]      
                    edge[source "X" target "Y"]
                    edge[source "X" target "Z1"]
                    edge[source "Y" target "Z1"]]
                    """

    graph = CausalGraph(
            "X",
            "Y",
            graph_str,
            observed_node_names=["Z1", "X", "Y"]
    )
    biased_backdoor_sets = [("Z1",)] 
    minimal_sufficient_adjustment_sets = []

    return graph, biased_backdoor_sets, minimal_sufficient_adjustment_sets

@pytest.fixture
def simple_no_confounder_graph():
    graph_str = """graph[directed 1 node[id "Z1" label "Z1"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]      
                edge[source "X" target "Y"]
                edge[source "Z1" target "X"]]
                """

    graph = CausalGraph(
            "X",
            "Y",
            graph_str,
            observed_node_names=["Z1", "X", "Y"]
    )
    biased_backdoor_sets = [] 
    minimal_sufficient_adjustment_sets = []

    return graph, biased_backdoor_sets, minimal_sufficient_adjustment_sets

# See this discussion for details: https://github.com/pytest-dev/pytest/issues/349
# Specifically, this solution: https://github.com/pytest-dev/pytest/issues/349#issuecomment-189370273.
# getfixturevalue repleces getfuncargvalue as the latter is depracated.
@pytest.fixture(params=["pearl_backdoor_example_graph", "simple_selection_bias_graph", "simple_no_confounder_graph"])
def example_graph(request):
    return request.getfixturevalue(request.param)

class TestBackdoorIdentification(object):

    def test_identify_backdoor_no_biased_sets(self, example_graph):
        graph, biased_backdoor_sets, _ = example_graph
        identifier = CausalIdentifier(graph, "nonparametric-ate", method_name="exhaustive-search")
        
        backdoor_results = identifier.identify_backdoor("X", "Y")
        backdoor_sets = [
            set(backdoor_result_dict["backdoor_set"]) 
            for backdoor_result_dict in backdoor_results
            if len(backdoor_result_dict["backdoor_set"]) > 0
        ]

        assert (
            (len(backdoor_sets) == 0 and len(biased_backdoor_sets) == 0) # No biased sets exist and that's expected.
            or  
            all([
                set(biased_backdoor_set) not in backdoor_sets 
                for biased_backdoor_set in biased_backdoor_sets
            ]) # No sets that would induce biased results are present in the solution.
        )

    def test_identify_backdoor_minimal_sufficient_adjustment_sets(self, example_graph):
        graph, _, minimal_sufficient_adjustment_sets = example_graph
        identifier = CausalIdentifier(graph, "nonparametric-ate", method_name="exhaustive-search")
        
        backdoor_results = identifier.identify_backdoor("X", "Y")
        backdoor_sets = [
            set(backdoor_result_dict["backdoor_set"]) 
            for backdoor_result_dict in backdoor_results
            if len(backdoor_result_dict["backdoor_set"]) > 0
        ]

        assert (
            (len(backdoor_sets) == 0 and len(minimal_sufficient_adjustment_sets) == 0) # No adjustments needed and that's expected.
            or
            all([
                set(minimal_sufficient_adjustment_set) in backdoor_sets 
                for minimal_sufficient_adjustment_set in minimal_sufficient_adjustment_sets
            ]) # The solution contains all the minimal sufficient adjustment sets.
        )

    
