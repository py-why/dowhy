from dowhy.causal_graph import CausalGraph
from dowhy.causal_identifiers.efficient_backdoor import EfficientBackdoor
from tests.causal_identifiers.example_graphs_efficient import TEST_EFFICIENT_BD_SOLUTIONS

#TODO we should adapt this to the logic used in the other tests

class TestEfficientBackdoorIdentification(object):

    def test_identify_backdoor_no_biased_sets(self):
        for example in TEST_EFFICIENT_BD_SOLUTIONS.items():
            G = CausalGraph(graph=example[1]['graph_str'],
                            treatment_name='X',
                            outcome_name='Y',
                            observed_node_names=example[1]['observed_node_names'],
                            )
            efficient_bd_identifier = EfficientBackdoor(graph=G, conditional_node_names=example[1]['conditional_node_names'],costs=example[1]['costs'])
            assert efficient_bd_identifier.optimal_minimal_adj_set() == example[1]['optimal_minimal_adjustment_set']
            assert efficient_bd_identifier.optimal_adj_set() == example[1]['optimal_adjustment_set']