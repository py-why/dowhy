"""
The example below illustrate some of the common examples of causal graph in books.
This file is meant to group all of the graph definitions as well as the expected results of identification algorithms in one place.
Each example graph is contained of the following values:

    * graph_str - The graph string in GML format.
    * observed_variables - A list of observed variables in the graph. This will be used to test no unobserved variables are offered in the solution.
    * biased_sets - The sets that we shouldn't get in the output as they incur biased estimates of the causal effect.
    * minimal_adjustment_sets - Sets of observed variables that should be returned when 'minimal-adjustment' is specified as the backdoor method. Contains the empty set as well.
    * maximal_adjustment_sets - Sets of observed variables that should be returned when 'maximal-adjustment' is specified as the backdoor method.
    * exhaustive_search_sets - Sets of observed variables that we want to check as the output of 'exhaustive-search' backdoor method. 'minimal-adjustment' and 'maximal-adjustment' will be included here.
"""

TEST_GRAPH_SOLUTIONS = {
    # Example is selected from Pearl J. "Causality" 2nd Edition, from chapter 3.3.1 on backoor criterion.
    "pearl_backdoor_example_graph": dict(
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
                    """,
        observed_variables = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "X", "Y"],
        biased_sets = [{"Z4"}, {"Z6"}, {"Z5"}, {"Z2"}, {"Z1"}, {"Z3",}, {"Z1", "Z3"}, {"Z2", "Z5"}, {"Z1", "Z2"}],
        minimal_adjustment_sets = [{"Z1", "Z4"}, {"Z2", "Z4"}, {"Z3", "Z4"}, {"Z5", "Z4"}],
        maximal_adjustment_sets = [{"Z1", "Z2", "Z3", "Z4", "Z5"}],
        adjustment_not_necessary = False
    ),
    "simple_selection_bias_graph": dict(
        graph_str = """graph[directed 1 node[id "Z1" label "Z1"]  
                    node[id "X" label "X"]
                    node[id "Y" label "Y"]      
                    edge[source "X" target "Y"]
                    edge[source "X" target "Z1"]
                    edge[source "Y" target "Z1"]]
                    """,
        observed_variables = ["Z1", "X", "Y"],
        biased_sets = [{"Z1",}], 
        minimal_adjustment_sets = [{}],
        maximal_adjustment_sets = [],
        adjustment_not_necessary = True
    ),
    "simple_no_confounder_graph": dict(
        graph_str = """graph[directed 1 node[id "Z1" label "Z1"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]      
                edge[source "X" target "Y"]
                edge[source "Z1" target "X"]]
                """,
        observed_variables=["Z1", "X", "Y"],
        biased_sets = [],
        minimal_adjustment_sets = [{}],
        maximal_adjustment_sets = [{"Z1",}],
        adjustment_not_necessary = True
    ),
    # The following simpsons paradox examples are taken from Pearl, J {2013}. "Understanding Simpson’s Paradox" - http://ftp.cs.ucla.edu/pub/stat_ser/r414.pdf
    "pearl_simpsons_paradox_1c": dict(
        graph_str = """graph[directed 1 node[id "Z" label "Z"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L1" label "L1"]
                node[id "L2" label "L2"]      
                edge[source "X" target "Y"]
                edge[source "L1" target "X"]
                edge[source "L1" target "Z"]
                edge[source "L2" target "Z"]
                edge[source "L2" target "Y"]]
                """,
        observed_variables=["Z", "X", "Y"],
        biased_sets = [{"Z",}],
        minimal_adjustment_sets = [{}],
        maximal_adjustment_sets = [],
        adjustment_not_necessary = True
    ),
    "pearl_simpsons_paradox_1d": dict(
        graph_str = """graph[directed 1 node[id "Z" label "Z"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L1" label "L1"]
                edge[source "X" target "Y"]
                edge[source "L1" target "X"]
                edge[source "L1" target "Z"]
                edge[source "Z" target "Y"]]
                """,
        observed_variables = ["Z", "X", "Y"],
        biased_sets = [],
        minimal_adjustment_sets = [{"Z",}],
        maximal_adjustment_sets = [{"Z",}],
        adjustment_not_necessary = False
    ),
    "pearl_simpsons_paradox_2a": dict(
        graph_str = """graph[directed 1 node[id "Z" label "Z"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L" label "L"]      
                edge[source "X" target "Y"]
                edge[source "X" target "Z"]
                edge[source "L" target "Z"]
                edge[source "L" target "Y"]]
                """,
        observed_variables = ["Z", "X", "Y"],
        biased_sets = [{"Z", }],
        minimal_adjustment_sets = [{}],
        maximal_adjustment_sets = [{}],
        adjustment_not_necessary = True
    ),
    "pearl_simpsons_paradox_2b": dict(
        graph_str = """graph[directed 1 node[id "Z" label "Z"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L" label "L"]      
                edge[source "X" target "Y"]
                edge[source "Z" target "X"]
                edge[source "L" target "X"]
                edge[source "L" target "Y"]]""",
        observed_variables = ["Z", "X", "Y"],
        biased_sets = [], 
        minimal_adjustment_sets = [],
        maximal_adjustment_sets = [], # Should this be {"Z"}?
        adjustment_not_necessary = False
    ),
    "pearl_simpsons_machine_lvl1": dict(
        graph_str = """graph[directed 1 node[id "Z1" label "Z1"]
                node[id "Z2" label "Z2"]
                node[id "Z3" label "Z3"]
                node[id "L" label "L"]  
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                edge[source "X" target "Y"]
                edge[source "Z1" target "L"]
                edge[source "L" target "Z2"]
                edge[source "Z3" target "Z2"]
                edge[source "L" target "X"]
                edge[source "Z3" target "Y"]]
                """,
        observed_variables=["Z1", "Z2", "Z3", "X", "Y"],
        biased_sets = [{"Z2",}, {"Z1", "Z2"}],
        minimal_adjustment_sets = [{}],
        maximal_adjustment_sets = [{"Z1", "Z2", "Z3"}],
        adjustment_not_necessary = True
    ),
    # The following are examples given in the "Book of Why" by Judea Pearl, chapter "The Do-operator and the Back-Door Criterion"
    "book_of_why_game2": dict(
        graph_str = """graph[directed 1 node[id "A" label "A"]
                node[id "B" label "B"]
                node[id "C" label "C"]
                node[id "D" label "D"]  
                node[id "E" label "E"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                edge[source "A" target "X"]
                edge[source "A" target "B"]
                edge[source "B" target "C"]
                edge[source "D" target "B"]
                edge[source "D" target "E"]
                edge[source "X" target "E"]
                edge[source "E" target "Y"]]
                """,
        observed_variables = ["A", "B", "C", "D", "E", "X", "Y"],
        biased_sets = [{"B",}, {"C",}, {"B", "C"}],
        minimal_adjustment_sets = [{}],
        maximal_adjustment_sets = [{"A", "B", "C", "D"}],
        adjustment_not_necessary = True
    ),
    "book_of_why_game5": dict(
        graph_str = """graph[directed 1 node[id "A" label "A"]
                node[id "B" label "B"]
                node[id "C" label "C"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                edge[source "A" target "X"]
                edge[source "A" target "B"]
                edge[source "B" target "X"]
                edge[source "C" target "B"]
                edge[source "C" target "Y"]
                edge[source "X" target "Y"]]
                """,
        observed_variables = ["A", "B", "C", "X", "Y"],
        biased_sets = [{"B",}],
        minimal_adjustment_sets = [{"C"}],
        maximal_adjustment_sets = [{"A", "B", "C"}],
        adjustment_not_necessary = False
    ),
    "book_of_why_game5_C_is_unobserved": dict(
        graph_str = """graph[directed 1 node[id "A" label "A"]
                node[id "B" label "B"]
                node[id "C" label "C"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                edge[source "A" target "X"]
                edge[source "A" target "B"]
                edge[source "B" target "X"]
                edge[source "C" target "B"]
                edge[source "C" target "Y"]
                edge[source "X" target "Y"]]
                """,
        observed_variables = ["A", "B", "X", "Y"],
        biased_sets = [{"B",}],
        minimal_adjustment_sets = [{"A", "B"}],
        maximal_adjustment_sets = [{"A", "B"}],
        adjustment_not_necessary = False
    )


}








