#TODO, should this be added to the general examples_graphs.py?

TEST_EFFICIENT_BD_SOLUTIONS = {
    # Figure 5 from Smucler, Sapienza and Rotnitzky (2021), Biometrika
    "ssr21_fig5_example_graph": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                        node[id "Y" label "Y"]
                        node[id "U" label "U"]
                        node[id "Z1" label "Z1"]
                        node[id "Z2" label "Z2"]
                        edge[source "X" target "Y"]
                        edge[source "Z1" target "X"]
                        edge[source "Z1" target "Z2"]
                        edge[source "U" target "Z2"]
                        edge[source "U" target "Y"]]
                        """,
        observed_node_names=["X", "Y", "Z1", "Z2"],
        conditional_node_names = [],
        optimal_adjustment_set = None,
        optimal_minimal_adjustment_set = set(),
        #optimal_minimum_cost_adjustment_set = set(),
        costs = {}
    ),
    # Figure 4 from Smucler, Sapienza and Rotnitzky (2021), Biometrika
    "ssr21_fig4_example_graph": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                        node[id "Y" label "Y"]
                        node[id "T" label "T"]
                        node[id "W1" label "W1"]
                        node[id "W2" label "W2"]
                        node[id "W3" label "W3"]
                        node[id "W4" label "W4"]
                        edge[source "X" target "Y"]
                        edge[source "T" target "W1"]
                        edge[source "T" target "Y"]
                        edge[source "W1" target "X"]
                        edge[source "W2" target "W1"]
                        edge[source "W2" target "Y"]
                        edge[source "W3" target "W1"]
                        edge[source "W3" target "Y"]
                        edge[source "W4" target "Y"]
                        ]
                        """,
        observed_node_names=["A", "Y", "T", "W1", "W2", "W3", "W4"],
        conditional_node_names = ["T"],
        optimal_adjustment_set = set(["T", "W2", "W3", "W4"]),
        optimal_minimal_adjustment_set = set(["T", "W2", "W3"]),
        #optimal_minimum_cost_adjustment_set = set(),
        costs = {}
    ),
}





