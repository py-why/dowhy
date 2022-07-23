#TODO, should this be added to the general examples_graphs.py?

TEST_EFFICIENT_BD_SOLUTIONS = {
    # For all examples from these papers we use X for the treatment variable instead of A
    # Figure 6 from Smucler, Sapienza and Rotnitzky (2021), Biometrika
    "ssr21_fig6_example_graph": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                        node[id "F" label "F"]
                        node[id "T" label "T"]
                        node[id "U" label "U"]
                        node[id "Y" label "Y"]
                        edge[source "T" target "X"]
                        edge[source "X" target "Y"]
                        edge[source "U" target "Y"]
                        edge[source "U" target "F"]
                        ]
                        """,
        observed_node_names=["X", "Y", "T", "F"],
        conditional_node_names = ["T"],
        optimal_adjustment_set = None,
        optimal_minimal_adjustment_set = {"T"},
        optimal_minimum_cost_adjustment_set = {"T"},
        costs = None
    ),
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
        optimal_minimum_cost_adjustment_set = set(),
        costs = None
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
        optimal_adjustment_set = {"T", "W2", "W3", "W4"},
        optimal_minimal_adjustment_set = {"T", "W2", "W3"},
        optimal_minimum_cost_adjustment_set = {"T", "W1"},
        costs = None
    ),
    # Figure 3 from Smucler, Sapienza and Rotnitzky (2021), Biometrika
    "ssr21_fig3_example_graph": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                        node[id "F" label "F"]
                        node[id "T" label "T"]
                        node[id "U" label "U"]
                        node[id "Y" label "Y"]
                        node[id "M" label "M"]
                        edge[source "T" target "X"]
                        edge[source "T" target "F"]
                        edge[source "F" target "X"]
                        edge[source "U" target "F"]
                        edge[source "U" target "Y"]
                        edge[source "X" target "M"]
                        edge[source "M" target "Y"]
                        ]
                        """,
        observed_node_names=["X", "Y", "T", "F", "M"],
        conditional_node_names = ["T"],
        optimal_adjustment_set = {"T", "F"},
        optimal_minimal_adjustment_set = {"T", "F"},
        optimal_minimum_cost_adjustment_set = {"T", "F"},
        costs = None
    ),
    # Figure 2 from Smucler and Rotnitzky (2022), Journal of Causal Inference
    # L replaces X as the conditional variable
    "sr22_fig2_example_graph": dict(
        graph_str="""graph[directed 1 node[id "L" label "L"] 
                        node[id "X" label "X"]
                        node[id "K" label "K"]
                        node[id "B" label "B"]
                        node[id "Q" label "Q"]
                        node[id "R" label "R"]
                        node[id "T" label "T"]
                        node[id "M" label "M"]
                        node[id "Y" label "Y"]
                        node[id "U" label "U"]
                        node[id "F" label "F"]
                        edge[source "L" target "X"]
                        edge[source "X" target "M"]
                        edge[source "K" target "X"]
                        edge[source "B" target "K"]
                        edge[source "B" target "R"]
                        edge[source "Q" target "K"]
                        edge[source "Q" target "T"]
                        edge[source "R" target "Y"]
                        edge[source "T" target "Y"]
                        edge[source "M" target "Y"]
                        edge[source "U" target "Y"]
                        edge[source "U" target "F"]
                        ]
                        """,
        observed_node_names=["L", "X", "B", "K", "Q", "R", "M", "T", "Y", "F"],
        conditional_node_names = ["L"],
        optimal_adjustment_set = None,
        optimal_minimal_adjustment_set = {"L", "T", "R"},
        optimal_minimum_cost_adjustment_set = {"L", "T", "R"},
        costs = [
    ("L", {"cost": 1}),
    ("B", {"cost": 2}),
    ("K", {"cost": 4}),
    ("Q", {"cost": 1}),
    ("R", {"cost": 1}),
    ("T", {"cost": 1}),
]
    ),
}
