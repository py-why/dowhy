# TODO, should this be added to the general examples_graphs.py?

TEST_EFFICIENT_BD_SOLUTIONS = {
    # For all examples from these papers we use X for the treatment variable
    # instead of A.
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
        conditional_node_names=["T"],
        efficient_adjustment=None,
        efficient_minimal_adjustment={"T"},
        efficient_mincost_adjustment={"T"},
        costs=None,
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
        conditional_node_names=[],
        efficient_adjustment=None,
        efficient_minimal_adjustment=set(),
        efficient_mincost_adjustment=set(),
        costs=None,
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
        conditional_node_names=["T"],
        efficient_adjustment={"T", "W2", "W3", "W4"},
        efficient_minimal_adjustment={"T", "W2", "W3"},
        efficient_mincost_adjustment={"T", "W1"},
        costs=None,
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
        conditional_node_names=["T"],
        efficient_adjustment={"T", "F"},
        efficient_minimal_adjustment={"T", "F"},
        efficient_mincost_adjustment={"T", "F"},
        costs=None,
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
        conditional_node_names=["L"],
        efficient_adjustment=None,
        efficient_minimal_adjustment={"L", "T", "R"},
        efficient_mincost_adjustment={"L", "T", "R"},
        costs=[
            ("L", {"cost": 1}),
            ("B", {"cost": 2}),
            ("K", {"cost": 4}),
            ("Q", {"cost": 1}),
            ("R", {"cost": 1}),
            ("T", {"cost": 1}),
        ],
    ),
    # Figure 2 from Smucler and Rotnitzky (2022), Journal of Causal Inference
    # L replaces X as the conditional variable. Uses different costs
    "sr22_fig2_diffcosts_example_graph": dict(
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
        conditional_node_names=["L"],
        efficient_adjustment=None,
        efficient_minimal_adjustment={"L", "T", "R"},
        efficient_mincost_adjustment={"L", "T", "B"},
        costs=[
            ("L", {"cost": 1}),
            ("B", {"cost": 1}),
            ("K", {"cost": 4}),
            ("Q", {"cost": 1}),
            ("R", {"cost": 2}),
            ("T", {"cost": 1}),
        ],
    ),
    # Figure 3 from Smucler and Rotnitzky (2022), Journal of Causal Inference
    "sr22_fig3_example_graph": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                        node[id "B" label "B"]
                        node[id "Q" label "Q"]
                        node[id "R" label "R"]
                        node[id "T" label "T"]
                        node[id "Y" label "Y"]
                        edge[source "B" target "X"]
                        edge[source "Q" target "X"]
                        edge[source "B" target "T"]
                        edge[source "Q" target "R"]
                        edge[source "T" target "Y"]
                        edge[source "R" target "Y"]
                        edge[source "X" target "Y"]
                        ]
                        """,
        observed_node_names=["X", "B", "Q", "R", "T", "Y"],
        conditional_node_names=[],
        efficient_adjustment={"R", "T"},
        efficient_minimal_adjustment={"R", "T"},
        efficient_mincost_adjustment={"B", "Q"},
        costs=[
            ("B", {"cost": 1}),
            ("Q", {"cost": 1}),
            ("R", {"cost": 2}),
            ("T", {"cost": 2}),
        ],
    ),
    # A graph where optimal, optimal minimal and optimal min cost are different
    "alldiff_example_graph": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                        node[id "Y" label "Y"]
                        node[id "W1" label "W1"]
                        node[id "W2" label "W2"]
                        node[id "W3" label "W3"]
                        node[id "W4" label "W4"]
                        node[id "W5" label "W5"]
                        node[id "W6" label "W6"]
                        node[id "W7" label "W7"]
                        node[id "W8" label "W8"]
                        node[id "W9" label "W9"]
                        node[id "K" label "K"]
                        node[id "O" label "O"]
                        edge[source "K" target "X"]
                        edge[source "X" target "Y"]
                        edge[source "W1" target "K"]
                        edge[source "W2" target "K"]
                        edge[source "W3" target "K"]
                        edge[source "W1" target "W4"]
                        edge[source "W2" target "W5"]
                        edge[source "W3" target "W6"]
                        edge[source "W4" target "W7"]
                        edge[source "W5" target "W8"]
                        edge[source "W6" target "W9"]
                        edge[source "W7" target "Y"]
                        edge[source "W8" target "Y"]
                        edge[source "W9" target "Y"]
                        edge[source "O" target "Y"]
                        ]
                        """,
        observed_node_names=[
            "X",
            "Y",
            "K",
            "O",
            "W1",
            "W2",
            "W3",
            "W4",
            "W5",
            "W6",
            "W7",
            "W8",
            "W9",
        ],
        conditional_node_names=[],
        efficient_adjustment={"O", "W7", "W8", "W9"},
        efficient_minimal_adjustment={"W7", "W8", "W9"},
        efficient_mincost_adjustment={"W7", "W8", "W6"},
        costs=[
            ("W1", {"cost": 2}),
            ("W2", {"cost": 1}),
            ("W3", {"cost": 1}),
            ("W4", {"cost": 1}),
            ("W5", {"cost": 2}),
            ("W6", {"cost": 1}),
            ("W7", {"cost": 1}),
            ("W8", {"cost": 1}),
            ("W9", {"cost": 2}),
            ("K", {"cost": 11}),
            ("O", {"cost": 1}),
        ],
    ),
}
