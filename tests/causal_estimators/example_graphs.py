TEST_GRAPHS = {
    # Example is selected from Perković et al. "Complete Graphical Characterization and Construction of
    # Adjustment Sets in Markov Equivalence Classes of Ancestral Graphs", Example 8 (in Section 5).
    "perkovic_example_8": dict(
        graph_str="""graph[directed 1 node[id "X1" label "X1"]
                    node[id "X2" label "X2"]
                    node[id "Y" label "Y"]
                    node[id "V1" label "V1"]
                    node[id "V2" label "V2"]
                    node[id "V3" label "V3"]
                    node[id "V4" label "V4"]
                    node[id "V5" label "V5"]
                    node[id "L" label "L"]
                    edge[source "V5" target "X1"]
                    edge[source "V4" target "X1"]
                    edge[source "X1" target "V1"]
                    edge[source "V1" target "V2"]
                    edge[source "V2" target "X2"]
                    edge[source "X2" target "Y"]
                    edge[source "X1" target "V3"]
                    edge[source "V3" target "Y"]
                    edge[source "L" target "V3"]
                    edge[source "L" target "V2"]]
                    """,
        observed_variables=["V1", "V2", "V3", "V4", "V5", "X1", "X2", "Y"],
        action_nodes=["X1", "X2"],
        outcome_node="Y",
    ),
}
