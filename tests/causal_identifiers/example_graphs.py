"""
The example below illustrate some of the common examples of causal graph in books.
This file is meant to group all of the graph definitions as well as the expected results of identification algorithms in one place.
Each example graph is contained of the following values:

    * graph_str - The graph string in GML format.
    * observed_variables - A list of observed variables in the graph. This will be used to test no unobserved variables are offered in the solution.
    * biased_sets - The sets that we shouldn't get in the output as they incur biased estimates of the causal effect.
    * minimal_adjustment_sets - Sets of observed variables that should be returned when 'minimal-adjustment' is specified as the backdoor method.
        If no adjustment is necessary given the graph, minimal adjustment set should be the empty set.
    * maximal_adjustment_sets - Sets of observed variables that should be returned when 'maximal-adjustment' is specified as the backdoor method.
"""

TEST_GRAPH_SOLUTIONS = {
    # Example is selected from Pearl J. "Causality" 2nd Edition, from chapter 3.3.1 on backoor criterion.
    "pearl_backdoor_example_graph": dict(
        graph_str="""graph[directed 1 node[id "Z1" label "Z1"]
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
        observed_variables=["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "X", "Y"],
        biased_sets=[{"Z4"}, {"Z6"}, {"Z5"}, {"Z2"}, {"Z1"}, {"Z3"}, {"Z1", "Z3"}, {"Z2", "Z5"}, {"Z1", "Z2"}],
        minimal_adjustment_sets=[{"Z1", "Z4"}, {"Z2", "Z4"}, {"Z3", "Z4"}, {"Z5", "Z4"}],
        maximal_adjustment_sets=[{"Z1", "Z2", "Z3", "Z4", "Z5"}],
        direct_maximal_adjustment_sets=[{"Z1", "Z2", "Z3", "Z4", "Z5", "Z6"}],
    ),
    "simple_selection_bias_graph": dict(
        graph_str="""graph[directed 1 node[id "Z1" label "Z1"]
                    node[id "X" label "X"]
                    node[id "Y" label "Y"]
                    edge[source "X" target "Y"]
                    edge[source "X" target "Z1"]
                    edge[source "Y" target "Z1"]]
                    """,
        observed_variables=["Z1", "X", "Y"],
        biased_sets=[
            {
                "Z1",
            }
        ],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{}],
    ),
    "simple_no_confounder_graph": dict(
        graph_str="""graph[directed 1 node[id "Z1" label "Z1"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                edge[source "X" target "Y"]
                edge[source "Z1" target "X"]]
                """,
        observed_variables=["Z1", "X", "Y"],
        biased_sets=[],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[
            {
                "Z1",
            }
        ],
    ),
    # The following simpsons paradox examples are taken from Pearl, J {2013}. "Understanding Simpson’s Paradox" - http://ftp.cs.ucla.edu/pub/stat_ser/r414.pdf
    "pearl_simpsons_paradox_1c": dict(
        graph_str="""graph[directed 1 node[id "Z" label "Z"]
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
        biased_sets=[
            {
                "Z",
            }
        ],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{}],
    ),
    "pearl_simpsons_paradox_1d": dict(
        graph_str="""graph[directed 1 node[id "Z" label "Z"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L1" label "L1"]
                edge[source "X" target "Y"]
                edge[source "L1" target "X"]
                edge[source "L1" target "Z"]
                edge[source "Z" target "Y"]]
                """,
        observed_variables=["Z", "X", "Y"],
        biased_sets=[],
        minimal_adjustment_sets=[
            {
                "Z",
            }
        ],
        maximal_adjustment_sets=[
            {
                "Z",
            }
        ],
    ),
    "pearl_simpsons_paradox_2a": dict(
        graph_str="""graph[directed 1 node[id "Z" label "Z"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L" label "L"]
                edge[source "X" target "Y"]
                edge[source "X" target "Z"]
                edge[source "L" target "Z"]
                edge[source "L" target "Y"]]
                """,
        observed_variables=["Z", "X", "Y"],
        biased_sets=[
            {
                "Z",
            }
        ],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{}],
    ),
    "pearl_simpsons_paradox_2b": dict(
        graph_str="""graph[directed 1 node[id "Z" label "Z"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L" label "L"]
                edge[source "X" target "Y"]
                edge[source "Z" target "X"]
                edge[source "L" target "X"]
                edge[source "L" target "Y"]]""",
        observed_variables=["Z", "X", "Y"],
        biased_sets=[],
        minimal_adjustment_sets=[],
        maximal_adjustment_sets=[],  # Should this be {"Z"}?
    ),
    "pearl_simpsons_paradox_2b_L_observed": dict(
        graph_str="""graph[directed 1 node[id "Z" label "Z"]
                node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "L" label "L"]
                edge[source "X" target "Y"]
                edge[source "Z" target "X"]
                edge[source "L" target "X"]
                edge[source "L" target "Y"]]""",
        observed_variables=["Z", "X", "Y", "L"],
        biased_sets=[],
        minimal_adjustment_sets=[{"L"}],
        maximal_adjustment_sets=[{"L", "Z"}],
    ),
    "pearl_simpsons_machine_lvl1": dict(
        graph_str="""graph[directed 1 node[id "Z1" label "Z1"]
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
        biased_sets=[
            {
                "Z2",
            },
            {"Z1", "Z2"},
        ],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{"Z1", "Z2", "Z3"}],
    ),
    # The following are examples given in the "Book of Why" by Judea Pearl, chapter "The Do-operator and the Back-Door Criterion"
    "book_of_why_game2": dict(
        graph_str="""graph[directed 1 node[id "A" label "A"]
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
        observed_variables=["A", "B", "C", "D", "E", "X", "Y"],
        biased_sets=[
            {
                "B",
            },
            {
                "C",
            },
            {"B", "C"},
        ],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{"A", "B", "C", "D"}],
        direct_maximal_adjustment_sets=[{"A", "B", "C", "D", "E"}],
    ),
    "book_of_why_game5": dict(
        graph_str="""graph[directed 1 node[id "A" label "A"]
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
        observed_variables=["A", "B", "C", "X", "Y"],
        biased_sets=[
            {
                "B",
            }
        ],
        minimal_adjustment_sets=[{"C"}],
        maximal_adjustment_sets=[{"A", "B", "C"}],
    ),
    "book_of_why_game5_C_is_unobserved": dict(
        graph_str="""graph[directed 1 node[id "A" label "A"]
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
        observed_variables=["A", "B", "X", "Y"],
        biased_sets=[
            {
                "B",
            }
        ],
        minimal_adjustment_sets=[{"A", "B"}],
        maximal_adjustment_sets=[{"A", "B"}],
    ),
    "no_treatment_but_valid_maximal_set": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "Z1" label "Z1"]
                node[id "Z2" label "Z2"]
                edge[source "X" target "Y"]
                edge[source "X" target "Z1"]
                edge[source "Z1" target "Y"]
                edge[source "Z2" target "Z1"]]
                """,
        observed_variables=["Z1", "Z2", "X", "Y"],
        biased_sets=[],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{"Z2"}],
        direct_maximal_adjustment_sets=[{"Z2", "Z1"}],
    ),
    "common_cause_of_mediator1": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "U" label "U"]
                node[id "Z" label "Z"]
                node[id "M" label "M"]
                edge[source "X" target "M"]
                edge[source "M" target "Y"]
                edge[source "U" target "X"]
                edge[source "U" target "Z"]
                edge[source "Z" target "M"]]
                """,
        observed_variables=["X", "Y", "Z", "M"],
        biased_sets=[],
        minimal_adjustment_sets=[{"Z"}],
        maximal_adjustment_sets=[{"Z"}],
        direct_maximal_adjustment_sets=[{"Z", "M"}],
    ),
    "common_cause_of_mediator2": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "U" label "U"]
                node[id "Z" label "Z"]
                node[id "M" label "M"]
                edge[source "X" target "M"]
                edge[source "M" target "Y"]
                edge[source "U" target "Z"]
                edge[source "U" target "M"]
                edge[source "Z" target "X"]]
                """,
        observed_variables=["X", "Y", "Z", "M"],
        biased_sets=[],
        minimal_adjustment_sets=[{"Z"}],
        maximal_adjustment_sets=[{"Z"}],
        direct_maximal_adjustment_sets=[{"Z", "M"}],
    ),
    "mbias_with_unobserved": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "U1" label "U1"]
                node[id "U2" label "U2"]
                node[id "Z" label "Z"]
                node[id "M" label "M"]
                edge[source "X" target "Y"]
                edge[source "U1" target "X"]
                edge[source "U1" target "M"]
                edge[source "U2" target "M"]
                edge[source "U2" target "Y"]
                edge[source "Z" target "X"]]
                """,
        observed_variables=["X", "Y", "Z", "M"],
        biased_sets=[{"Z", "M"}, {"M"}],
        minimal_adjustment_sets=[{}],
        maximal_adjustment_sets=[{"Z"}],
    ),
    "iv": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "U" label "U"]
                node[id "Z" label "Z"]
                edge[source "X" target "Y"]
                edge[source "U" target "X"]
                edge[source "U" target "Y"]
                edge[source "Z" target "X"]]
                """,
        observed_variables=["X", "Y", "Z"],
        biased_sets=[{"Z"}],
        minimal_adjustment_sets=[],
        maximal_adjustment_sets=[],
    ),
    "mediator": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "M" label "M"]
                node[id "W" label "W"]
                edge[source "X" target "Y"]
                edge[source "W" target "X"]
                edge[source "W" target "Y"]
                edge[source "X" target "M"]
                edge[source "M" target "Y"]]
                """,
        observed_variables=["X", "Y", "M", "W"],
        biased_sets=[{"M"}],
        minimal_adjustment_sets=[{"W"}],
        maximal_adjustment_sets=[{"W"}],
        direct_maximal_adjustment_sets=[{"W", "M"}],
    ),
    "mediator-with-conf": dict(
        graph_str="""graph[directed 1 node[id "X" label "X"]
                node[id "Y" label "Y"]
                node[id "M" label "M"]
                node[id "W1" label "W1"]
                node[id "W2" label "W2"]
                edge[source "X" target "Y"]
                edge[source "W1" target "X"]
                edge[source "W1" target "Y"]
                edge[source "W2" target "M"]
                edge[source "W2" target "Y"]
                edge[source "X" target "M"]
                edge[source "M" target "Y"]]
                """,
        observed_variables=["X", "Y", "M", "W1", "W2"],
        biased_sets=[{"M"}, {"M", "W1"}],
        minimal_adjustment_sets=[{"W1"}],
        maximal_adjustment_sets=[{"W1", "W2"}],
        direct_maximal_adjustment_sets=[{"W1", "M", "W2"}],
    ),
}

TEST_GRAPH_SOLUTIONS_COMPLETE_ADJUSTMENT = {
    # Example is selected from Shpitser et al. "On the Validity of Covariate Adjustment for Estimating Causal
    # Effects", figure 1(b).
    "shpitser_simple_non_backdoor_adjustment_set": dict(
        graph_str="digraph{Z;X;Y; X->Z;X->Y}",
        observed_variables=["Z", "X", "Y"],
        action_nodes=["X"],
        outcome_nodes=["Y"],
        minimal_adjustment_sets=[set()],
        exhaustive_adjustment_sets=[{"Z"}, set()],
    ),
    # Example is selected from van der Zander et al. "Constructing Separators and Adjustment Sets in Ancestral
    # Graphs", figure 2.
    "van_der_zander_minimal_non_backdoor_adjustment_set": dict(
        graph_str="digraph{Z1;Z2;X1;X2;Y1;Y2; X1->Y1;X1->Z1;Z1->Z2;Z2->X2;Y2->Z2}",
        observed_variables=["Z1", "Z2", "X1", "X2", "Y1", "Y2"],
        action_nodes=["X1", "X2"],
        outcome_nodes=["Y1", "Y2"],
        minimal_adjustment_sets=[{"Z1", "Z2"}],
        exhaustive_adjustment_sets=[{"Z1", "Z2"}],
    ),
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
        outcome_nodes=["Y"],
        minimal_adjustment_sets=[{"V1", "V2"}],
    ),
    # Example is selected from Perković et al. "Complete Graphical Characterization and Construction of
    # Adjustment Sets in Markov Equivalence Classes of Ancestral Graphs", Example 9 (in Section 5).
    "perkovic_example_9": dict(
        graph_str="digraph{V1;V2;V3;X1;X2;Y; X1->Y;V1->X1;V2->X1;V3->V2;V3->Y;X2->V1;X2->Y}",
        observed_variables=["V1", "V2", "V3", "X1", "X2", "Y"],
        action_nodes=["X1", "X2"],
        outcome_nodes=["Y"],
        minimal_adjustment_sets=[{"V2"}, {"V3"}],
        exhaustive_adjustment_sets=[{"V2"}, {"V3"}, {"V2", "V3"}, {"V1", "V3"}, {"V1", "V2"}, {"V1", "V2", "V3"}],
    ),
}


TEST_FRONTDOOR_GRAPH_SOLUTIONS = {
    "valid_singleton": dict(
        graph_str="digraph {X; Y; Z; M1; X->M1; Z->X; Z->Y; M1->Y;}",
        observed_variables=["X", "Y", "M1"],
        valid_frontdoor_sets=[{"M1"}],
        invalid_frontdoor_sets=[{"Z"}],
    ),
    "valid_doubleton": dict(
        graph_str="digraph {X; Y; Z; M1; M2; X->M1; X->M2; Z->X; Z->Y; M1->Y; M2->Y}",
        observed_variables=["X", "Y", "M1", "M2"],
        valid_frontdoor_sets=[
            {"M1", "M2"},
        ],
        invalid_frontdoor_sets=[{"Z"}, {"M1"}, {"M2"}],
    ),
    "no_frontdoor": dict(
        graph_str="digraph{E;X;R;M;Y; E->X;E->R;X->M;R->M;M->Y}",
        observed_variables=["X", "R", "M", "Y"],
        valid_frontdoor_sets=[],
        invalid_frontdoor_sets=[{"M"}, {"M", "E"}, {"M", "R"}],
    ),
    "no_frontdoor_simple": dict(
        graph_str="digraph{X;Y;D;B;X->B;D->B;B->Y;D->Y}",
        observed_variables=["X", "B", "Y"],
        valid_frontdoor_sets=[],
        invalid_frontdoor_sets=[{"B"}, {"D"}],
    ),
    "no_frontdoor_in_obs": dict(
        graph_str="digraph {X; Y; Z; M1; M2; X->M1; X->M2; Z->X; Z->Y; M1->Y; M2->Y}",
        observed_variables=["X", "Y", "M1", "Z"],
        valid_frontdoor_sets=[],
        invalid_frontdoor_sets=[{"Z"}, {"M1"}, {"M2"}, {"M1", "M2"}],
    ),
    # This example is reproduced from the generalized_adjustment examples, and is
    # added to test that the frontdoor criterion successfully filters out all the action
    # nodes as ineligible variables.
    "perkovic_example_9_multiple_action_nodes_no_frontdoor": dict(
        graph_str="digraph{V1;V2;V3;X1;X2;Y; X1->Y;V1->X1;V2->X1;V3->V2;V3->Y;X2->V1;X2->Y}",
        observed_variables=["V1", "V2", "V3", "X1", "X2", "Y"],
        action_nodes=["X1", "X2"],
        outcome_nodes=["Y"],
        valid_frontdoor_sets=[],
        invalid_frontdoor_sets=[{"V1"}, {"V2"}, {"V3"}],
    ),
}
