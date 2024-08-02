import unittest
from typing import List, Optional

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal

from dowhy.timeseries.temporal_shift import add_lagged_edges, shift_columns_by_lag_using_unrolled_graph


class TestAddLaggedEdges(unittest.TestCase):

    def test_basic_functionality(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", time_lag=(1,))
        graph.add_edge("B", "C", time_lag=(2,))

        new_graph = add_lagged_edges(graph, "C")

        self.assertIsInstance(new_graph, nx.DiGraph)
        self.assertTrue(new_graph.has_node("C_0"))
        self.assertTrue(new_graph.has_node("B_-2"))
        self.assertTrue(new_graph.has_node("A_-3"))
        self.assertTrue(new_graph.has_edge("B_-2", "C_0"))
        self.assertTrue(new_graph.has_edge("A_-3", "B_-2"))

    def test_multiple_time_lags(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", time_lag=(1, 2))
        graph.add_edge("B", "C", time_lag=(1, 3))

        new_graph = add_lagged_edges(graph, "C")

        self.assertIsInstance(new_graph, nx.DiGraph)
        self.assertTrue(new_graph.has_node("C_0"))
        self.assertTrue(new_graph.has_node("B_-1"))
        self.assertTrue(new_graph.has_node("B_-3"))
        self.assertTrue(new_graph.has_node("A_-2"))
        self.assertTrue(new_graph.has_node("A_-3"))
        self.assertTrue(new_graph.has_node("A_-4"))
        self.assertTrue(new_graph.has_node("A_-5"))
        self.assertTrue(new_graph.has_edge("B_-1", "C_0"))
        self.assertTrue(new_graph.has_edge("B_-3", "C_0"))
        self.assertTrue(new_graph.has_edge("A_-2", "B_-1"))
        self.assertTrue(new_graph.has_edge("A_-4", "B_-3"))
        self.assertTrue(new_graph.has_edge("B_-3", "B_-1"))
        self.assertTrue(new_graph.has_edge("A_-5", "A_-4"))
        self.assertTrue(new_graph.has_edge("A_-4", "A_-3"))
        self.assertTrue(new_graph.has_edge("A_-3", "A_-2"))

    def test_complex_graph_structure(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B", time_lag=(1,))
        graph.add_edge("B", "C", time_lag=(2,))
        graph.add_edge("A", "C", time_lag=(3,))

        new_graph = add_lagged_edges(graph, "C")

        self.assertIsInstance(new_graph, nx.DiGraph)
        self.assertTrue(new_graph.has_node("C_0"))
        self.assertTrue(new_graph.has_node("B_-2"))
        self.assertTrue(new_graph.has_node("A_-3"))
        self.assertTrue(new_graph.has_edge("B_-2", "C_0"))
        self.assertTrue(new_graph.has_edge("A_-3", "B_-2"))
        self.assertTrue(new_graph.has_edge("A_-3", "C_0"))

    def test_no_time_lag(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        new_graph = add_lagged_edges(graph, "C")

        self.assertIsInstance(new_graph, nx.DiGraph)
        self.assertEqual(len(new_graph.nodes()), 0)
        self.assertEqual(len(new_graph.edges()), 0)


class TestShiftColumnsByLagUsingUnrolledGraph(unittest.TestCase):

    def test_basic_functionality(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

        unrolled_graph = nx.DiGraph()
        unrolled_graph.add_nodes_from(["A_0", "A_-1", "B_0", "B_-2"])

        expected_df = pd.DataFrame(
            {"A_0": [1, 2, 3, 4, 5], "A_-1": [0, 1, 2, 3, 4], "B_0": [5, 4, 3, 2, 1], "B_-2": [0, 0, 5, 4, 3]}
        )

        result_df = shift_columns_by_lag_using_unrolled_graph(df, unrolled_graph)

        assert_frame_equal(result_df, expected_df)

    def test_complex_graph_structure(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1], "C": [1, 3, 5, 7, 9]})

        unrolled_graph = nx.DiGraph()
        unrolled_graph.add_nodes_from(["A_0", "A_-1", "B_0", "B_-2", "C_-1", "C_-3"])

        expected_df = pd.DataFrame(
            {
                "A_0": [1, 2, 3, 4, 5],
                "A_-1": [0, 1, 2, 3, 4],
                "B_0": [5, 4, 3, 2, 1],
                "B_-2": [0, 0, 5, 4, 3],
                "C_-1": [0, 1, 3, 5, 7],
                "C_-3": [0, 0, 0, 1, 3],
            }
        )

        result_df = shift_columns_by_lag_using_unrolled_graph(df, unrolled_graph)

        assert_frame_equal(result_df, expected_df)

    def test_invalid_node_format(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

        unrolled_graph = nx.DiGraph()
        unrolled_graph.add_nodes_from(["A_0", "B_invalid"])

        expected_df = pd.DataFrame({"A_0": [1, 2, 3, 4, 5]})

        result_df = shift_columns_by_lag_using_unrolled_graph(df, unrolled_graph)

        assert_frame_equal(result_df, expected_df)

    def test_non_matching_columns(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})

        unrolled_graph = nx.DiGraph()
        unrolled_graph.add_nodes_from(["C_0", "C_-1"])

        expected_df = pd.DataFrame()

        result_df = shift_columns_by_lag_using_unrolled_graph(df, unrolled_graph)

        assert_frame_equal(result_df, expected_df)


if __name__ == "__main__":
    unittest.main()
