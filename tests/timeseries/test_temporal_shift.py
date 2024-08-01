import unittest
import pandas as pd
import networkx as nx
from typing import List, Optional

from dowhy.timeseries.temporal_shift import add_lagged_edges

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
        self.assertTrue(new_graph.has_node("A_-4"))
        self.assertTrue(new_graph.has_edge("B_-1", "C_0"))
        self.assertTrue(new_graph.has_edge("B_-3", "C_0"))
        self.assertTrue(new_graph.has_edge("A_-2", "B_-1"))
        self.assertTrue(new_graph.has_edge("A_-4", "B_-3"))

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
        self.assertTrue(new_graph.has_node("A_-3"))
        self.assertTrue(new_graph.has_edge("A_-3", "C_0"))

    def test_no_time_lag(self):
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        new_graph = add_lagged_edges(graph, "C")
        
        self.assertIsInstance(new_graph, nx.DiGraph)
        self.assertEqual(len(new_graph.nodes()), 0)
        self.assertEqual(len(new_graph.edges()), 0)

if __name__ == '__main__':
    unittest.main()
