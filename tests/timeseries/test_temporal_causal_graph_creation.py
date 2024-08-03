import unittest
from io import StringIO

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.utils.timeseries import create_graph_from_csv, create_graph_from_dot_format, create_graph_from_networkx_array


class TestCreateGraphFromCSV(unittest.TestCase):

    def test_basic_functionality(self):
        csv_content = """node1,node2,time_lag
A,B,5
B,C,2
A,C,7"""
        df = pd.read_csv(StringIO(csv_content))
        df.to_csv("test.csv", index=False)

        graph = create_graph_from_csv("test.csv")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(graph["A"]["B"]["time_lag"], (5,))
        self.assertEqual(graph["B"]["C"]["time_lag"], (2,))
        self.assertEqual(graph["A"]["C"]["time_lag"], (7,))

    def test_multiple_time_lags(self):
        csv_content = """node1,node2,time_lag
A,B,5
A,B,3
A,C,7"""
        df = pd.read_csv(StringIO(csv_content))
        df.to_csv("test.csv", index=False)

        graph = create_graph_from_csv("test.csv")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 2)
        self.assertEqual(graph["A"]["B"]["time_lag"], (5, 3))
        self.assertEqual(graph["A"]["C"]["time_lag"], (7,))

    def test_invalid_time_lag(self):
        csv_content = """node1,node2,time_lag
A,B,five
B,C,2
A,C,7"""
        df = pd.read_csv(StringIO(csv_content))
        df.to_csv("test.csv", index=False)

        graph = create_graph_from_csv("test.csv")

        self.assertIsNone(graph)

    def test_empty_csv(self):
        csv_content = """node1,node2,time_lag"""
        df = pd.read_csv(StringIO(csv_content))
        df.to_csv("test.csv", index=False)

        graph = create_graph_from_csv("test.csv")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 0)

    def test_self_loop(self):
        csv_content = """node1,node2,time_lag
A,A,5"""
        df = pd.read_csv(StringIO(csv_content))
        df.to_csv("test.csv", index=False)

        graph = create_graph_from_csv("test.csv")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 1)
        self.assertEqual(graph["A"]["A"]["time_lag"], (5,))


class TestCreateGraphFromDotFormat(unittest.TestCase):

    def setUp(self):
        # Helper method to create a DOT file from string content
        def create_dot_file(dot_content, file_name="test.dot"):
            with open(file_name, "w") as f:
                f.write(dot_content)

        self.create_dot_file = create_dot_file

    def test_basic_functionality(self):
        dot_content = """digraph G {
A -> B [label="(5)"];
B -> C [label="(2)"];
A -> C [label="(7)"];
}"""
        self.create_dot_file(dot_content)
        graph = create_graph_from_dot_format("test.dot")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(graph["A"]["B"]["time_lag"], (5,))
        self.assertEqual(graph["B"]["C"]["time_lag"], (2,))
        self.assertEqual(graph["A"]["C"]["time_lag"], (7,))

    def test_multiple_time_lags(self):
        dot_content = """digraph G {
A -> B [label="(5,3)"];
A -> C [label="(7)"];
}"""
        self.create_dot_file(dot_content)
        graph = create_graph_from_dot_format("test.dot")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 2)
        self.assertEqual(graph["A"]["B"]["time_lag"], (5, 3))
        self.assertEqual(graph["A"]["C"]["time_lag"], (7,))

    def test_invalid_time_lag(self):
        dot_content = """digraph G {
A -> B [label="(five)"];
B -> C [label="(2)"];
A -> C [label="(7)"];
}"""
        self.create_dot_file(dot_content)
        graph = create_graph_from_dot_format("test.dot")

        self.assertIsNone(graph)

    def test_empty_dot(self):
        dot_content = """digraph G {}"""
        self.create_dot_file(dot_content)
        graph = create_graph_from_dot_format("test.dot")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 0)

    def test_self_loop(self):
        dot_content = """digraph G {
A -> A [label="(5)"];
}"""
        self.create_dot_file(dot_content)
        graph = create_graph_from_dot_format("test.dot")

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 1)
        self.assertEqual(graph["A"]["A"]["time_lag"], (5,))


class TestCreateGraphFromNetworkxArray(unittest.TestCase):

    def test_basic_functionality(self):
        array = np.zeros((3, 3, 2), dtype=object)
        array[0, 1, 0] = "-->"
        array[1, 2, 0] = "-->"
        array[0, 2, 1] = "-->"

        var_names = ["X1", "X2", "X3"]

        graph = create_graph_from_networkx_array(array, var_names)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 3)
        self.assertTrue(graph.has_edge("X1", "X2"))
        self.assertEqual(graph["X1"]["X2"]["time_lag"], (0,))
        self.assertTrue(graph.has_edge("X1", "X3"))
        self.assertEqual(graph["X1"]["X3"]["time_lag"], (1,))
        self.assertTrue(graph.has_edge("X2", "X3"))
        self.assertEqual(graph["X2"]["X3"]["time_lag"], (0,))

    def test_multiple_time_lags(self):
        array = np.zeros((3, 3, 3), dtype=object)
        array[0, 1, 0] = "-->"
        array[0, 1, 1] = "-->"
        array[1, 2, 2] = "-->"

        var_names = ["X1", "X2", "X3"]

        graph = create_graph_from_networkx_array(array, var_names)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 2)
        self.assertTrue(graph.has_edge("X1", "X2"))
        self.assertEqual(graph["X1"]["X2"]["time_lag"], (0, 1))
        self.assertTrue(graph.has_edge("X2", "X3"))
        self.assertEqual(graph["X2"]["X3"]["time_lag"], (2,))

    def test_invalid_link_type_oo(self):
        array = np.zeros((2, 2, 1), dtype=object)
        array[0, 1, 0] = "o-o"

        var_names = ["X1", "X2"]

        with self.assertRaises(ValueError):
            create_graph_from_networkx_array(array, var_names)

    def test_invalid_link_type_xx(self):
        array = np.zeros((2, 2, 1), dtype=object)
        array[0, 1, 0] = "x-x"

        var_names = ["X1", "X2"]

        with self.assertRaises(ValueError):
            create_graph_from_networkx_array(array, var_names)

    def test_empty_array(self):
        array = np.zeros((0, 0, 0), dtype=object)
        var_names = []

        graph = create_graph_from_networkx_array(array, var_names)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.nodes()), 0)
        self.assertEqual(len(graph.edges()), 0)

    def test_self_loop(self):
        array = np.zeros((2, 2, 1), dtype=object)
        array[0, 0, 0] = "-->"

        var_names = ["X1", "X2"]

        graph = create_graph_from_networkx_array(array, var_names)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 0)


if __name__ == "__main__":
    unittest.main()
