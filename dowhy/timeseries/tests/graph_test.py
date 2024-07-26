import unittest
import networkx as nx
import pandas as pd
import numpy as np
from io import StringIO
from dowhy.utils.timeseries import create_graph_from_user, create_graph_from_csv, create_graph_from_dot_format, create_graph_from_array

# Import the functions from your module
# from your_module import create_graph_from_user, create_graph_from_csv, create_graph_from_dot_format, create_graph_from_array

class TestGraphFunctions(unittest.TestCase):

    def test_create_graph_from_csv(self):
        csv_content = """node1,node2,time_lag
A,B,5
B,C,2
A,C,7"""
        df = pd.read_csv(StringIO(csv_content))
        df.to_csv("test.csv", index=False)
        
        graph = create_graph_from_csv("test.csv")
        
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(graph['A']['B']['time_lag'], 5)
        self.assertEqual(graph['B']['C']['time_lag'], 2)
        self.assertEqual(graph['A']['C']['time_lag'], 7)

    def test_create_graph_from_dot_format(self):
        dot_content = """digraph G {
A -> B [label="5"];
B -> C [label="2"];
A -> C [label="7"];
}"""
        with open("test.dot", "w") as f:
            f.write(dot_content)
        
        graph = create_graph_from_dot_format("test.dot")
        
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 3)
        self.assertEqual(graph['A']['B']['time_lag'], 5)
        self.assertEqual(graph['B']['C']['time_lag'], 2)
        self.assertEqual(graph['A']['C']['time_lag'], 7)

    def test_create_graph_from_array(self):
        array = np.zeros((3, 3, 2), dtype=object)
        array[0, 1, 0] = '-->'
        array[1, 2, 0] = '-->'
        array[0, 1, 1] = '-->'
        array[2, 0, 1] = '-->'
        
        var_names = ['X1', 'X2', 'X3']
        
        graph = create_graph_from_array(array, var_names)
        
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.edges()), 3)
        self.assertTrue(graph.has_edge('X1', 'X2'))
        self.assertEqual(graph['X1']['X2']['type'], 'contemporaneous')
        self.assertTrue(graph.has_edge('X1', 'X2'))
        self.assertEqual(graph['X1']['X2']['time_lag'], 1)
        self.assertTrue(graph.has_edge('X3', 'X1'))
        self.assertEqual(graph['X3']['X1']['time_lag'], 1)

if __name__ == '__main__':
    unittest.main()
