import numpy as np
import networkx as nx
# from cdt.causality.graph import LiNGAM, PC, GES

from dowhy.graph_learners import GraphLearner
from dowhy.utils.graph_operations import *

class CDT(GraphLearner):

    def __init__(self, data, library_class, *args, **kwargs):
        super().__init__(data, library_class, *args, **kwargs)
        
        self._method = library_class(*args, **kwargs)
    
    def learn_graph(self, labels=None):
        '''
        Discover causal graph and return the graph in DOT format.

        '''
        self.graph = self._method.predict(self._data)
        
        # Get adjacency matrix
        self.adj_matrix = nx.to_numpy_matrix(self.graph)
        self.adj_matrix = np.asarray(self.adj_matrix)

        # If labels not provided
        if labels is not None:
            self._labels = labels

        graph_dot = adjacency_matrix_to_graph(self.adj_matrix, self._labels)
        
        # Obtain valid DOT format
        graph_dot = str_to_dot(graph_dot.source)
        return graph_dot

    def _get_adjacency_matrix(self):
        '''
        Get adjacency matrix from the networkx graph
        
        '''
        return self.adj_matrix
