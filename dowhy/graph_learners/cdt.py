import numpy as np
import networkx as nx
from .utils import *
from cdt.causality.graph import LiNGAM, PC, GES

from dowhy.graph_learners import GraphLearner

class CDT(GraphLearner):

    def __init__(self, data, library_class, *args, **kwargs):
        super().__init__(data, library_class, *args, **kwargs)
        
        self._method = library_class(*args, **kwargs)
    
    def discover(self):
        '''
        Discover causal graph.

        '''
        self.graph = self._method.predict(self._data)
        
    def _get_adjacency_matrix(self):
        '''
        Get adjacency matrix from the networkx graph
        
        '''
        adj_matrix = nx.to_numpy_matrix(self.graph)
        adj_matrix = np.asarray(adj_matrix)
        return adj_matrix

    def _get_dot_graph(self, labels=None):
        '''
        Return graph in DOT format.

        '''
        adj_matrix = self._get_adjacency_matrix()

        # If labels not provided
        if labels is not None:
            self._labels = labels

        graph_dot = make_graph(adj_matrix, self._labels)
        
        # Obtain valid DOT format
        graph_dot = str_to_dot(graph_dot.source)
        return graph_dot
