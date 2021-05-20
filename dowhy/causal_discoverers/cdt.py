import numpy as np
import networkx as nx
from .utils import *
from cdt.causality.graph import LiNGAM, PC, GES

from dowhy.causal_discovery import CausalDiscovery

functions = {
    'lingam' : LiNGAM,
    'pc' : PC,
    'ges' : GES,
}

class CDT(CausalDiscovery):

    def __init__(self, data, method_name, *args, **kwargs):
        super().__init__(data, method_name, *args, **kwargs)
        
        self._method = functions[method_name](*args, **kwargs)
    
    def discover(self):
        '''
        Discover causal graph.

        '''
        self.labels = list(self._data.columns)
        self.graph = self._method.predict(self._data)
        return self.graph

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
            self.labels = labels

        graph_dot = make_graph(adj_matrix, self.labels)
        
        # Obtain valid DOT format
        graph_dot = str_to_dot(graph_dot.source)
        return graph_dot
        
    def render(self, filename, labels=None, view=True):
        print("Graph for %s"%(self._method_name))
        graph_dot = self._get_dot_graph(labels=labels)
        graph_dot.render(filename, view=view)
        