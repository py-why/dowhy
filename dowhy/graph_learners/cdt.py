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
		graph = self._method.predict(self._data)
		
		# Get adjacency matrix
		self._adj_matrix = nx.to_numpy_matrix(graph)
		self._adj_matrix = np.asarray(self._adj_matrix)

		# If labels not provided
		if labels is not None:
			self._labels = labels

		self._graph_dot = adjacency_matrix_to_graph(self._adj_matrix, self._labels)
		
		# Obtain valid DOT format
		graph_str = str_to_dot(self._graph_dot.source)
		return graph_str
