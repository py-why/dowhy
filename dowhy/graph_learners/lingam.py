import numpy as np
import networkx as nx

from dowhy.graph_learners import GraphLearner
from dowhy.utils.graph_operations import *

class LINGAM(GraphLearner):

	def __init__(self, data, library_class, *args, **kwargs):
		super().__init__(data, library_class, *args, **kwargs)
		
		self._method = library_class(*args, **kwargs)

	def learn_graph(self, labels=None):
		'''
		Discover causal graph and return the graph in DOT format.

		'''
		self._method.fit(self._data)
		self._adj_matrix = self._method.adjacency_matrix_

		# If labels provided
		if labels is not None:
			self._labels = labels

		self._graph_dot = adjacency_matrix_to_graph(self._adj_matrix, self._labels)
		
		# Return in valid DOT format
		graph_str = str_to_dot(self._graph_dot.source)
		return graph_str