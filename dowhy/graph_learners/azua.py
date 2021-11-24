import numpy as np
import networkx as nx

from . import get_library_class_object
from dowhy.graph_learners import GraphLearner
from dowhy.utils.graph_operations import *

class AZUA(GraphLearner):
	'''
	Causal discovery using the Azua library.
	Link: https://github.com/microsoft/project-azua/
	'''
	def __init__(self, data, full_method_name, *args, **kwargs):
		super().__init__(data, full_method_name, *args, **kwargs)

                # Loading the relevant class from azua.models
		library_class = get_library_class_object(full_method_name)
		self._method = library_class(*args, **kwargs)

	def learn_graph(self, labels=None):
		'''
		Discover causal graph and return the graph in DOT format.

		'''
                # Azua fit method returns a networkx object
		graph = self._method.fit(self._data)
                return graph
