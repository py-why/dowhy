class GraphLearner:
	"""Base class for an causal discovery methods.

	Subclasses implement different discovery methods. All discovery methods are in the package "dowhy.causal_discoverers"

	"""

	def __init__(self, data, library_class, *args, **kwargs):

		self._data = data
		self._labels = list(self._data.columns)
		self._adj_matrix = None
		self._graph_dot = None
		
	def learn_graph(self):
		'''
		Discover causal graph and the graph in DOT format.

		'''
		raise NotImplementedError

	def _get_adjacency_matrix(self):
		'''
		Get adjacency matrix from the networkx graph
		
		'''
		return self._adj_matrix
				
	def render(self, filename, labels=None, view=True):
		print("Rendering graph for %s"%(self._method_name))
		
		self._graph_dot.render(filename, view=view)