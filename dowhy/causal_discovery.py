class GraphLearner:
	"""Base class for an causal discovery methods.

	Subclasses implement different discovery methods. All discovery methods are in the package "dowhy.causal_discoverers"

	"""

	def __init__(self, data, library_class, *args, **kwargs):

		self._data = data
		self._labels = list(self._data.columns)
		
	def discover(self):
		'''
		Discover causal graph.

		'''
		raise NotImplementedError

	def _get_adjacency_matrix(self):
		'''
		Get adjacency matrix from the networkx graph
		
		'''
		raise NotImplementedError
		
	def _get_dot_graph(self, labels=None):
		'''
		Return graph in DOT format.

		'''
		raise NotImplementedError
		
	def render(self, filename, labels=None, view=True):
		print("Rendering graph for %s"%(self._method_name))
		
		# If labels not provided
		if labels is not None:
			self._labels = labels

		graph_dot = self._get_dot_graph(labels=self._labels)
		graph_dot.render(filename, view=view)