class CausalDiscovery:
	"""Base class for an causal discovery methods.

	Subclasses implement different discovery methods. All discovery methods are in the package "dowhy.causal_discoverers"

	"""

	def __init__(self, data, method_name, *args, **kwargs):

		self._data = data
		self._method_name = method_name
	
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
		raise NotImplementedError
		