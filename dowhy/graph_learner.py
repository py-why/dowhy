class GraphLearner:
    """Base class for causal discovery methods.

    Subclasses implement different discovery methods. All discovery methods are in the package "dowhy.causal_discoverers"

    """

    def __init__(self, data, library_class, *args, **kwargs):

        self._data = data
        self._labels = list(self._data.columns)
        self._adjacency_matrix = None
        self._graph_dot = None

    def learn_graph(self):
        """
        Discover causal graph and the graph in DOT format.

        """
        raise NotImplementedError
