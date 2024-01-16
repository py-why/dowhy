from importlib import import_module

from dowhy.graph_learners import GraphLearner
from dowhy.utils.graph_operations import *


class GES(GraphLearner):
    """
    Causal Discovery using GES method.
    Link: https://pypi.org/project/ges/
    """

    def __init__(self, data, full_method_name, *args, **kwargs):
        super().__init__(data, full_method_name, *args, **kwargs)

        library_class = import_module(full_method_name)
        self._method = library_class

    def learn_graph(self, labels=None):
        """
        Discover causal graph and return the graph in DOT format.

        """
        self._adjacency_matrix, self.score = self._method.fit_bic(self._data.to_numpy())
        self._adjacency_matrix = np.asarray(self._adjacency_matrix)

        # If labels provided
        if labels is not None:
            self._labels = labels

        self._graph_dot = adjacency_matrix_to_graph(self._adjacency_matrix, self._labels)

        # Return in valid DOT format
        self._graph_dot = str_to_dot(self._graph_dot.source)
        return self._graph_dot
