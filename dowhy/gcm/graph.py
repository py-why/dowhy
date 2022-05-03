"""This module defines the fundamental interfaces and functions related to causal graphs in graphical causal models."""

from abc import abstractmethod, ABC
from typing import Any, List

import networkx as nx
import numpy as np
from networkx.algorithms.dag import has_cycle
from typing_extensions import Protocol

CAUSAL_MECHANISM = 'causal_mechanism'
PARENTS_DURING_FIT = 'parents_during_fit'


class HasNodes(Protocol):
    @property
    @abstractmethod
    def nodes(self):
        """:returns Dict[Any, Dict[Any, Any]]"""
        raise NotImplementedError


class HasEdges(Protocol):
    @property
    @abstractmethod
    def edges(self):
        """:returns a Dict[Tuple[Any, Any], Dict[Any, Any]]"""
        raise NotImplementedError


class DirectedGraph(HasNodes, HasEdges, Protocol):
    @abstractmethod
    def predecessors(self, node):
        raise NotImplementedError


class StochasticModel(ABC):
    @abstractmethod
    def draw_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def clone(self):
        raise NotImplementedError


class ConditionalStochasticModel(ABC):
    @abstractmethod
    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def clone(self):
        raise NotImplementedError


class FunctionalCausalModel(ConditionalStochasticModel):
    """Represents a Functional Causal Model (FCM)"""

    def draw_samples(self, parent_samples: np.ndarray) -> np.ndarray:
        return self.evaluate(parent_samples, self.draw_noise_samples(parent_samples.shape[0]))

    @abstractmethod
    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, parent_samples: np.ndarray, noise_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class InvertibleFunctionalCausalModel(FunctionalCausalModel, ABC):
    @abstractmethod
    def estimate_noise(self, target_samples: np.ndarray, parent_samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def is_root_node(causal_graph: DirectedGraph, node: Any) -> bool:
    return list(causal_graph.predecessors(node)) == []


def get_ordered_predecessors(causal_graph: DirectedGraph, node: Any) -> List[Any]:
    return sorted(causal_graph.predecessors(node))


def node_connected_subgraph_view(g: DirectedGraph, node: Any) -> Any:
    """Returns a view of the provided graph g that contains only nodes connected to the node passed in"""
    # can't use nx.node_connected_component, because it doesn't work with DiGraphs.
    # Hence a manual loop:
    return nx.induced_subgraph(g, [n for n in g.nodes if nx.has_path(g, n, node)])


def clone_causal_models(source: HasNodes, destination: HasNodes):
    for node in destination.nodes:
        if CAUSAL_MECHANISM in source.nodes[node]:
            destination.nodes[node][CAUSAL_MECHANISM] = source.nodes[node][CAUSAL_MECHANISM].clone()


def validate_acyclic(causal_graph: DirectedGraph) -> None:
    if has_cycle(causal_graph):
        raise RuntimeError('The graph contains a cycle, but an acyclic graph is expected!')


def validate_causal_dag(causal_graph: DirectedGraph) -> None:
    validate_acyclic(causal_graph)
    validate_causal_graph(causal_graph)


def validate_causal_graph(causal_graph: DirectedGraph) -> None:
    for node in causal_graph.nodes:
        validate_node(causal_graph, node)


def validate_node(causal_graph: DirectedGraph, node: Any) -> None:
    validate_causal_model_assignment(causal_graph, node)
    validate_local_structure(causal_graph, node)


def validate_causal_model_assignment(causal_graph: DirectedGraph, target_node: Any) -> None:
    validate_node_has_causal_model(causal_graph, target_node)

    causal_model = causal_graph.nodes[target_node][CAUSAL_MECHANISM]

    if is_root_node(causal_graph, target_node):
        if not isinstance(causal_model, StochasticModel):
            raise RuntimeError('Node %s is a root node and, thus, requires a StochasticModel, '
                               'but a %s was found!' % (target_node, causal_model))
    elif not isinstance(causal_model, ConditionalStochasticModel):
        raise RuntimeError('Node %s has parents and, thus, requires a ConditionalStochasticModel, '
                           'but a %s was found!' % (target_node, causal_model))


def validate_local_structure(causal_graph: DirectedGraph, node: Any) -> None:
    if PARENTS_DURING_FIT not in causal_graph.nodes[node] \
            or causal_graph.nodes[node][PARENTS_DURING_FIT] \
            != get_ordered_predecessors(causal_graph, node):
        raise RuntimeError('The causal mechanism of node %s is not fitted to the graphical structure! Fit all'
                           'causal models in the graph first. If the mechanism is already fitted based on the causal'
                           'parents, consider to update the persisted parents for that node manually.' % node)


def validate_node_has_causal_model(causal_graph: HasNodes, node: Any) -> None:
    validate_node_in_graph(causal_graph, node)

    if CAUSAL_MECHANISM not in causal_graph.nodes[node]:
        raise ValueError("Node %s has no assigned causal mechanism!" % node)


def validate_node_in_graph(causal_graph: HasNodes, node: Any) -> None:
    if node not in causal_graph.nodes:
        raise ValueError("Node %s can not be found in the given graph!" % node)
