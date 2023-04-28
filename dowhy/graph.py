"""This module defines the fundamental interfaces and functions related to causal graphs..

Classes and functions in this module should be considered experimental, meaning there might be breaking API changes in
the future.
"""

from abc import abstractmethod
from typing import Any, List

import networkx as nx
from networkx.algorithms.dag import has_cycle
from typing_extensions import Protocol


class HasNodes(Protocol):
    """This protocol defines a trait for classes having nodes."""

    @property
    @abstractmethod
    def nodes(self):
        """:returns Dict[Any, Dict[Any, Any]]"""
        raise NotImplementedError


class HasEdges(Protocol):
    """This protocol defines a trait for classes having edges."""

    @property
    @abstractmethod
    def edges(self):
        """:returns a Dict[Tuple[Any, Any], Dict[Any, Any]]"""
        raise NotImplementedError


class DirectedGraph(HasNodes, HasEdges, Protocol):
    """A protocol representing a directed graph as needed by graphical causal models.

    This protocol specifically defines a subset of the networkx.DiGraph class, which make that class automatically
    compatible with DirectedGraph. While in most cases a networkx.DiGraph is the class of choice when constructing
    a causal graph, anyone can choose to provide their own implementation of the DirectGraph interface.
    """

    @abstractmethod
    def predecessors(self, node):
        raise NotImplementedError


def is_root_node(causal_graph: DirectedGraph, node: Any) -> bool:
    return list(causal_graph.predecessors(node)) == []


def get_ordered_predecessors(causal_graph: DirectedGraph, node: Any) -> List[Any]:
    """This function returns predecessors of a node in a well-defined order.

    This is necessary, because we select subsets of columns in Dataframes by using a node's parents, and these parents
    might not be returned in a reliable order.
    """
    return sorted(causal_graph.predecessors(node))


def node_connected_subgraph_view(g: DirectedGraph, node: Any) -> Any:
    """Returns a view of the provided graph g that contains only nodes connected to the node passed in"""
    # can't use nx.node_connected_component, because it doesn't work with DiGraphs.
    # Hence, a manual loop:
    return nx.induced_subgraph(g, [n for n in g.nodes if nx.has_path(g, n, node)])


def validate_acyclic(causal_graph: DirectedGraph) -> None:
    if has_cycle(causal_graph):
        raise RuntimeError("The graph contains a cycle, but an acyclic graph is expected!")


def validate_node_in_graph(causal_graph: HasNodes, node: Any) -> None:
    if node not in causal_graph.nodes:
        raise ValueError("Node %s can not be found in the given graph!" % node)
