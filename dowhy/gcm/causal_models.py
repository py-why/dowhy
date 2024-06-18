"""This module defines the fundamental classes for graphical causal models (GCMs)."""

from typing import Any, Callable, Optional, Union

import networkx as nx

from dowhy.gcm.causal_mechanisms import (
    ConditionalStochasticModel,
    FunctionalCausalModel,
    InvertibleFunctionalCausalModel,
    StochasticModel,
)
from dowhy.graph import (
    DirectedGraph,
    HasNodes,
    get_ordered_predecessors,
    is_root_node,
    validate_acyclic,
    validate_node_in_graph,
)

# This constant is used as key when storing/accessing models as causal mechanisms in graph node attributes
CAUSAL_MECHANISM = "causal_mechanism"
# This constant is used as key when storing the parents of a node during fitting. It's used for validation purposes
# afterwards.
PARENTS_DURING_FIT = "parents_during_fit"


class ProbabilisticCausalModel:
    """Represents a probabilistic graphical causal model, i.e. it combines a graphical representation of causal
    causal relationships and corresponding causal mechanism for each node describing the data generation process. The
    causal mechanisms can be any general stochastic models."""

    def __init__(
        self,
        graph: Optional[DirectedGraph] = None,
        graph_copier: Callable[[DirectedGraph], DirectedGraph] = nx.DiGraph,
        remove_existing_mechanisms: bool = False,
    ):
        """
        :param graph: Optional graph object to be used as causal graph.
        :param graph_copier: Optional function that can copy a causal graph. Defaults to a networkx.DiGraph
                             constructor.
        :param remove_existing_mechanisms: If True, removes existing causal mechanisms assigned to nodes if they exist.
                                           Otherwise, does not modify graph.
        """
        # Todo: Remove after https://github.com/py-why/dowhy/pull/943.
        from dowhy.causal_graph import CausalGraph
        from dowhy.causal_model import CausalModel

        if graph is None:
            graph = nx.DiGraph()
        elif isinstance(graph, CausalModel):
            graph = graph_copier(graph._graph._graph)
        elif isinstance(graph, CausalGraph):
            graph = graph_copier(graph._graph)

        if remove_existing_mechanisms:
            for node in graph.nodes:
                if CAUSAL_MECHANISM in graph.nodes[node]:
                    del graph.nodes[node][CAUSAL_MECHANISM]

        self.graph = graph
        self.graph_copier = graph_copier

    def set_causal_mechanism(self, node: Any, mechanism: Union[StochasticModel, ConditionalStochasticModel]) -> None:
        """Assigns the generative causal model of node in the causal graph.

        :param node: Target node whose causal model is to be assigned.
        :param mechanism: Causal mechanism to be assigned. A root node must be a
                          :class:`~dowhy.gcm.graph.StochasticModel`, whereas a non-root node must be a
                          :class:`~dowhy.gcm.graph.ConditionalStochasticModel`.
        """
        if node not in self.graph.nodes:
            raise ValueError("Node %s can not be found in the given graph!" % node)
        self.graph.nodes[node][CAUSAL_MECHANISM] = mechanism

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, ConditionalStochasticModel]:
        """Returns the generative causal model of node in the causal graph.

        :param node: Target node whose causal model is to be assigned.
        :returns: The causal mechanism for this node. A root node is of type
                  :class:`~dowhy.gcm.graph.StochasticModel`, whereas a non-root node is of type
                  :class:`~dowhy.gcm.graph.ConditionalStochasticModel`.
        """
        return self.graph.nodes[node][CAUSAL_MECHANISM]

    def clone(self):
        """Clones the causal model, but keeps causal mechanisms untrained."""
        graph_copy = self.graph_copier(self.graph)
        clone_causal_models(self.graph, graph_copy)
        return self.__class__(graph_copy)


class StructuralCausalModel(ProbabilisticCausalModel):
    """Represents a structural causal model (SCM), as required e.g. by
    :func:`~dowhy.gcm.whatif.counterfactual_samples`. As compared to a :class:`~dowhy.gcm.cms.ProbabilisticCausalModel`,
    an SCM describes the data generation process in non-root nodes by functional causal models.
    """

    def set_causal_mechanism(self, node: Any, mechanism: Union[StochasticModel, FunctionalCausalModel]) -> None:
        super().set_causal_mechanism(node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, FunctionalCausalModel]:
        return super().causal_mechanism(node)


class InvertibleStructuralCausalModel(StructuralCausalModel):
    """Represents an invertible structural graphical causal model, as required e.g. by
    :func:`~dowhy.gcm.whatif.counterfactual_samples`. This is a subclass of
    :class:`~dowhy.gcm.cms.StructuralCausalModel` and has further restrictions on the class of causal mechanisms.
    Here, the mechanisms of non-root nodes need to be invertible with respect to the noise,
    such as :class:`~dowhy.gcm.causal_mechanisms.PostNonlinearModel`.
    """

    def set_causal_mechanism(
        self, target_node: Any, mechanism: Union[StochasticModel, InvertibleFunctionalCausalModel]
    ) -> None:
        super().set_causal_mechanism(target_node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, InvertibleFunctionalCausalModel]:
        return super().causal_mechanism(node)


def validate_causal_model_assignment(causal_graph: DirectedGraph, target_node: Any) -> None:
    validate_node_has_causal_model(causal_graph, target_node)

    causal_model = causal_graph.nodes[target_node][CAUSAL_MECHANISM]

    if is_root_node(causal_graph, target_node):
        if not isinstance(causal_model, StochasticModel):
            raise RuntimeError(
                "Node %s is a root node and, thus, requires a StochasticModel, "
                "but a %s was found!" % (target_node, causal_model)
            )
    elif not isinstance(causal_model, ConditionalStochasticModel):
        raise RuntimeError(
            "Node %s has parents and, thus, requires a ConditionalStochasticModel, "
            "but a %s was found!" % (target_node, causal_model)
        )


def validate_node_has_causal_model(causal_graph: HasNodes, node: Any) -> None:
    validate_node_in_graph(causal_graph, node)

    if CAUSAL_MECHANISM not in causal_graph.nodes[node]:
        raise ValueError("Node %s has no assigned causal mechanism!" % node)


def validate_causal_dag(causal_graph: DirectedGraph) -> None:
    validate_acyclic(causal_graph)
    validate_causal_graph(causal_graph)


def validate_causal_graph(causal_graph: DirectedGraph) -> None:
    for node in causal_graph.nodes:
        validate_node(causal_graph, node)


def validate_node(causal_graph: DirectedGraph, node: Any) -> None:
    validate_causal_model_assignment(causal_graph, node)
    validate_local_structure(causal_graph, node)


def validate_local_structure(causal_graph: DirectedGraph, node: Any) -> None:
    if PARENTS_DURING_FIT not in causal_graph.nodes[node] or causal_graph.nodes[node][
        PARENTS_DURING_FIT
    ] != get_ordered_predecessors(causal_graph, node):
        raise RuntimeError(
            "The causal mechanism of node %s is not fitted to the graphical structure! Fit all "
            "causal models in the graph first. If the mechanism is already fitted based on the causal "
            "parents, consider to update the persisted parents for that node manually." % node
        )


def clone_causal_models(source: HasNodes, destination: HasNodes):
    for node in destination.nodes:
        if CAUSAL_MECHANISM in source.nodes[node]:
            destination.nodes[node][CAUSAL_MECHANISM] = source.nodes[node][CAUSAL_MECHANISM].clone()
