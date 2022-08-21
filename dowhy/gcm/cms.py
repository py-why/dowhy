"""This module defines the fundamental classes for graphical causal models (GCMs).

Classes in this module should be considered experimental, meaning there might be breaking API changes in the future.
"""

from typing import Any, Callable, Optional, Union

import networkx as nx

from dowhy.gcm.graph import (
    CAUSAL_MECHANISM,
    ConditionalStochasticModel,
    DirectedGraph,
    FunctionalCausalModel,
    InvertibleFunctionalCausalModel,
    StochasticModel,
    clone_causal_models,
)


class ProbabilisticCausalModel:
    """Represents a probabilistic graphical causal model, i.e. it combines a graphical representation of causal
    causal relationships and corresponding causal mechanism for each node describing the data generation process. The
    causal mechanisms can be any general stochastic models."""

    def __init__(
        self, graph: Optional[DirectedGraph] = None, graph_copier: Callable[[DirectedGraph], DirectedGraph] = nx.DiGraph
    ):
        """
        :param graph: Optional graph object to be used as causal graph.
        :param graph_copier: Optional function that can copy a causal graph. Defaults to a networkx.DiGraph
                             constructor.
        """
        if graph is None:
            graph = nx.DiGraph()
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
    such as :class:`~dowhy.gcm.fcms.PostNonlinearModel`.
    """

    def set_causal_mechanism(
        self, target_node: Any, mechanism: Union[StochasticModel, InvertibleFunctionalCausalModel]
    ) -> None:
        super().set_causal_mechanism(target_node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, InvertibleFunctionalCausalModel]:
        return super().causal_mechanism(node)
