"""This module defines the fundamental classes for graphical causal models (GCMs)."""

from typing import Optional, Any, Union

from dowhy.gcm.graph import DirectedGraph, StochasticModel, ConditionalStochasticModel, FunctionalCausalModel, \
    CAUSAL_MECHANISM, InvertibleFunctionalCausalModel


class ProbabilisticCausalModel:
    """Represents a probabilistic graphical causal model, i.e. it combines a graphical representation of causal
    causal relationships and corresponding causal mechanism for each node describing the data generation process. The
    causal mechanisms can be any general stochastic models."""

    def __init__(self, graph: Optional[DirectedGraph] = None):
        if graph is None:
            import networkx as nx
            graph = nx.DiGraph()
        self.graph = graph

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


class StructuralCausalModel(ProbabilisticCausalModel):
    """Represents a structural causal model (SCM), as required e.g. by
    :func:`~dowhy.gcm.counterfactual_samples`. As compared to a :class:`~dowhy.gcm.ProbabilisticCausalModel`,
    an SCM describes the data generation process in non-root nodes by functional causal models.
    """

    def set_causal_mechanism(self, node: Any, mechanism: Union[StochasticModel, FunctionalCausalModel]) -> None:
        super().set_causal_mechanism(node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, FunctionalCausalModel]:
        return super().causal_mechanism(node)


class InvertibleStructuralCausalModel(StructuralCausalModel):
    """Represents an invertible structural graphical causal model, as required e.g. by
    :func:`~dowhy.gcm.counterfactual_samples`. This is a subclass of :class:`~dowhy.gcm.StructuralCausalModel` and
    has further restrictions on the class of causal mechanisms. Here, the mechanisms of non-root nodes need to be
    invertible with respect to the noise, such as :class:`~dowhy.gcm.PostNonlinearModel`.
    """

    def set_causal_mechanism(self, target_node: Any,
                             mechanism: Union[StochasticModel, InvertibleFunctionalCausalModel]) -> None:
        super().set_causal_mechanism(target_node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, InvertibleFunctionalCausalModel]:
        return super().causal_mechanism(node)
