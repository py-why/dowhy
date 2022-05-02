from typing import Optional, Any, Union

from dowhy.gcm.graph import DirectedGraph, StochasticModel, ConditionalStochasticModel, FunctionalCausalModel, \
    CAUSAL_MECHANISM, InvertibleFunctionalCausalModel


class ProbabilisticCausalModel:
    def __init__(self, graph: Optional[DirectedGraph] = None):
        if graph is None:
            import networkx as nx
            graph = nx.DiGraph()
        self.graph = graph

    def set_causal_mechanism(self, node: Any, mechanism: Union[StochasticModel, ConditionalStochasticModel]) -> None:
        """Assigns the generative causal model of node in the causal graph.

        :param node: Target node whose causal model is to be assigned.
        :param mechanism: Causal mechanism to be assigned. A root node must be a :py:mod:`StochasticModel
                          <dowhy.scm.graph.StochasticModel>`, whereas a non-root node must be a
                          :py:mod:`ConditionalStochasticModel <dowhy.scm.graph.ConditionalStochasticModel>`.
        """
        if node not in self.graph.nodes:
            raise ValueError("Node %s can not be found in the given graph!" % node)
        self.graph.nodes[node][CAUSAL_MECHANISM] = mechanism

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, ConditionalStochasticModel]:
        return self.graph.nodes[node][CAUSAL_MECHANISM]


class StructuralCausalModel(ProbabilisticCausalModel):
    def set_causal_mechanism(self, node: Any, mechanism: Union[StochasticModel, FunctionalCausalModel]) -> None:
        super().set_causal_mechanism(node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, FunctionalCausalModel]:
        return super().causal_mechanism(node)


class InvertibleStructuralCausalModel(StructuralCausalModel):
    def set_causal_mechanism(self, target_node: Any,
                             mechanism: Union[StochasticModel, InvertibleFunctionalCausalModel]) -> None:
        super().set_causal_mechanism(target_node, mechanism)

    def causal_mechanism(self, node: Any) -> Union[StochasticModel, InvertibleFunctionalCausalModel]:
        return super().causal_mechanism(node)
