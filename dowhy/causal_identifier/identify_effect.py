from typing import List, Protocol, Union

import networkx as nx

from dowhy.causal_identifier.auto_identifier import BackdoorAdjustment, EstimandType, identify_effect_auto
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand


class CausalIdentifier(Protocol):
    """
    Protocol to define a CausalIdentifier, all CausalIdentifiers must conform to at least this list of methods.

    This class is for backwards compatibility with CausalModel
    Will be deprecated in the future in favor of function call auto_identify_effect()

    """

    def identify_effect(
        self, graph: nx.DiGraph, action_nodes: Union[str, List[str]], outcome_nodes: Union[str, List[str]], **kwargs
    ):
        """Identify the causal effect to be estimated based on a causal graph
        :param graph: Causal graph to be analyzed
        :param action_nodes: name of the treatment
        :param outcome_nodes: name of the outcome
        :param **kwargs: Additional parameters required by the identify_effect of a specific CausalIdentifier
        for example: conditional_node_names in AutoIdentifier or node_names in IDIdentifier
        :returns: a probability expression (estimand) for the causal effect if identified, else NULL
        """
        ...


def identify_effect(
    graph: nx.DiGraph,
    action_nodes: Union[str, List[str]],
    outcome_nodes: Union[str, List[str]],
    observed_nodes: Union[str, List[str]],
) -> IdentifiedEstimand:
    """Identify the causal effect to be estimated based on a causal graph

    :param graph: Causal graph to be analyzed
    :param treatment: name of the treatment
    :param outcome: name of the outcome
    :returns: a probability expression (estimand) for the causal effect if identified, else NULL
    """
    return identify_effect_auto(
        graph,
        action_nodes,
        outcome_nodes,
        observed_nodes,
        EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=BackdoorAdjustment.BACKDOOR_DEFAULT,
        optimize_backdoor=False,
    )
