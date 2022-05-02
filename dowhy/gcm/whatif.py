from typing import Any, Dict, Callable, Iterable, List, Union, Optional

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm.cms import ProbabilisticCausalModel
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.gcm.graph import get_ordered_predecessors, is_root_node, DirectedGraph, validate_causal_dag, \
    validate_node_in_graph
from dowhy.gcm.util.general import convert_numpy_array_to_pandas_column


def interventional_samples(causal_model: ProbabilisticCausalModel,
                           interventions: Dict[Any, Callable[[np.ndarray], Union[float, np.ndarray]]],
                           observed_data: Optional[pd.DataFrame] = None,
                           num_samples_to_draw: Optional[int] = None) -> pd.DataFrame:
    """Performs intervention on nodes in the causal graph.

    :param causal_model: The probabilistic causal model we perform this intervention on .
    :param interventions: Dictionary containing the interventions we want to perform, keyed by node name. An
                          intervention is a function that takes a value as input and returns another value.
                          For example, `{'X': lambda x: 2}` mimics the atomic intervention *do(X:=2)*.
                          A soft intervention can be formulated as `{'X': lambda x: 0.2 * x}`.
    :param observed_data: Optionally, data on which to perform interventions. If None are given, data is generated based
                          on the generative models.
    :param num_samples_to_draw: Sample size to draw from the interventional distribution.
    :return: Samples from the interventional distribution.
    """
    validate_causal_dag(causal_model.graph)
    for node in interventions:
        validate_node_in_graph(causal_model.graph, node)

    if observed_data is None and num_samples_to_draw is None:
        raise ValueError("Either observed_samples or num_samples_to_draw need to be set!")
    if observed_data is not None and num_samples_to_draw is not None:
        raise ValueError("Either observed_samples or num_samples_to_draw need to be set, not both!")

    if num_samples_to_draw is not None:
        observed_data = draw_samples(causal_model, num_samples_to_draw)

    return _interventional_samples(causal_model, observed_data, interventions)


def _interventional_samples(causal_model: ProbabilisticCausalModel,
                            observed_data: pd.DataFrame,
                            interventions: Dict[Any, Callable[[np.ndarray], np.ndarray]]) -> pd.DataFrame:
    interventional_samples = observed_data.copy()
    affected_nodes = _get_nodes_affected_by_intervention(causal_model.graph, interventions.keys())

    for node in nx.topological_sort(causal_model.graph):
        if node not in affected_nodes:
            continue

        if is_root_node(causal_model.graph, node):
            node_data = interventional_samples[node]
        else:
            node_data = convert_numpy_array_to_pandas_column(
                causal_model.causal_mechanism(node).draw_samples(
                    interventional_samples[get_ordered_predecessors(causal_model.graph, node)].to_numpy()))

        interventional_samples[node] = _evaluate_intervention(node, interventions, node_data)

    return interventional_samples


def _get_nodes_affected_by_intervention(causal_graph: DirectedGraph, target_nodes: Iterable[Any]) -> List[Any]:
    result = []

    for node in nx.topological_sort(causal_graph):
        if node in target_nodes:
            result.append(node)
            continue

        for target_node in target_nodes:
            if target_node in nx.ancestors(causal_graph, source=node):
                result.append(node)
                break

    return result


def _evaluate_intervention(node: Any,
                           interventions: Dict[Any, Callable[[np.ndarray], np.ndarray]],
                           node_data: np.ndarray) -> np.ndarray:
    if node in interventions:
        post_intervention_data = np.array(list(map(interventions[node], node_data)))
        if node_data.shape != post_intervention_data.shape:
            raise RuntimeError(
                'Dimension of data corresponding to the node `%s` after intervention is different than before '
                'intervention.' % node)

        return post_intervention_data
    else:
        return node_data
