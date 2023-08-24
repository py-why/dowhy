from typing import Any, Callable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm.causal_models import (
    InvertibleStructuralCausalModel,
    ProbabilisticCausalModel,
    StructuralCausalModel,
    validate_causal_dag,
)
from dowhy.gcm.ml.prediction_model import PredictionModel
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node, node_connected_subgraph_view


def compute_data_from_noise(causal_model: StructuralCausalModel, noise_data: pd.DataFrame) -> pd.DataFrame:
    validate_causal_dag(causal_model.graph)

    sorted_nodes = list(nx.topological_sort(causal_model.graph))
    data = pd.DataFrame(np.empty((noise_data.shape[0], len(sorted_nodes))), columns=sorted_nodes)

    for node in sorted_nodes:
        if is_root_node(causal_model.graph, node):
            data[node] = noise_data[node].to_numpy().squeeze()
        else:
            data[node] = (
                causal_model.causal_mechanism(node)
                .evaluate(
                    data[get_ordered_predecessors(causal_model.graph, node)].to_numpy(), noise_data[node].to_numpy()
                )
                .squeeze()
            )

    return data


def compute_noise_from_data(causal_model: InvertibleStructuralCausalModel, observed_data: pd.DataFrame) -> pd.DataFrame:
    validate_causal_dag(causal_model.graph)

    sorted_noise = list(nx.topological_sort(causal_model.graph))
    noise = pd.DataFrame(np.empty((observed_data.shape[0], len(sorted_noise))), columns=sorted_noise)

    for node in sorted_noise:
        if is_root_node(causal_model.graph, node):
            noise[node] = observed_data[node].to_numpy().squeeze()
        else:
            noise[node] = (
                causal_model.causal_mechanism(node)
                .estimate_noise(observed_data[node].to_numpy(), _parent_samples_of(node, causal_model, observed_data))
                .squeeze()
            )

    return noise


def get_noise_dependent_function(
    causal_model: StructuralCausalModel,
    target_node: Any,
    approx_prediction_model: Optional[PredictionModel] = None,
    num_training_samples: int = 20000,
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[Any]]:
    """Returns a function that represents the given target_node and that is only dependent on upstream noise nodes.
    This is, Y = f(N_0, N_1, ..., N_n), where Y is the target node and N_i the noise node of an upstream node. Since
    the order of the noise variables can be ambiguous, this method also returns a list with the expected order of the
    noise variables indicated by the name of the corresponding node. For instance:
    Lets say we have a target variable X4 which has only 2 other upstream variables X1 and X3, then this methods returns
    a callable that represents X4 = f(N_1, N_3, N_4). This callable expects a numpy array as input. Since the order
    of the columns in this array is unclear, the method also returns a list with the names of the corresponding columns,
    e.g. ['X1', 'X3', 'X4']. Note that the noise of X4 will also be an input variable here.

    If an approx_prediction_model is given, the model is used instead to train a model from scratch to represent f.
    Typically, it would be much faster to evaluate this model than propagating through the graph. The model is trained
    based on generated noise and target samples from the given causal graph. In theory, the approximated version of f
    should be close to the "true" one based on the underlying SCMs, but there can still be (significant) differences,
    especially if the provided model is inappropriate for representing f.

    Note: All nodes in the graph that have no direct path to the target node are omitted. The noise node of the
          target variable itself will also be included here.

    :param causal_model: A structural causal model.
    :param target_node: The target node for which the function f should be returned.
    :param approx_prediction_model: Prediction model for approximating f. The model is trained based on drawn noise and
                                    target samples.
    :param num_training_samples: Number of drawn samples for training the predictor based on the provided
                                 approx_prediction_model.
                                 Note: This parameter is ignored if approx_prediction_model is None.
    :return: A tuple, where the first value is a callable (the function f) that expects a numpy array X as input and
    the second value is a list with nodes that represents the expected order of the columns in X.
    """
    validate_causal_dag(causal_model.graph)

    if approx_prediction_model is not None:
        return _get_approx_noise_dependent_function(
            StructuralCausalModel(node_connected_subgraph_view(causal_model.graph, target_node)),
            target_node,
            approx_prediction_model,
            num_training_samples,
        )
    else:
        return _get_exact_noise_dependent_function(
            StructuralCausalModel(node_connected_subgraph_view(causal_model.graph, target_node)), target_node
        )


def _get_exact_noise_dependent_function(
    causal_model: StructuralCausalModel, target_node: Any
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[Any]]:
    nodes_order = list(nx.topological_sort(causal_model.graph))

    def predict_method(noise_samples: np.ndarray) -> np.ndarray:
        return compute_data_from_noise(causal_model, pd.DataFrame(noise_samples, columns=[x for x in nodes_order]))[
            target_node
        ].to_numpy()

    return predict_method, nodes_order


def _get_approx_noise_dependent_function(
    causal_model: StructuralCausalModel,
    target_node: Any,
    approx_prediction_model: PredictionModel,
    num_training_samples: int,
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[Any]]:
    nodes_order = list(nx.topological_sort(causal_model.graph))

    node_samples, noise_samples = noise_samples_of_ancestors(causal_model, target_node, num_training_samples)

    approx_prediction_model.fit(
        shape_into_2d(noise_samples[nodes_order].to_numpy()), shape_into_2d(node_samples[target_node].to_numpy())
    )

    return approx_prediction_model.predict, nodes_order


def noise_samples_of_ancestors(
    causal_model: StructuralCausalModel, target_node: Any, num_samples: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_nodes = list(nx.topological_sort(causal_model.graph))
    all_ancestors_of_node = nx.ancestors(causal_model.graph, target_node)
    all_ancestors_of_node.update({target_node})

    drawn_samples = pd.DataFrame(np.empty((num_samples, len(sorted_nodes))), columns=sorted_nodes)
    drawn_noise_samples = pd.DataFrame(np.empty((num_samples, len(sorted_nodes))), columns=sorted_nodes)

    for node in sorted_nodes:
        if node not in all_ancestors_of_node:
            continue

        if is_root_node(causal_model.graph, node):
            noise = causal_model.causal_mechanism(node).draw_samples(num_samples).reshape(-1)
            drawn_noise_samples[node] = noise.squeeze()
            drawn_samples[node] = noise.squeeze()
        else:
            noise = causal_model.causal_mechanism(node).draw_noise_samples(num_samples).reshape(-1)
            drawn_noise_samples[node] = noise.squeeze()
            drawn_samples[node] = (
                causal_model.causal_mechanism(node)
                .evaluate(_parent_samples_of(node, causal_model, drawn_samples), noise)
                .squeeze()
            )

        if node == target_node:
            break

    return drawn_samples, drawn_noise_samples


def _parent_samples_of(node: Any, scm: ProbabilisticCausalModel, samples: pd.DataFrame) -> np.ndarray:
    return samples[get_ordered_predecessors(scm.graph, node)].to_numpy()
