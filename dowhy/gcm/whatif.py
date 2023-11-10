"""This module provides functionality to answer what-if questions."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm._noise import compute_noise_from_data
from dowhy.gcm.causal_mechanisms import ClassifierFCM
from dowhy.gcm.causal_models import (
    InvertibleStructuralCausalModel,
    ProbabilisticCausalModel,
    StructuralCausalModel,
    validate_causal_dag,
)
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.graph import (
    DirectedGraph,
    get_ordered_predecessors,
    is_root_node,
    node_connected_subgraph_view,
    validate_node_in_graph,
)


def interventional_samples(
    causal_model: ProbabilisticCausalModel,
    interventions: Dict[Any, Callable[[np.ndarray], Union[float, np.ndarray]]],
    observed_data: Optional[pd.DataFrame] = None,
    num_samples_to_draw: Optional[int] = None,
) -> pd.DataFrame:
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


def _interventional_samples(
    pcm: ProbabilisticCausalModel,
    observed_data: pd.DataFrame,
    interventions: Dict[Any, Callable[[np.ndarray], np.ndarray]],
) -> pd.DataFrame:
    samples = observed_data.copy()

    affected_nodes = _get_nodes_affected_by_intervention(pcm.graph, interventions.keys())
    sorted_nodes = nx.topological_sort(pcm.graph)

    # Simulating interventions by propagating the effects through the graph. For this, we iterate over the nodes based
    # on their topological order.
    for node in sorted_nodes:
        if node not in affected_nodes:
            continue

        if is_root_node(pcm.graph, node):
            node_data = samples[node].to_numpy()
        else:
            node_data = pcm.causal_mechanism(node).draw_samples(_parent_samples_of(node, pcm, samples))

        # After drawing samples of the node based on the data generation process, we apply the corresponding
        # intervention. The inputs of downstream nodes are therefore based on the outcome of the intervention in this
        # node.
        samples[node] = _evaluate_intervention(node, interventions, node_data.reshape(-1))

    return samples


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


def counterfactual_samples(
    causal_model: Union[StructuralCausalModel, InvertibleStructuralCausalModel],
    interventions: Dict[Any, Callable[[np.ndarray], Union[float, np.ndarray]]],
    observed_data: Optional[pd.DataFrame] = None,
    noise_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimates counterfactual data for observed data if we were to perform specified interventions. This function
    implements the 3-step process for computing counterfactuals by Pearl (see https://ftp.cs.ucla.edu/pub/stat_ser/r485.pdf).

    :param causal_model: The (invertible) structural causal model we perform this intervention on. If noise_data is
                         None and observed_data is provided, this must be an invertible structural model, otherwise,
                         this can be either a structural causal model or an invertible one.
    :param interventions: Dictionary containing the interventions we want to perform keyed by node name. An
                          intervention is a function that takes a value as input and returns another value.
                          For example, `{'X': lambda x: 2}` mimics the atomic intervention *do(X:=2)*.
    :param observed_data: Factual data that we observe for the nodes in the causal graph.
    :param noise_data: Data of noise terms corresponding to nodes in the causal graph. If not provided,
                       these have to be estimated from observed data. Then we require causal models of nodes to be
                       invertible.
    :return: Estimated counterfactual data.
    """
    for node in interventions:
        validate_node_in_graph(causal_model.graph, node)

    validate_causal_dag(causal_model.graph)

    if observed_data is None and noise_data is None:
        raise ValueError("Either observed_data or noise_data need to be given!")
    if observed_data is not None and noise_data is not None:
        raise ValueError("Either observed_data or noise_data can be given, not both!")

    if noise_data is None and observed_data is not None:
        if not isinstance(causal_model, InvertibleStructuralCausalModel):
            raise ValueError(
                "Since no noise_data is given, this has to be estimated from the given "
                "observed_data. This can only be done with InvertibleStructuralCausalModel."
            )
        # Abduction: For invertible SCMs, we recover exact noise values from data.
        noise_data = compute_noise_from_data(causal_model, observed_data)

    # Action + Prediction: Propagate the intervention downstream using recovered noise values.
    return _counterfactual_samples(causal_model, interventions, noise_data)


def _counterfactual_samples(
    scm: StructuralCausalModel,
    interventions: Dict[Any, Callable[[np.ndarray], Union[float, np.ndarray]]],
    noise_data: pd.DataFrame,
) -> pd.DataFrame:
    topologically_sorted_nodes = list(nx.topological_sort(scm.graph))
    samples = pd.DataFrame(
        np.empty((noise_data.shape[0], len(topologically_sorted_nodes))), columns=topologically_sorted_nodes
    )

    for node in topologically_sorted_nodes:
        if is_root_node(scm.graph, node):
            node_data = noise_data[node].to_numpy()
        else:
            node_data = scm.causal_mechanism(node).evaluate(
                _parent_samples_of(node, scm, samples), noise_data[node].to_numpy()
            )

        samples[node] = _evaluate_intervention(node, interventions, node_data.reshape(-1))

    return samples


def _evaluate_intervention(
    node: Any, interventions: Dict[Any, Callable[[np.ndarray], np.ndarray]], pre_intervention_data: np.ndarray
) -> np.ndarray:
    # Check if we need to apply an intervention on the given node.
    if node in interventions:
        # Apply intervention function to the data of the node.
        post_intervention_data = np.array(list(map(interventions[node], pre_intervention_data)))

        # Check if the intervention function changes the shape of the data.
        if pre_intervention_data.shape != post_intervention_data.shape:
            raise RuntimeError(
                "Dimension of data corresponding to the node `%s` after intervention is different than before "
                "intervention." % node
            )

        return post_intervention_data
    else:
        return pre_intervention_data


def average_causal_effect(
    causal_model: ProbabilisticCausalModel,
    target_node: Any,
    interventions_alternative: Dict[Any, Callable[[np.ndarray], Union[float, np.ndarray]]],
    interventions_reference: Dict[Any, Callable[[np.ndarray], Union[float, np.ndarray]]],
    observed_data: Optional[pd.DataFrame] = None,
    num_samples_to_draw: Optional[int] = None,
) -> float:
    """Estimates the average causal effect (ACE) on the target of two different sets of interventions.
    The interventions can be specified through the parameters `interventions_alternative` and `interventions_reference`.
    For example, if the alternative intervention is do(T := 1) and the reference intervention
    is do(T := 0), then the average causal effect is given by ACE = E[Y | do(T := 1)] - E[Y | do(T := 0)]:
        >>> average_causal_effect(causal_model, 'Y', {'T': lambda _ : 1}, {'T': lambda _ : 0})

    We can also specify more complex interventions on multiple nodes:
        >>> average_causal_effect(causal_model,
        >>>                       'Y',
        >>>                       {'T': lambda _ : 1, 'X0': lambda x : x + 1},
        >>>                       {'T': lambda _ : 0, 'X0': lambda x : x * 2})
    In the above, we would estimate ACE = E[Y | do(T := 1), do(X0 := X0 + 1)] - E[Y | do(T := 0), do(X0 := X0 * 2)].

    Note: The target node can be a continuous real-valued variable or a categorical variable with at most two classes
    (i.e. binary).

    :param causal_model: The probabilistic causal model we perform this intervention on .
    :param target_node: Target node for which the ACE is estimated.
    :param interventions_alternative: Dictionary defining the interventions for the alternative values.
    :param interventions_reference: Dictionary defining the interventions for the reference values.
    :param observed_data: Factual data that we observe for the nodes in the causal graph. By default, new data
                          is sampled using the causal model. If observational data is available, providing them
                          might improve the accuracy by mitigating issues due to a misspecified graph and/or causal
                          models.
    :param num_samples_to_draw: Number of samples drawn from the causal model for estimating ACE if no observed data is
                                given.
    :return: The estimated average causal effect (ACE).
    """
    # For estimating the effect, we only need to consider the nodes that have a directed path to the target node, i.e.
    # all ancestors of the target.
    causal_model = ProbabilisticCausalModel(node_connected_subgraph_view(causal_model.graph, target_node))

    validate_causal_dag(causal_model.graph)
    for node in interventions_alternative:
        validate_node_in_graph(causal_model.graph, node)
    for node in interventions_reference:
        validate_node_in_graph(causal_model.graph, node)

    if observed_data is None and num_samples_to_draw is None:
        raise ValueError("Either observed_samples or num_samples_to_draw need to be set!")
    if observed_data is not None and num_samples_to_draw is not None:
        raise ValueError("Either observed_samples or num_samples_to_draw need to be set, not both!")

    if num_samples_to_draw is not None:
        observed_data = draw_samples(causal_model, num_samples_to_draw)

    samples_from_target_alt = _interventional_samples(causal_model, observed_data, interventions_alternative)[
        target_node
    ].to_numpy()
    samples_from_target_ref = _interventional_samples(causal_model, observed_data, interventions_reference)[
        target_node
    ].to_numpy()

    target_causal_model = causal_model.causal_mechanism(target_node)
    if isinstance(target_causal_model, ClassifierFCM):
        # The target node can be a continuous real-valued variable or a categorical variable with at most two classes
        # (i.e. binary).
        if observed_data[target_node].nunique() > 2:
            raise ValueError(
                "Cannot estimate average treatment effect of categorical data with more than 2 categories!"
            )

        class_names = target_causal_model.get_class_names(np.array([0, 1]))
        samples_from_target_alt[samples_from_target_alt == class_names[0]] = 0
        samples_from_target_alt[samples_from_target_alt == class_names[1]] = 1
        samples_from_target_ref[samples_from_target_ref == class_names[0]] = 0
        samples_from_target_ref[samples_from_target_ref == class_names[1]] = 1

    return np.mean(samples_from_target_alt) - np.mean(samples_from_target_ref)


def _parent_samples_of(node: Any, scm: ProbabilisticCausalModel, samples: pd.DataFrame) -> np.ndarray:
    return samples[get_ordered_predecessors(scm.graph, node)].to_numpy()
