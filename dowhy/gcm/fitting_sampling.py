"""This module provides functionality for fitting probabilistic causal models and drawing samples from them."""

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from dowhy.gcm import config
from dowhy.gcm.causal_models import (
    PARENTS_DURING_FIT,
    ProbabilisticCausalModel,
    validate_causal_dag,
    validate_causal_model_assignment,
)
from dowhy.graph import get_ordered_predecessors, is_root_node


def fit(causal_model: ProbabilisticCausalModel, data: pd.DataFrame, return_evaluation_summary: bool = False):
    """Fits the causal mechanism of each node to the data. This is done by iterating over the nodes in the graph and
    fitting their assigned causal mechanisms individually to the data by calling the corresponding fit function. Due to
    the modularity assumption, we can fit each mechanism in the graph independently of the other mechanisms. For root
    nodes, the training data is the corresponding column in the provided data. For non-root nodes, the data is based on
    a node's parents and the node itself. Before a node is fitted, this function first validates whether the assigned
    mechanism is valid, i.e., whether a root node follows a StochasticModel and whether a non-root node follows a
    ConditionalStochasticModel.

    The details of fitting a causal mechanism depend on their implementation. For example, if a node follows an additive
    noise model X_i = f_i(PA_i) + N_i, where N_i is unobserved noise, the fitting involves fitting the function f_i
    (which could be any scikit-learn regressor) to the data and modeling the distribution N_i based on the residuals
    X_i - f_i(PA_i). For more details on how each individual mechanism is fitted, refer to the corresponding
    documentation, since these are individual implementation details.

    This function optionally, returns a summary of different metrics of the causal mechanisms evaluated via
    cross-validation. Note, this will use the evaluate_causal_model method. For more detailed and extensive evaluations,
    consider using the evaluate_causal_model method directly.

    :param causal_model: The causal model containing the mechanisms of the node that will be fitted.
    :param data: Observations of nodes in the causal model.
    :param return_evaluation_summary: If True, returns a summary of the performances of the fitted mechanisms using the
                                      evaluate_causal_model method. If False, nothing is returned.
    :return: Optionally, a CausalModelEvaluationResult summarizing the performances of the causal mechanisms via
             cross-validation.
    """
    progress_bar = tqdm(
        causal_model.graph.nodes,
        desc="Fitting causal models",
        position=0,
        leave=True,
        disable=not config.show_progress_bars,
    )
    for node in progress_bar:
        if node not in data:
            raise RuntimeError(
                "Could not find data for node %s in the given training data! There should be a column "
                "containing samples for node %s." % (node, node)
            )

        progress_bar.set_description("Fitting causal mechanism of node %s" % node)

        fit_causal_model_of_target(causal_model, node, data)

    if return_evaluation_summary:
        from dowhy.gcm import evaluate_causal_model

        return evaluate_causal_model(
            causal_model,
            data,
            evaluate_causal_mechanisms=True,
            compare_mechanism_baselines=False,
            evaluate_invertibility_assumptions=False,
            evaluate_overall_kl_divergence=False,
            evaluate_causal_structure=False,
        )


def fit_causal_model_of_target(
    causal_model: ProbabilisticCausalModel, target_node: Any, training_data: pd.DataFrame
) -> None:
    """Fits only the causal mechanism of the given target node based on the training data.

    :param causal_model: The causal model containing the target node.
    :param target_node: Target node for which the mechanism is fitted.
    :param training_data: Training data for fitting the causal mechanism.
    :return: None
    """
    validate_causal_model_assignment(causal_model.graph, target_node)

    y_nan_mask = pd.isna(training_data[target_node].to_numpy())

    if is_root_node(causal_model.graph, target_node):
        causal_model.causal_mechanism(target_node).fit(X=training_data[target_node].to_numpy()[~y_nan_mask])
    else:
        causal_model.causal_mechanism(target_node).fit(
            X=training_data[get_ordered_predecessors(causal_model.graph, target_node)].to_numpy()[~y_nan_mask],
            Y=training_data[target_node].to_numpy()[~y_nan_mask],
        )

    # To be able to validate that the graph structure did not change between fitting and causal query, we store the
    # parents of a node during fit. That way, before sampling, we can verify the parents are still the same. While
    # this would automatically fail when the number of parents is different, there are other more subtle cases,
    # where the number is still the same, but it's different parents, and therefore different data. That would yield
    # wrong results, but would not fail.
    causal_model.graph.nodes[target_node][PARENTS_DURING_FIT] = get_ordered_predecessors(
        causal_model.graph, target_node
    )


def draw_samples(causal_model: ProbabilisticCausalModel, num_samples: int) -> pd.DataFrame:
    """Draws new joint samples from the given graphical causal model. This is done by first generating random samples
    from root nodes and then propagating causal downstream effects through the graph.

    :param causal_model: New samples are generated based on the given causal model.
    :param num_samples: Number of samples to draw.
    :return: A pandas data frame where columns correspond to the nodes in the graph and rows to the drawn joint samples.
    """
    validate_causal_dag(causal_model.graph)

    sorted_nodes = list(nx.topological_sort(causal_model.graph))
    drawn_samples = pd.DataFrame(np.empty((num_samples, len(sorted_nodes))), columns=sorted_nodes)

    for node in sorted_nodes:
        causal_mechanism = causal_model.causal_mechanism(node)

        if is_root_node(causal_model.graph, node):
            drawn_samples[node] = causal_mechanism.draw_samples(num_samples).squeeze()
        else:
            drawn_samples[node] = causal_mechanism.draw_samples(
                _parent_samples_of(node, causal_model, drawn_samples)
            ).squeeze()

    return drawn_samples


def _parent_samples_of(node: Any, scm: ProbabilisticCausalModel, samples: pd.DataFrame) -> np.ndarray:
    return samples[get_ordered_predecessors(scm.graph, node)].to_numpy()
