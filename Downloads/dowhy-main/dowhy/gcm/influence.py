"""This module provides functions to estimate causal influences."""

import logging
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import dowhy.gcm.auto as auto
from dowhy.gcm import feature_relevance_sample
from dowhy.gcm._noise import compute_data_from_noise, compute_noise_from_data, noise_samples_of_ancestors
from dowhy.gcm.causal_mechanisms import ClassifierFCM, ConditionalStochasticModel, ProbabilityEstimatorModel
from dowhy.gcm.causal_models import (
    InvertibleStructuralCausalModel,
    ProbabilisticCausalModel,
    StructuralCausalModel,
    validate_causal_dag,
    validate_node,
)
from dowhy.gcm.divergence import estimate_kl_divergence_of_probabilities
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.gcm.ml import ClassificationModel, PredictionModel
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values
from dowhy.gcm.stats import marginal_expectation
from dowhy.gcm.uncertainty import estimate_entropy_of_probabilities, estimate_variance
from dowhy.gcm.util.general import has_categorical, is_categorical, means_difference, set_random_seed, shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node, node_connected_subgraph_view

_logger = logging.getLogger(__name__)


def arrow_strength(
    causal_model: ProbabilisticCausalModel,
    target_node: Any,
    parent_samples: Optional[pd.DataFrame] = None,
    num_samples_conditional: int = 2000,
    max_num_runs: int = 5000,
    tolerance: float = 0.01,
    n_jobs: int = -1,
    difference_estimation_func: Optional[Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]] = None,
) -> Dict[Tuple[Any, Any], float]:
    """Computes the causal strength of each edge directed to the target node.
    The strength of an edge is quantified in terms of distance between conditional distributions of the target node in
    the original graph and the imputed graph wherein the edge has been removed and the target node is fed a random
    permutation of the observations of the source node. For more scientific details behind this API, please refer to
    the research paper below.

    **Research Paper**:
    Dominik Janzing, David Balduzzi, Moritz Grosse-Wentrup, Bernhard Schölkopf. *Quantifying Causal Influences*. The
    Annals of Statistics, Vol. 41, No. 5, 2324-2358, 2013.

    :param causal_model: The probabilistic causal model for whose target node we compute the strength of incoming
                         edges for.
    :param target_node: The target node whose incoming edges' strength is to be computed.
    :param parent_samples: Optional samples from the parents of the target_node. If None are given, they are generated
                           based on the provided causal model. Providing observational data can help to mitigate
                           misspecifications in the graph, such as missing interactions between root nodes or
                           confounders.
    :param num_samples_conditional: Sample size to use for estimating the distance between distributions. The more
                                    more samples, the higher the accuracy.
    :param max_num_runs: The maximum number of times to resample and estimate the strength to report the average
                         strength.
    :param tolerance: If the percentage change in the estimated strength between two consecutive runs falls below the
                      specified tolerance, the algorithm will terminate before reaching the maximum number of runs.
                      A value of 0.01 would indicate a change of less than 1%. However, in order to minimize the impact
                      of randomness, there must be at least three consecutive runs where the change is below the
                      threshold.
    :param n_jobs: The number of jobs to run in parallel. Set it to -1 to use all processors.
    :param difference_estimation_func: Optional: How to measure the distance between two distributions. By default,
                                       the difference of the variance is estimated for a continuous target node
                                       and the KL divergence for a categorical target node.
    :return: Causal strength of each edge.
    """
    if target_node not in causal_model.graph.nodes:
        raise ValueError("Target node %s can not be found in given graph!" % target_node)
    if is_root_node(causal_model.graph, target_node):
        raise ValueError("Target node %s is a root node, but it requires to have ancestors!" % target_node)

    # Creating a smaller subgraph, which only contains upstream nodes that are connected to the target node.
    sub_causal_model = ProbabilisticCausalModel(node_connected_subgraph_view(causal_model.graph, target_node))
    validate_node(sub_causal_model.graph, target_node)

    ordered_predecessors = get_ordered_predecessors(sub_causal_model.graph, target_node)

    if parent_samples is None:
        parent_samples = draw_samples(sub_causal_model, num_samples_conditional * 20)[ordered_predecessors]

    direct_influences = arrow_strength_of_model(
        sub_causal_model.causal_mechanism(target_node),
        parent_samples[ordered_predecessors].to_numpy(),
        num_samples_from_conditional=num_samples_conditional,
        max_num_runs=max_num_runs,
        tolerance=tolerance,
        n_jobs=n_jobs,
        difference_estimation_func=difference_estimation_func,
    )
    return {(predecessor, target_node): direct_influences[i] for i, predecessor in enumerate(ordered_predecessors)}


def arrow_strength_of_model(
    conditional_stochastic_model: ConditionalStochasticModel,
    input_samples: np.ndarray,
    num_samples_from_conditional: int = 2000,
    max_num_runs: int = 5000,
    tolerance: float = 0.01,
    n_jobs: int = -1,
    difference_estimation_func: Optional[Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]] = None,
    input_subsets: Optional[List[List[int]]] = None,
) -> np.ndarray:
    input_samples = shape_into_2d(input_samples)

    if input_subsets is None:
        input_subsets = [[i] for i in range(input_samples.shape[1])]

    if difference_estimation_func is None:
        if isinstance(conditional_stochastic_model, ProbabilityEstimatorModel):
            difference_estimation_func = estimate_kl_divergence_of_probabilities
        else:

            def difference_estimation_func(old, new):
                return np.var(new) - np.var(old)

    if isinstance(conditional_stochastic_model, ProbabilityEstimatorModel):
        samples_creation_method = conditional_stochastic_model.estimate_probabilities
    else:
        samples_creation_method = conditional_stochastic_model.draw_samples

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        def parallel_job(subset: List[int], parallel_random_seed: int):
            set_random_seed(parallel_random_seed)

            return _estimate_direct_strength(
                samples_creation_method,
                input_samples,
                subset,
                difference_estimation_func,
                num_samples_from_conditional,
                max_num_runs,
                tolerance,
            )

        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=len(input_subsets))
        results = Parallel(n_jobs=n_jobs)(
            delayed(parallel_job)(subset, int(random_seed)) for subset, random_seed in zip(input_subsets, random_seeds)
        )

    if np.any(results == np.inf):
        _logger.warning(
            "At least one arrow strength is infinite. This typically happens if the causal models are "
            "deterministic, i.e. there is no noise or it is extremely small."
        )

    return np.array(results)


def _estimate_direct_strength(
    draw_samples_func: Callable[[np.ndarray], np.ndarray],
    distribution_samples: np.ndarray,
    parents_subset: List[int],
    difference_estimation_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
    num_samples_conditional: int,
    max_num_runs: int,
    tolerance: float,
) -> float:
    distribution_samples = shape_into_2d(distribution_samples)

    num_samples_conditional = min(num_samples_conditional, distribution_samples.shape[0])

    aggregated_conditional_difference_result = 0
    average_difference_result = 0
    converged_run = 0
    for run, sample in enumerate(distribution_samples):
        tmp_samples = np.tile(sample, (num_samples_conditional, 1))

        rnd_permutation = np.random.choice(distribution_samples.shape[0], num_samples_conditional, replace=False)

        # Sampling from the conditional distribution based on the current sample.
        conditional_distribution_samples = draw_samples_func(tmp_samples)

        # Sampling from the conditional based on the current sample, but randomizing the inputs of all variables that
        # are in the given subset. By this, we can simulate the impact on the conditional distribution when removing
        # only the incoming edges of the variables in the subset.
        tmp_samples[:, parents_subset] = distribution_samples[:, parents_subset][rnd_permutation]
        cond_dist_removed_arr_samples = draw_samples_func(tmp_samples)

        old_average_difference_result = average_difference_result

        aggregated_conditional_difference_result += difference_estimation_func(
            conditional_distribution_samples, cond_dist_removed_arr_samples
        )

        average_difference_result = aggregated_conditional_difference_result / (run + 1)

        if run >= max_num_runs:
            break
        elif run > 0:
            if old_average_difference_result == 0:
                converging = average_difference_result == 0
            else:
                converging = abs(1 - average_difference_result / old_average_difference_result) < tolerance

            if converging:
                converged_run += 1
                if converged_run >= 3:
                    break
            else:
                converged_run = 0

    return average_difference_result


def intrinsic_causal_influence(
    causal_model: StructuralCausalModel,
    target_node: Any,
    prediction_model: Union[PredictionModel, ClassificationModel, str] = "approx",
    attribution_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    num_training_samples: int = 100000,
    num_samples_randomization: int = 250,
    num_samples_baseline: int = 1000,
    max_batch_size: int = -1,
    auto_assign_quality: auto.AssignmentQuality = auto.AssignmentQuality.GOOD,
    shapley_config: Optional[ShapleyConfig] = None,
) -> Dict[Any, float]:
    """Computes the causal contribution of each upstream noise term of the target node (including the noise of the
    target itself) to the statistical property (e.g. mean, variance) of the target. We call this contribution
    *intrinsic* as noise terms, by definition, do not inherit properties of observed parents. The contribution of each
    noise term is then the *intrinsic* causal contribution of the corresponding node. For more scientific details,
    please refer to the paper below.

    **Research Paper**:
    Janzing et al. *Quantifying causal contributions via structure preserving interventions*. arXiv:2007.00714, 2021.

    :param causal_model: The structural causal model for whose target node we compute the intrinsic causal influence
                         of its ancestors.
    :param target_node: Target node whose statistical property is to be attributed.
    :param prediction_model: Prediction model for estimating the functional relationship between subsets of ancestor
                             noise terms and the target node. This can be an instance of a PredictionModel, the string
                             'approx' or the string 'exact'. With 'exact', the underlying causal models in the graph
                             are utilized directly by propagating given noise inputs through the graph, which ensures
                             that generated samples follow the fitted models. In contrast, the 'approx' method involves
                             selecting and training a suitable model based on data sampled from the graph. This might
                             lead to deviations from the outcomes of the fitted models, but is faster and can be more
                             robust in certain settings.
    :param attribution_func: Optional attribution function to measure the statistical property of the target node. This
                             function expects two inputs; predictions after the randomization of certain features (i.e.
                             samples from noise nodes) and a baseline where no features were randomized. The baseline
                             predictions can be typically ignored if one is interested in uncertainty measures such as
                             entropy or variance, but they might be relevant if, for instance, these shall be estimated
                             based on the residuals. By default, entropy is used if prediction model is a classifier,
                             variance otherwise.
    :param num_training_samples: Number of samples drawn from the graphical causal model that are used for fitting the
                                 prediction_model (if necessary).
    :param num_samples_randomization: Number of noise samples drawn from the graphical causal model that are used for
                                      evaluating the set function. Here, these samples are samples from
                                      the noise distributions used for randomizing features that are not in the subset.
    :param num_samples_baseline: Number of noise samples drawn from the graphical causal model that are used for
                                 evaluating the set function. Here, these samples are used as fixed observations for
                                 features that are in the subset.
    :param max_batch_size: Maximum batch size for estimating the predictions from evaluation samples. This has a
                           significant impact on the overall memory usage. If set to -1, all samples are used in one
                           batch.
    :param auto_assign_quality: Auto assign quality for the 'approx' prediction_model option.
    :param shapley_config: :class:`~dowhy.gcm.shapley.ShapleyConfig` for the Shapley estimator.
    :return: Intrinsic causal contribution of each ancestor node to the statistical property defined by the
             attribution_func of the target node.
    """
    validate_causal_dag(causal_model.graph)

    # Creating a smaller subgraph, which only contains upstream nodes that are connected to the target node.
    sub_causal_model = StructuralCausalModel(node_connected_subgraph_view(causal_model.graph, target_node))

    data_samples, noise_samples = noise_samples_of_ancestors(sub_causal_model, target_node, num_training_samples)
    node_names = noise_samples.columns
    noise_samples, target_samples = shape_into_2d(noise_samples.to_numpy(), data_samples[target_node].to_numpy())

    target_is_categorical = is_categorical(data_samples[target_node].to_numpy())

    prediction_method = _get_icc_noise_function(
        causal_model,
        target_node,
        prediction_model,
        noise_samples,
        node_names,
        target_samples,
        auto_assign_quality,
        target_is_categorical,
    )

    if attribution_func is None:
        if target_is_categorical:

            def attribution_func(x, _):
                return -estimate_entropy_of_probabilities(x)

        else:

            def attribution_func(x, _):
                return estimate_variance(x)

    _, noise_samples = noise_samples_of_ancestors(
        sub_causal_model, target_node, num_samples_randomization + num_samples_baseline
    )
    noise_samples = shape_into_2d(noise_samples.to_numpy())

    iccs = _estimate_iccs(
        attribution_func,
        prediction_method,
        noise_samples[:num_samples_randomization],
        noise_samples[num_samples_randomization : num_samples_randomization + num_samples_baseline],
        max_batch_size,
        ShapleyConfig() if shapley_config is None else shapley_config,
    )

    return {node: iccs[i] for i, node in enumerate(node_names)}


def intrinsic_causal_influence_sample(
    causal_model: InvertibleStructuralCausalModel,
    target_node: Any,
    baseline_samples: pd.DataFrame,
    noise_feature_samples: Optional[pd.DataFrame] = None,
    prediction_model: Union[PredictionModel, ClassificationModel, str] = "approx",
    subset_scoring_func: Optional[Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]] = None,
    num_noise_feature_samples: int = 5000,
    max_batch_size: int = 100,
    auto_assign_quality: auto.AssignmentQuality = auto.AssignmentQuality.GOOD,
    shapley_config: Optional[ShapleyConfig] = None,
) -> List[Dict[Any, Any]]:
    """Estimates the intrinsic causal impact of upstream nodes on a specified target_node, using the provided
    baseline_samples as a reference. In this context, observed values are attributed to the noise factors present in
    upstream nodes. Compared to intrinsic_causal_influence, this method quantifies the influences with respect to single
    observations instead of the distribution. Note that the current implementation only supports non-categorical data,
    since the noise terms need to be reconstructed.

    **Research Paper**:
    Janzing et al. *Quantifying causal contributions via structure preserving interventions*. arXiv:2007.00714, 2021.

    :param causal_model: The fitted invertible structural causal model.
    :param target_node: Node of interest.
    :param baseline_samples: Samples for which the influence should be estimated.
    :param noise_feature_samples: Optional noise samples of upstream nodes used as 'background' samples. If None is
                                  given, new noise samples are generated based on the graph. These samples are used for
                                  randomizing features that are not in the subset.
    :param prediction_model: Prediction model for estimating the functional relationship between subsets of ancestor
                             noise terms and the target node. This can be an instance of a PredictionModel, the string
                             'approx' or the string 'exact'. With 'exact', the underlying causal models in the graph
                             are utilized directly by propagating given noise inputs through the graph, which ensures
                             that generated samples follow the fitted models. In contrast, the 'approx' method involves
                             selecting and training a suitable model based on data sampled from the graph. This might
                             lead to deviations from the outcomes of the fitted models, but is faster and can be more
                             robust in certain settings.
    :param subset_scoring_func: Set function for estimating the quantity of interest based. This function
                                expects two inputs; the outcome of the model for some samples if certain features are permuted and the
                                outcome of the model for the same samples when no features were permuted. By default,
                                the difference between means of these samples are estimated.
    :param num_noise_feature_samples: If no noise_feature_samples are given, noise samples are drawn from the graph.
                                      This parameter indicates how many.
    :param max_batch_size: Maximum batch size for estimating multiple predictions at once. This has a significant influence on the
                          overall memory usage. If set to -1, all samples are used in one batch.
    :param auto_assign_quality: Auto assign quality for the 'approx' prediction_model option.
    :param shapley_config: :class:`~dowhy.gcm.shapley.ShapleyConfig` for the Shapley estimator.
    :return: A list of dictionaries indicating the intrinsic causal influence of a node on the target for a particular
             sample. This is, each dictionary belongs to one baseline sample.
    """
    validate_node(causal_model.graph, target_node)
    causal_model = InvertibleStructuralCausalModel(node_connected_subgraph_view(causal_model.graph, target_node))

    feature_samples, tmp_noise_feature_samples = noise_samples_of_ancestors(
        causal_model, target_node, num_noise_feature_samples
    )

    if has_categorical(feature_samples.to_numpy()):
        raise ValueError(
            "The current implementation requires all variables to be numeric, i.e., non-categorical! "
            "There is at least one node in the graph that is categorical."
        )

    if noise_feature_samples is None:
        noise_feature_samples = tmp_noise_feature_samples

    if subset_scoring_func is None:
        subset_scoring_func = means_difference

    target_samples = feature_samples[target_node].to_numpy()
    node_names = noise_feature_samples.columns
    noise_feature_samples, target_samples = shape_into_2d(noise_feature_samples.to_numpy(), target_samples)

    prediction_method = _get_icc_noise_function(
        causal_model,
        target_node,
        prediction_model,
        noise_feature_samples,
        node_names,
        target_samples,
        auto_assign_quality,
        False,  # Currently only supports continues target since we need to reconstruct its noise term.
    )

    shapley_vales = feature_relevance_sample(
        prediction_method,
        feature_samples=noise_feature_samples,
        baseline_samples=compute_noise_from_data(causal_model, baseline_samples)[node_names].to_numpy(),
        subset_scoring_func=subset_scoring_func,
        max_batch_size=max_batch_size,
        shapley_config=shapley_config,
    )

    return [
        {(predecessor, target_node): shapley_vales[i][q] for q, predecessor in enumerate(node_names)}
        for i in range(shapley_vales.shape[0])
    ]


def _estimate_iccs(
    attribution_func: Callable[[np.ndarray, np.ndarray], float],
    prediction_method: Callable[[np.ndarray], np.ndarray],
    noise_samples: np.ndarray,
    baseline_noise_samples: np.ndarray,
    max_batch_size: int,
    shapley_config: ShapleyConfig,
):
    target_values = shape_into_2d(prediction_method(baseline_noise_samples))

    def icc_set_function(subset: np.ndarray) -> Union[np.ndarray, float]:
        if np.all(subset == 1):
            # In case of the full subset (no randomization), we get the same predictions as when we apply the
            # prediction method to the samples of interest, since all noise samples are replaced with a sample of
            # interest.
            predictions = target_values
        elif np.all(subset == 0):
            # In case of the empty subset (all are jointly randomize), it boils down to taking the average over all
            # predictions, seeing that the randomization yields the same values for each sample of interest (none of the
            # samples of interest are used to replace a (jointly) 'randomized' sample).
            predictions = np.tile(
                np.mean(prediction_method(noise_samples), axis=0), (baseline_noise_samples.shape[0], 1)
            )
        else:
            predictions = marginal_expectation(
                prediction_method,
                feature_samples=noise_samples,
                baseline_samples=baseline_noise_samples,
                baseline_feature_indices=np.arange(0, noise_samples.shape[1])[subset == 1],
                return_averaged_results=True,
                feature_perturbation="randomize_columns_jointly",
                max_batch_size=max_batch_size,
            )
        return attribution_func(shape_into_2d(predictions), target_values)

    return estimate_shapley_values(icc_set_function, noise_samples.shape[1], shapley_config)


def _get_icc_noise_function(
    causal_model: StructuralCausalModel,
    target_node: Any,
    prediction_model: Union[PredictionModel, ClassificationModel, str],
    noise_samples: np.ndarray,
    node_names: Iterator[Any],
    target_samples: np.ndarray,
    auto_assign_quality: auto.AssignmentQuality,
    target_is_categorical: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(prediction_model, str) and prediction_model not in ("approx", "exact"):
        raise ValueError(
            "Invalid value for prediction_model: %s! This should either be an instance of a PredictionModel or"
            "one of the two string options 'exact' or 'approx'." % prediction_model
        )

    if not isinstance(prediction_model, str):
        prediction_model.fit(noise_samples, target_samples)

        return prediction_model.predict

    if prediction_model == "approx":
        prediction_model = auto.select_model(noise_samples, target_samples, auto_assign_quality)[0]
        prediction_model.fit(noise_samples, target_samples)

        if target_is_categorical:
            return prediction_model.predict_probabilities
        else:
            return prediction_model.predict
    else:
        # Exact model
        def exact_model(X: np.ndarray) -> np.ndarray:
            return compute_data_from_noise(causal_model, pd.DataFrame(X, columns=[x for x in node_names]))[
                target_node
            ].to_numpy()

        if target_is_categorical:
            list_of_classes = cast(ClassifierFCM, causal_model.causal_mechanism(target_node)).classifier_model.classes

            def prediction_method(X):
                return (shape_into_2d(exact_model(X)) == list_of_classes).astype(float)

            return prediction_method
        else:
            return exact_model
