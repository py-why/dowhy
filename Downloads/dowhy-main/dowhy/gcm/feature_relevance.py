"""This module allows to estimate the feature relevance of inputs with respect to a given model. While these models can
be blackbox prediction models, it is also possible to explain causal mechanisms with respect to the direct parents.
In these cases, it would be possible to incorporate the noise to represent the part of the generation process that
cannot be explained by the parents."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dowhy.gcm.causal_mechanisms import ProbabilityEstimatorModel
from dowhy.gcm.causal_models import StructuralCausalModel, validate_node
from dowhy.gcm.fitting_sampling import draw_samples
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values
from dowhy.gcm.stats import marginal_expectation
from dowhy.gcm.util.general import shape_into_2d, variance_of_deviations, variance_of_matching_values
from dowhy.graph import get_ordered_predecessors, is_root_node


def parent_relevance(
    causal_model: StructuralCausalModel,
    target_node: Any,
    parent_samples: Optional[pd.DataFrame] = None,
    subset_scoring_func: Optional[Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]] = None,
    num_samples_randomization: int = 5000,
    num_samples_baseline: int = 500,
    max_batch_size: int = 100,
    shapley_config: Optional[ShapleyConfig] = None,
) -> Tuple[Dict[Any, Any], np.ndarray]:
    """Estimates the distribution based relevance of the direct parents of the given target_node. This is, the
    relevance of direct parents as input features of the the underlying causal model of target_node. Here, the
    unobserved noise is considered as a direct parent (input) as well. Samples utilized for the estimation are drawn
    from the given causal graph.

    By default, the used subset_scoring_func is based on the variance between Y and Y', where Y is the outputs of
    the causal model and Y' the outputs of the models when certain features are randomized. In case of continuous data,
    the feature relevance adds up to Var(Y - Y').

    Note: The feature relevance based on the distribution cannot be directly compared with the feature relevance for
    single samples. If this is desired, the set function needs to be defined accordingly.

    Related paper:
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR.

    :param causal_model: The fitted structural causal model.
    :param target_node: Node with the causal model of interest.
    :param parent_samples: Samples for the parents of the given target_node. If None is given, new samples are
                           generated based on the graph. These samples are used for randomizing features that are not in the subset.
    :param subset_scoring_func: Set function for estimating the quantity of interest based on the model outcomes. This function
                                expects two inputs; the outcome of the causal model for some samples if certain features are permuted and the
                                outcome of the model for the same samples when no features were permuted. The set functions represents the
                                comparison between the samples, for instance, the variance of deviations. This is then used as the 'characteristic function'
                                in coalition games when estimating the Shapley values.
    :param num_samples_randomization: Number of samples used as background parent samples for evaluating the set function.
                                      If no parent_samples are given, this represents the number of generated samples from
                                      the joint distribution of the parents and are used for randomizing features that are
                                      not in the subset. Consider increasing this number for more accurate results or reducing it for less memory consumption and faster runtime.
    :param num_samples_baseline: Number of samples on which the set functions are evaluated on. These samples are used as fixed observations for
                                 parents that are in the subset. Consider increasing this number for more accurate results or reducing it for less memory consumption and faster runtime.
    :param max_batch_size: Maximum batch size for estimating multiple predictions at once. This has a significant influence on the
                          overall memory usage. If set to -1, all samples are used in one batch.
    :param shapley_config: :class:`~dowhy.gcm.shapley.ShapleyConfig` for the Shapley estimator.
    :return: There are two return vales. A dictionary with the feature relevance for each direct parent of the given
             target_node and the feature relevance of noise.
    """
    validate_node(causal_model.graph, target_node)

    if is_root_node(causal_model.graph, target_node):
        raise ValueError(
            "Cannot compute feature relevance of parents for the target node %s as it is a root node."
            "It does not have parents." % target_node
        )

    ordered_predecessors = get_ordered_predecessors(causal_model.graph, target_node)

    if parent_samples is None:
        parent_samples = draw_samples(causal_model, max(num_samples_randomization, num_samples_baseline))[
            ordered_predecessors
        ]

    if subset_scoring_func is None:
        if isinstance(causal_model.causal_mechanism(target_node), ProbabilityEstimatorModel):
            subset_scoring_func = variance_of_matching_values
        else:
            subset_scoring_func = variance_of_deviations

    parent_samples = shape_into_2d(parent_samples[ordered_predecessors].to_numpy())
    noise_samples = shape_into_2d(
        causal_model.causal_mechanism(target_node).draw_noise_samples(parent_samples.shape[0])
    )
    samples_features = np.column_stack([parent_samples.astype(object), noise_samples.astype(object)])

    def model(X: np.ndarray) -> np.ndarray:
        return causal_model.causal_mechanism(target_node).evaluate(
            X[:, : -noise_samples.shape[1]], X[:, -noise_samples.shape[1] :]
        )

    shapley_vales = feature_relevance_distribution(
        model,
        feature_samples=samples_features,
        subset_scoring_func=subset_scoring_func,
        max_num_samples_randomization=num_samples_randomization,
        max_num_baseline_samples=num_samples_baseline,
        max_batch_size=max_batch_size,
        shapley_config=shapley_config,
    )

    result = {(predecessor, target_node): shapley_vales[i] for i, predecessor in enumerate(ordered_predecessors)}

    return result, shapley_vales[-noise_samples.shape[1] :]


def feature_relevance_distribution(
    prediction_method: Callable[[np.ndarray], np.ndarray],
    feature_samples: np.ndarray,
    subset_scoring_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
    max_num_samples_randomization: int = 5000,
    max_num_baseline_samples: int = 500,
    max_batch_size: int = 100,
    randomize_features_jointly: bool = True,
    shapley_config: Optional[ShapleyConfig] = None,
) -> np.ndarray:
    """Estimates the population based feature relevance of the input features for the given prediction_method. This
    method uses all samples given in feature_samples by comparing the output of the prediction_method given certain
    features are randomized with the outputs when no features are randomized. The subset_scoring_func defines
    how these predictions are compared. For instance, the variance of deviations.

    If the randomized predictions should rather be compared to the original data, this has (and can) be defined via
    the set function by ignoring the second input parameter (the predicted values using all feauters). Instead, the
    original data can be used.

    Note: The distribution level relevance is estimated by taking the expectation of the outcome of the set functions
    when applied to multiple samples. Due to the linearity of the Shapley value estimation, this is equivalent to taking
    the expectation over the Shapley values.

    Related paper:
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR.

    :param prediction_method: A callable that is expected to return a prediction for given samples.
    :param feature_samples: Samples from the joint distribution.
    :param subset_scoring_func: Set function for estimating the quantity of interest based on the model outcomes. This function
                                expects two inputs; the outcome of the prediction model for some samples if certain features are permuted and the
                                outcome of the model for the same samples when no features were permuted. The set functions represents the
                                comparison between the samples, for instance, the variance of deviations. This is then used as the 'characteristic function'
                                in coalition games when estimating the Shapley values.
    :param max_num_samples_randomization: Maximum number of samples used for randomizing the feature that are not in the susbet. Consider increasing this
                                          number for more accurate results (if enough samples are available) or reducing it for less memory consumption and
                                          faster runtime.
    :param max_num_baseline_samples: Maximum number of samples on which the set function is evaluated on. These samples are used as fixed observations for
                                     features that are in the subset. For instance, in case of taking the mean as set_function_summary_func, this defines the maximum number
                                     of samples used to estimate the mean. Consider increasing this number for more accurate results (if enough samples are
                                     available) or reducing it for less memory consumption and faster runtime.
    :param max_batch_size: Maximum batch size for a estimating the predictions. This has a significant influence on the
                           overall memory usage. If set to -1, all samples are used in one batch.
    :param randomize_features_jointly: If set to True, features that are not in a subset are jointly permuted.
                                       Note that this still represents an interventional distribution. If set to False, features that are not in a subset
                                       are independently permuted. Note: The theory in the linked publication assumes that this is set to True.
    :param shapley_config: Config for the Shapley estimator.
    :return: A numpy array with the feature relevance of each input feature.
    """
    feature_samples = shape_into_2d(feature_samples)

    if shapley_config is None:
        shapley_config = ShapleyConfig()

    baseline_samples = feature_samples[
        np.random.choice(
            feature_samples.shape[0], min(max_num_baseline_samples, feature_samples.shape[0]), replace=False
        )
    ]
    feature_samples = feature_samples[
        np.random.choice(
            feature_samples.shape[0], min(max_num_samples_randomization, feature_samples.shape[0]), replace=False
        )
    ]

    return feature_relevance_sample(
        prediction_method,
        feature_samples,
        baseline_samples,
        subset_scoring_func,
        None,
        True,
        max_batch_size,
        randomize_features_jointly,
        shapley_config,
    )


def feature_relevance_sample(
    prediction_method: Callable[[np.ndarray], np.ndarray],
    feature_samples: np.ndarray,
    baseline_samples: np.ndarray,
    subset_scoring_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
    baseline_target_values: Optional[np.ndarray] = None,
    average_set_function: bool = False,
    max_batch_size: int = 100,
    randomize_features_jointly: bool = True,
    shapley_config: Optional[ShapleyConfig] = None,
) -> np.ndarray:
    """Estimates the feature relevance of the prediction_method for each sample in baseline_noise_samples. This
    method uses all samples given in feature_samples as 'background' samples. This is, they should represent samples
    from the joint distribution of the input features. The subset_scoring_func defines the comparison between the
    output of the prediction_method when certain features are randomized and the outputs when no features are
    randomized. The most common function would be the difference between the expectations.

    If the randomized predictions should rather be compared to the original data, this has (and can) be defined via
    the set function by ignoring the second input parameter (the predicted values using all feauters). Instead, the
    original data can be used.

    Related paper:
    Janzing, D., Minorics, L., & Bloebaum, P. (2020).
    Feature relevance quantification in explainable AI: A causal problem.
    In International Conference on Artificial Intelligence and Statistics (pp. 2907-2916). PMLR.

    :param prediction_method: A callable that is expected to return a prediction for given samples.
    :param feature_samples: Samples from the joint distribution. These are used as 'background samples' to randomize features that are not in a subset.
    :param baseline_samples: Samples for which the feature relevance should be estimated.
    :param subset_scoring_func: Set function for estimating the quantity of interest based on the model outcomes. This function
                                expects two inputs; the outcome of the prediction model for some samples if certain features are permuted and the
                                outcome of the model for the same samples when no features were permuted. A typical choice for regression models
                                would be the difference between expectations. This is then used as the 'characteristic function'
                                in coalition games when estimating the Shapley values.
    :param baseline_target_values: These baseline values are compared with the subset specific outcomes of the prediction method. If set to None (default),
                                   the baseline values are the outcomes of the given prediction_method applied to the baseline_noise_samples, i.e. the outcome of the empty subset.
    :param max_batch_size: Maximum batch size for a estimating the predictions. This has a significant influence on the
                           overall memory usage. If set to -1, all samples are used in one batch.
    :param average_set_function: If set to True, the averaged result of the set function applied to each sample of
                                 interest is used for estimating the Shapley values. If set to False, Shapley values for each sample of interest
                                 are estimated separately.
    :param randomize_features_jointly: If set to True, features that are not in a subset are jointly permuted.
                                       Note that this still represents an interventional distribution. If set to False, features that are not in a subset
                                       are independently permuted. Note: The theory in the linked publication assumes that this is set to True.
    :param shapley_config: Config for the Shapley estimator.
    :return: A numpy array with the feature relevance for each sample in baseline_noise_samples.
    """
    feature_samples = shape_into_2d(feature_samples)
    if baseline_target_values is None:
        baseline_target_values = prediction_method(baseline_samples)
    else:
        if baseline_samples.shape[0] != baseline_target_values.shape[0]:
            raise ValueError(
                "Samples of interest and the given baseline values need to have the same sample size! "
                "Make sure that the given baseline values correspond to the samples of interest."
            )

    if shapley_config is None:
        shapley_config = ShapleyConfig()

    def single_sample_set_function(subset: np.ndarray) -> Union[np.ndarray, float]:
        results = np.zeros(baseline_target_values.shape[0])
        predictions = marginal_expectation(
            prediction_method,
            feature_samples=feature_samples,
            baseline_samples=baseline_samples,
            baseline_feature_indices=np.arange(0, feature_samples.shape[1])[subset == 1],
            return_averaged_results=False,
            feature_perturbation=(
                "randomize_columns_jointly" if randomize_features_jointly else "randomize_columns_independently"
            ),
            max_batch_size=max_batch_size,
        )

        for i in range(baseline_target_values.shape[0]):
            results[i] = subset_scoring_func(predictions[i], baseline_target_values[i])

        if average_set_function:
            return np.mean(results)
        else:
            return results

    return estimate_shapley_values(single_sample_set_function, feature_samples.shape[1], shapley_config)
