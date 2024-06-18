from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from dowhy.gcm import config
from dowhy.gcm._noise import compute_noise_from_data, get_noise_dependent_function, noise_samples_of_ancestors
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm.anomaly_scorers import MedianCDFQuantileScorer, RescaledMedianCDFQuantileScorer
from dowhy.gcm.causal_mechanisms import ConditionalStochasticModel
from dowhy.gcm.causal_models import InvertibleStructuralCausalModel, ProbabilisticCausalModel, validate_causal_dag
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values
from dowhy.gcm.stats import permute_features
from dowhy.gcm.util.general import shape_into_2d
from dowhy.graph import get_ordered_predecessors, is_root_node


def conditional_anomaly_scores(
    parent_samples: np.ndarray,
    target_samples: np.ndarray,
    causal_mechanism: ConditionalStochasticModel,
    anomaly_scorer_factory: Callable[[], AnomalyScorer] = MedianCDFQuantileScorer,
    num_samples_conditional: int = 10000,
) -> np.ndarray:
    """Estimates the conditional anomaly scores based on the expected outcomes of the causal model.

    :param parent_samples: Samples from all parents of the target node.
    :param target_samples: Samples from the target node.
    :param causal_mechanism: Causal mechanism of the target node.
    :param anomaly_scorer_factory: A callable that returns an anomaly scorer.
    :param num_samples_conditional: Number of samples drawn from the conditional distribution based on the given
                                    parent samples. The more samples, the more accurate the results.
    :return: The conditional anomaly score for each sample in target_samples.
    """
    parent_samples, target_samples = shape_into_2d(parent_samples, target_samples)
    if parent_samples.shape[0] != target_samples.shape[0]:
        raise ValueError("There should be as many parent samples as target samples!")

    result = np.zeros(parent_samples.shape[0])
    for i in range(parent_samples.shape[0]):
        samples_from_conditional = causal_mechanism.draw_samples(
            np.tile(parent_samples[i], (num_samples_conditional, 1))
        )
        anomaly_scorer = anomaly_scorer_factory()
        anomaly_scorer.fit(samples_from_conditional)
        result[i] = anomaly_scorer.score(target_samples[i])[0]

    return result


def anomaly_scores(
    causal_model: ProbabilisticCausalModel,
    anomaly_data: pd.DataFrame,
    num_samples_conditional: int = 10000,
    num_samples_unconditional: int = 10000,
    anomaly_scorer_factory: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer,
) -> Dict[Any, np.ndarray]:
    if isinstance(anomaly_data, pd.Series):
        anomaly_data = pd.DataFrame([anomaly_data])

    validate_causal_dag(causal_model.graph)

    results = {}
    for node in tqdm(
        causal_model.graph.nodes,
        desc="Estimating conditional anomaly scores",
        position=0,
        leave=True,
        disable=not config.show_progress_bars,
    ):

        if is_root_node(causal_model.graph, node):
            anomaly_scorer = anomaly_scorer_factory()
            anomaly_scorer.fit(causal_model.causal_mechanism(node).draw_samples(num_samples_unconditional))
            results[node] = anomaly_scorer.score(anomaly_data[node].to_numpy())
        else:
            tmp_anomaly_parent_samples = anomaly_data[get_ordered_predecessors(causal_model.graph, node)].to_numpy()
            tmp_anomaly_target_samples = anomaly_data[node].to_numpy()
            results[node] = conditional_anomaly_scores(
                tmp_anomaly_parent_samples,
                tmp_anomaly_target_samples,
                causal_model.causal_mechanism(node),
                anomaly_scorer_factory,
                num_samples_conditional,
            )

    return results


def attribute_anomalies(
    causal_model: InvertibleStructuralCausalModel,
    target_node: Any,
    anomaly_samples: pd.DataFrame,
    anomaly_scorer: Optional[AnomalyScorer] = None,
    attribute_mean_deviation: bool = False,
    num_distribution_samples: int = 3000,
    shapley_config: Optional[ShapleyConfig] = None,
) -> Dict[Any, np.ndarray]:
    """Estimates the contributions of upstream nodes to the anomaly score of the target_node for each sample in
    anomaly_samples. By default, the anomaly score is based on the information theoretic (IT) score
    -log(P(g(X) >= g(x))), where g is the anomaly_scorer, X samples from the marginal
    distribution of the target_node and x an observation of the target_node in anomaly_samples. If
    attribute_mean_deviation is set to True, the contribution to g(x) - E[g(X)] is estimated instead, i.e. the feature
    relevance for the given scoring function. The underlying algorithm utilizes the reconstructed noise of upstream
    nodes (including the target_node itself) for the given anomaly_samples. By this, it is possible to estimate how
    much of the anomaly score can be explained by upstream anomalies with respect to anomalous noise values.

    Note: This function requires that the noise can be recovered from samples, i.e. the causal models of non-root nodes
    need to be an InvertibleNoiseModel (e.g. AdditiveNoiseModel).

    Related paper:
    Janzing, D., Budhathoki, K., Minorics, L., & Bloebaum, P. (2019).
    Causal structure based root cause analysis of outliers
    https://arxiv.org/abs/1912.02724

    :param causal_model: The fitted InvertibleStructuralCausalModel.
    :param target_node: Target node for which the contributions are estimated.
    :param anomaly_samples: Anomalous observations for which the contributions are estimated.
    :param anomaly_scorer: Anomaly scorer g. If None is given, a MedianCDFQuantileScorer is used.
    :param attribute_mean_deviation: If set to False, the contribution is estimated based on the IT score and if it is
                                     set to True, the contribution is based on the feature relevance with respect to the given scoring function.
    :param num_distribution_samples: Number of samples from X, the marginal distribution of the target. These are used
                                     for evaluating the tail probability in case of the IT score
                                     (attribute_mean_deviation is False) or as samples for randomization in case of
                                     feature relevance (attribute_mean_deviation is True).
    :param shapley_config: :class:`~dowhy.gcm.shapley.ShapleyConfig` for the Shapley estimator.
    :return: A dictionary that assigns a numpy array to each upstream node including the target_node itself. The
             i-th entry of an array indicates the contribution of the corresponding node to the anomaly score of the target
             for the i-th observation in anomaly_samples.
    """
    validate_causal_dag(causal_model.graph)

    if anomaly_scorer is None:
        anomaly_scorer = MedianCDFQuantileScorer()

    noise_of_anomaly_samples = compute_noise_from_data(causal_model, anomaly_samples)
    node_samples, noise_samples = noise_samples_of_ancestors(causal_model, target_node, num_distribution_samples)
    noise_dependent_function, nodes_order = get_noise_dependent_function(causal_model, target_node)
    anomaly_scorer.fit(node_samples[target_node].to_numpy())

    attributions = attribute_anomaly_scores(
        noise_of_anomaly_samples[nodes_order].to_numpy(),
        noise_samples[nodes_order].to_numpy(),
        lambda x: anomaly_scorer.score(noise_dependent_function(x)),
        attribute_mean_deviation,
        shapley_config,
    )

    return {node: attributions[:, i] for i, node in enumerate(nodes_order)}


def attribute_anomaly_scores(
    anomaly_samples: np.ndarray,
    distribution_samples: np.ndarray,
    anomaly_scoring_func: Callable[[np.ndarray], np.ndarray],
    attribute_mean_deviation: bool,
    shapley_config: Optional[ShapleyConfig] = None,
) -> np.ndarray:
    """Estimates the contributions of the features for each sample in anomaly_samples to the anomaly score obtained
    by the anomaly_scoring_func. If attribute_mean_deviation is set to False, the anomaly score is based on the
    information theoretic (IT) score -log(P(g(X) >= g(x))), where g is the anomaly_scoring_func, X samples from the
    marginal distribution of the target_node and x an observation of the target_node in anomaly_samples. If
    attribute_mean_deviation is set to True, the contribution to g(x) - E[g(X)] is estimated instead, i.e. the
    feature relevance for the given scorer.

    Note that the anomaly scoring function needs to handle the dimension and modality of the data. An example for a
    function for multidimensional continues data would be:
        density_estimator = GaussianMixtureDensityEstimator()
        density_estimator.fit(original_observations)
        anomaly_scoring_func = lambda x, y: estimate_inverse_density_score(x, y, density_estimator)

    Related paper:
    Janzing, D., Budhathoki, K., Minorics, L., & Bloebaum, P. (2022).
    Causal structure based root cause analysis of outliers
    https://arxiv.org/abs/1912.02724

    :param anomaly_samples: Samples x for which the contributions are estimated. The dimensionality of these samples
                            doesn't matter as long as the anomaly_scoring_func supports it.
    :param distribution_samples: Samples from the (non-anomalous) distribution X.
    :param anomaly_scoring_func: A function g that takes a sample from X as input and returns an anomaly score.
    :param attribute_mean_deviation: If set to False, the contribution is estimated based on the IT score and if it is
                                     set to True, the contribution is based on the feature relevance with respect to the
                                     given scoring function.
    :param shapley_config: :class:`~dowhy.gcm.shapley.ShapleyConfig` for the Shapley estimator.
    :return: A numpy array with the feature contributions to the anomaly score for each sample in anomaly_samples.
    """
    if attribute_mean_deviation:
        expectation_of_score = np.mean(anomaly_scoring_func(distribution_samples))
    else:
        anomaly_scores = anomaly_scoring_func(anomaly_samples)

    def set_function(subset: np.ndarray) -> Union[np.ndarray, float]:
        feature_samples = permute_features(distribution_samples, np.arange(0, subset.shape[0])[subset == 0], True)

        result = np.zeros(anomaly_samples.shape[0])
        for i in range(anomaly_samples.shape[0]):
            feature_samples[:, subset == 1] = anomaly_samples[i, subset == 1]

            if attribute_mean_deviation:
                # Usual feature relevance using the mean deviation as set function, i.e. g(x) - E[g(X)]
                result[i] = np.mean(anomaly_scoring_func(feature_samples)) - expectation_of_score
            else:
                result[i] = np.log(_relative_frequency(anomaly_scoring_func(feature_samples) >= anomaly_scores[i]))

        return result

    return estimate_shapley_values(set_function, anomaly_samples.shape[1], shapley_config)


def _relative_frequency(conditions: np.ndarray):
    return (np.sum(conditions) + 0.5) / (len(conditions) + 0.5)
