import operator

import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    InverseDensityScorer,
    ITAnomalyScorer,
    MeanDeviationScorer,
    MedianCDFQuantileScorer,
    MedianDeviationScorer,
    ProbabilisticCausalModel,
    RescaledMedianCDFQuantileScorer,
    anomaly_scores,
    auto,
    fit,
)
from dowhy.gcm.anomaly import conditional_anomaly_scores
from dowhy.gcm.constant import EPS
from dowhy.gcm.distribution_change import estimate_distribution_change_scores
from dowhy.gcm.ml import create_linear_regressor


def test_given_outlier_observation_when_estimate_anomaly_scores_using_median_cdf_quantile_scorer_then_returns_expected_result():
    pcm, outlier_observation = _create_pcm_and_outlier_observation()

    assert (
        max(
            anomaly_scores(pcm, outlier_observation, anomaly_scorer_factory=MedianCDFQuantileScorer).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )

    assert (
        max(
            anomaly_scores(
                pcm, outlier_observation, anomaly_scorer_factory=lambda: ITAnomalyScorer(MedianCDFQuantileScorer())
            ).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )


def test_given_outlier_observation_when_estimate_anomaly_scores_using_rescaled_median_cdf_quantile_scorer_then_returns_expected_result():
    pcm, outlier_observation = _create_pcm_and_outlier_observation()

    assert (
        max(
            anomaly_scores(pcm, outlier_observation, anomaly_scorer_factory=RescaledMedianCDFQuantileScorer).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )

    assert (
        max(
            anomaly_scores(
                pcm,
                outlier_observation,
                anomaly_scorer_factory=lambda: ITAnomalyScorer(RescaledMedianCDFQuantileScorer()),
            ).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )


def test_given_outlier_observation_when_estimate_anomaly_scores_using_mean_deviation_scorer_then_returns_expected_result():
    pcm, outlier_observation = _create_pcm_and_outlier_observation()

    assert (
        max(
            anomaly_scores(pcm, outlier_observation, anomaly_scorer_factory=lambda: MeanDeviationScorer()).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )

    assert (
        max(
            anomaly_scores(
                pcm, outlier_observation, anomaly_scorer_factory=lambda: ITAnomalyScorer(MeanDeviationScorer())
            ).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )


def test_given_outlier_observation_when_estimate_anomaly_scores_using_median_deviation_scorer_then_returns_expected_result():
    pcm, outlier_observation = _create_pcm_and_outlier_observation()

    assert (
        max(
            anomaly_scores(pcm, outlier_observation, anomaly_scorer_factory=lambda: MedianDeviationScorer()).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )

    assert (
        max(
            anomaly_scores(
                pcm, outlier_observation, anomaly_scorer_factory=lambda: ITAnomalyScorer(MedianDeviationScorer())
            ).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )


def test_given_outlier_observation_when_estimate_anomaly_scores_using_inverse_density_scorer_then_returns_expected_result():
    pcm, outlier_observation = _create_pcm_and_outlier_observation()

    assert (
        max(
            anomaly_scores(pcm, outlier_observation, anomaly_scorer_factory=lambda: InverseDensityScorer()).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X1"
    )


def test_given_data_with_change_in_mechanism_when_estimate_distribution_change_score_then_returns_expected_result():
    causal_model, original_observations = _create_scm_for_distribution_change()

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    assert (
        max(
            estimate_distribution_change_scores(causal_model, original_observations, outlier_observations).items(),
            key=operator.itemgetter(1),
        )[0]
        == "X2"
    )


def test_given_data_with_change_in_mechanism_when_estimate_distribution_change_score_using_difference_in_means_then_returns_expected_result():
    causal_model, original_observations = _create_scm_for_distribution_change()

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 2 + 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.mean(x) - np.mean(y)),
    )

    assert scores["X0"] == approx(0, abs=0.1)
    assert scores["X1"] == approx(0, abs=0.1)
    assert scores["X2"] == approx(2, abs=0.1)
    assert scores["X3"] == approx(0, abs=0.1)

    assert max(scores.items(), key=operator.itemgetter(1))[0] == "X2"


def test_given_data_with_change_in_root_node_when_estimate_distribution_change_score_using_difference_in_means_then_returns_expected_result():
    causal_model, original_observations = _create_scm_for_distribution_change()

    # Here, changing the mechanism.
    X0 = np.random.uniform(-1.5, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.mean(x) - np.mean(y)),
    )

    assert scores["X0"] == approx(0.25, abs=0.1)
    assert scores["X1"] == approx(0, abs=0.1)
    assert scores["X2"] == approx(0, abs=0.1)
    assert scores["X3"] == approx(0, abs=0.1)

    assert max(scores.items(), key=operator.itemgetter(1))[0] == "X0"


@flaky(max_runs=3)
def test_given_graph_with_multiple_parents_when_estimate_distribution_change_scores_then_returns_expected_result():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = np.random.uniform(-1, 1, 1000)
    X2 = X0 + X1

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = np.random.uniform(-1, 1, 1000)
    X2 = X0 + X1 + 1
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2")]))
    auto.assign_causal_mechanisms(causal_model, original_observations, auto.AssignmentQuality.GOOD)

    fit(causal_model, original_observations)

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.mean(x) - np.mean(y)),
    )

    assert scores["X0"] == 0
    assert scores["X1"] == 0
    assert scores["X2"] == approx(1, abs=0.005)


@flaky(max_runs=5)
def test_given_data_with_change_in_mechanism_when_estimate_distribution_change_score_using_difference_in_variance_then_returns_expected_result():
    causal_model, original_observations = _create_scm_for_distribution_change()

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 0.5 * X0 + np.random.normal(0, 2, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.var(x) - np.var(y)),
    )

    assert scores["X0"] == approx(0, abs=0.1)
    assert scores["X1"] == approx(0, abs=0.1)
    assert scores["X2"] == approx(4, abs=0.5)
    assert scores["X3"] == approx(0, abs=0.1)

    assert max(scores.items(), key=operator.itemgetter(1))[0] == "X2"


@flaky(max_runs=3)
def test_given_multivariate_inputs_when_estimate_anomaly_scores_then_does_not_raise_error():
    """This test verifies that estimate_anomaly_scores correctly handles multivariate input features, which
    caused problems in an earlier version."""

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X1", "X0"), ("X2", "X0"), ("X3", "X0"), ("X4", "X0")]))

    data = np.random.normal(0, 1, (10000, 4))
    data = pd.DataFrame(
        {
            "X0": (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]).reshape(-1),
            "X1": data[:, 0].reshape(-1),
            "X2": data[:, 1].reshape(-1),
            "X3": data[:, 2].reshape(-1),
            "X4": data[:, 3].reshape(-1),
        }
    )
    data_anomaly = pd.DataFrame(
        {
            "X0": [13],  # X1 + X2 + X3 + X4
            "X1": [10],
            "X2": [1],
            "X3": [1],
            "X4": [1],
        }
    )

    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)

    fit(causal_model, data)
    scores = anomaly_scores(causal_model, data_anomaly)
    assert scores["X1"][0] == approx(-np.log(1 / (data.shape[0] + 1)), abs=3)


@flaky(max_runs=3)
def test_given_simple_linear_data_when_estimate_conditional_anomaly_scores_then_returns_expected_result():
    X = np.random.normal(0, 1, 1000)
    N = np.random.normal(0, 1, 1000)

    Y = 2 * X + N

    causal_model = AdditiveNoiseModel(prediction_model=create_linear_regressor())
    causal_model.fit(X, Y)

    anomaly_scorer = MeanDeviationScorer()
    anomaly_scorer.fit(N)

    assert conditional_anomaly_scores(X[:5], Y[:5], causal_model, MeanDeviationScorer).reshape(-1) == approx(
        anomaly_scorer.score(Y[:5] - 2 * X[:5]), abs=0.1
    )


def _create_scm_for_distribution_change():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    causal_model.set_causal_mechanism("X1", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism("X3", AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    fit(causal_model, original_observations)

    return causal_model, original_observations


def _create_pcm_and_outlier_observation():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})
    outlier_observation = pd.DataFrame(
        {"X0": X0[:1], "X1": X1[:1] + np.std(X1.data) * 3, "X2": X2[:1], "X3": X3[:1]}  # Creating an anomaly here
    )

    pcm = ProbabilisticCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X2", "X3")]))
    auto.assign_causal_mechanisms(pcm, original_observations)
    fit(pcm, original_observations)

    return pcm, outlier_observation
