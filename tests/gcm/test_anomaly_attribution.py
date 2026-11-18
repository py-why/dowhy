from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    InverseDensityScorer,
    InvertibleStructuralCausalModel,
    MedianCDFQuantileScorer,
    attribute_anomalies,
    auto,
    fit,
)
from dowhy.gcm._noise import noise_samples_of_ancestors
from dowhy.gcm.anomaly import _relative_frequency, attribute_anomaly_scores
from dowhy.gcm.density_estimators import GaussianMixtureDensityEstimator
from dowhy.gcm.ml import PredictionModel, create_linear_regressor


def _attribute_anomalies_with_expected_total(causal_model, anomaly_data, num_samples, attribute_mean_deviation):
    sampled_target = None

    def record_samples(model, target_node, num_distribution_samples):
        nonlocal sampled_target
        node_samples, noise_samples = noise_samples_of_ancestors(model, target_node, num_distribution_samples)
        sampled_target = node_samples[target_node].to_numpy()
        return node_samples, noise_samples

    # Share the sampled reference, but compute the expected score independently of the coalition evaluator.
    with patch("dowhy.gcm.anomaly.noise_samples_of_ancestors", side_effect=record_samples):
        scores = attribute_anomalies(
            causal_model,
            "X3",
            anomaly_data,
            anomaly_scorer=MedianCDFQuantileScorer(),
            num_distribution_samples=num_samples,
            attribute_mean_deviation=attribute_mean_deviation,
        )

    assert sampled_target is not None
    scorer = MedianCDFQuantileScorer()
    scorer.fit(sampled_target)
    reference_scores = scorer.score(sampled_target)
    anomaly_scores = scorer.score(anomaly_data["X3"].to_numpy())
    if attribute_mean_deviation:
        expected_total = anomaly_scores - np.mean(reference_scores)
    else:
        expected_total = np.array(
            [-np.log((np.sum(reference_scores >= score) + 0.5) / (num_samples + 0.5)) for score in anomaly_scores]
        )
    return scores, expected_total


@flaky(max_runs=3)
def test_given_simple_gaussian_data_when_attribute_anomaly_scores_with_it_score_then_returns_qualitatively_correct_results():
    Z = np.random.normal(0, 1, 5000)
    X0 = Z + np.random.normal(0, 1, 5000)
    X1 = Z + np.random.normal(0, 1, 5000)
    X2 = Z + np.random.normal(0, 1, 5000)
    X3 = Z + np.random.normal(0, 1, 5000)

    original_observations = np.column_stack([X0, X1, X2, X3])

    # Defining an anomaly scorer that handles multidimensional inputs.
    density_estimator = GaussianMixtureDensityEstimator()
    anomaly_scorer = InverseDensityScorer(density_estimator)
    anomaly_scorer.fit(original_observations)

    # Seeing that the expectation of the noise in all nodes is 0, we introduce anomalies by setting some of them to 3.
    anomaly_samples = np.array([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 3, 3, 0], [3, 0, 3, 3]])

    contributions = attribute_anomaly_scores(anomaly_samples, original_observations, anomaly_scorer.score, False)

    # In the first sample, only the first variable is anomalous. Therefore, it should have the highest contribution
    # and it should be "significantly" higher than the contribution of the other ones (here, we just arbitrarily say
    # it should be 3x higher. Due to the confounding factor Z, the reconstructed noise variables are pairwise dependent,
    # which is a violation of our causal sufficiency assumption. However, a confounder is included here to demonstrate
    # some robustness. Note that due to this and stochastic behaviour of the density estimator, it is
    # not possible to analytically compute expected results. Therefore, we rather look at the relations here.
    assert np.argmax(contributions[0]) == 0
    assert np.all(contributions[0][0] > contributions[0][1:] * 3)

    # Same idea for the second sample, but here, it is the second variable that is anomalous.
    assert np.argmax(contributions[1]) == 1
    assert np.all(contributions[1][1] > contributions[1][0] * 3)
    assert np.all(contributions[1][1] > contributions[1][2:] * 3)

    assert np.argmax(contributions[2]) == 2
    assert np.all(contributions[2][2] > contributions[2][0:2] * 3)
    assert np.all(contributions[2][2] > contributions[2][3:] * 3)

    # In the fourth sample, there are 2 anomalous variables. Therefore, the contribution of these 2 variables should be
    # "significantly" higher than the contribution of the other variables. The contribution of both anomalous variables
    # should be equal (approximately).
    assert np.all(contributions[3][1] > contributions[3][0] * 3)
    assert np.all(contributions[3][2] > contributions[3][0] * 3)
    assert np.all(contributions[3][1] > contributions[3][3] * 3)
    assert np.all(contributions[3][2] > contributions[3][3] * 3)
    assert contributions[3][1] == approx(contributions[3][2], abs=0.5)

    assert np.all(contributions[4][0] > contributions[4][1] * 3)
    assert np.all(contributions[4][2] > contributions[4][1] * 3)
    assert np.all(contributions[4][3] > contributions[4][1] * 3)
    assert contributions[4][0] == approx(contributions[4][2], abs=0.5)
    assert contributions[4][2] == approx(contributions[4][3], abs=0.5)


@flaky(max_runs=3)
def test_given_simple_gaussian_data_when_attribute_anomaly_scores_with_feature_relevance_then_returns_qualitatively_correct_results():
    X0 = np.random.normal(0, 1, 5000)
    X1 = np.random.normal(0, 1, 5000)
    X2 = np.random.normal(0, 1, 5000)
    X3 = np.random.normal(0, 1, 5000)

    original_observations = np.column_stack([X0, X1, X2, X3])

    # Defining an anomaly scorer that handles multidimensional inputs.
    density_estimator = GaussianMixtureDensityEstimator(num_components=5)
    anomaly_scorer = InverseDensityScorer(density_estimator)
    anomaly_scorer.fit(original_observations)
    expectation_of_score = np.mean(anomaly_scorer.score(original_observations))

    # Seeing that the expectation of the noise in all nodes is 0, we introduce anomalies by setting some of them to 3.
    anomaly_samples = np.array([[3, 0, 0, 0], [0, 3, 3, 0], [3, 0, 3, 3]])

    contributions = attribute_anomaly_scores(anomaly_samples, original_observations, anomaly_scorer.score, True)

    assert np.argmax(contributions[0]) == 0  # The biggest contribution to the score should be for the first variable.
    assert contributions[0][0] > 0  # The contribution should be positive (anomalous variable).
    assert np.all(contributions[0][1:] < 0)  # Contributions should be negative (non-anomalous variables), since they
    # reduce the score.
    # The contributions should add up to g(x) - E[g(X)]
    assert np.sum(contributions[0]) == approx(
        anomaly_scorer.score(anomaly_samples[0].reshape(1, -1)) - expectation_of_score
    )

    assert contributions[1][1] > 0  # The contribution should be positive (anomalous variable).
    assert contributions[1][2] > 0  # The contribution should be positive (anomalous variable).
    assert contributions[1][0] < 0  # The contribution should be negative (non-anomalous variable).
    assert contributions[1][3] < 0  # The contribution should be negative (non-anomalous variable).
    # The contributions should add up to g(x) - E[g(X)]
    assert np.sum(contributions[1]) == approx(
        anomaly_scorer.score(anomaly_samples[1].reshape(1, -1)) - expectation_of_score
    )

    assert contributions[2][0] > 0  # The contribution should be positive (anomalous variable).
    assert contributions[2][2] > 0  # The contribution should be positive (anomalous variable).
    assert contributions[2][3] > 0  # The contribution should be positive (anomalous variable).
    assert contributions[2][1] < 0  # The contribution should be negative (non-anomalous variable).
    # The contributions should add up to g(x) - E[g(X)]
    assert np.sum(contributions[2]) == approx(
        anomaly_scorer.score(anomaly_samples[2].reshape(1, -1)) - expectation_of_score
    )


@flaky(max_runs=3)
def test_given_simple_causal_chain_with_linear_relationships_when_attribute_anomaly_scores_with_it_score_then_returns_qualitatively_correct_results():
    num_training_samples = 5000
    X0 = np.random.normal(0, 1, num_training_samples)
    X1 = X0 + np.random.normal(0, 1, num_training_samples)
    X2 = X1 + np.random.normal(0, 1, num_training_samples)
    X3 = X2 + np.random.normal(0, 1, num_training_samples)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    for node in ["X1", "X2", "X3"]:
        causal_model.set_causal_mechanism(node, AdditiveNoiseModel(create_linear_regressor()))

    fit(causal_model, training_data)

    # Three examples:
    # 1. X1 is the root cause (+ 10 to the noise)
    # 2. X0 is the root cause (+ 10 to the noise)
    # 3. X0 and X3 are both root causes (+ 10 to both noises)
    anomaly_data = pd.DataFrame(
        {
            "X0": np.array([0, 10, 10]),
            "X1": np.array([10, 10, 10]),
            "X2": np.array([10, 10, 10]),
            "X3": np.array([10, 10, 20]),
        }
    )

    scores, expected_total = _attribute_anomalies_with_expected_total(
        causal_model, anomaly_data, num_training_samples, attribute_mean_deviation=False
    )

    assert scores["X0"][0] < 0.5  # Not anomalous.
    assert scores["X0"][1] > 8  # Anomalous, but the only root cause.
    assert scores["X0"][2] > 4  # Anomalous together with X1.
    assert scores["X1"][0] > 8  # Anomalous, but the only root cause.
    assert scores["X1"][1] < 0.5  # Not anomalous given upstream nodes.
    assert scores["X1"][2] < 0.5  # Not anomalous given upstream nodes.
    assert np.all(scores["X2"] < 0.5)  # Not anomalous given upstream nodes.
    assert np.all(scores["X3"][:2] < 0.5)  # Not anomalous given upstream nodes.
    assert scores["X3"][2] > 4  # Anomalous together with X0.

    np.testing.assert_allclose(np.sum(list(scores.values()), axis=0), expected_total, rtol=0, atol=1e-10)


@flaky(max_runs=3)
def test_given_simple_causal_chain_with_linear_relationships_when_attribute_anomaly_scores_with_feature_relevance_then_returns_qualitatively_correct_results():
    num_training_samples = 5000
    X0 = np.random.normal(0, 1, num_training_samples)
    X1 = X0 + np.random.normal(0, 1, num_training_samples)
    X2 = X1 + np.random.normal(0, 1, num_training_samples)
    X3 = X2 + np.random.normal(0, 1, num_training_samples)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3})

    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X0", EmpiricalDistribution())
    for node in ["X1", "X2", "X3"]:
        causal_model.set_causal_mechanism(node, AdditiveNoiseModel(create_linear_regressor()))

    fit(causal_model, training_data)

    # Three examples:
    # 1. X1 is the root cause (+ 10 to the noise)
    # 2. X0 is the root cause (+ 10 to the noise)
    # 3. X0 and X3 are both root causes (+ 10 to both noises)
    anomaly_data = pd.DataFrame(
        {
            "X0": np.array([0, 10, 10]),
            "X1": np.array([10, 10, 10]),
            "X2": np.array([10, 10, 10]),
            "X3": np.array([10, 10, 20]),
        }
    )

    scores, expected_total = _attribute_anomalies_with_expected_total(
        causal_model, anomaly_data, num_training_samples, attribute_mean_deviation=True
    )

    assert scores["X0"][0] == approx(0, abs=0.1)  # Not anomalous.
    assert scores["X0"][1] > 0.4  # Anomalous, but the only root cause.
    assert scores["X0"][2] > 0.2  # Anomalous together with X1.
    assert scores["X1"][0] > 0.4  # Anomalous, but the only root cause.
    assert scores["X1"][1] == approx(0, abs=0.1)  # Not anomalous given upstream nodes.
    assert scores["X1"][2] == approx(0, abs=0.1)  # Not anomalous given upstream nodes.
    assert scores["X2"] == approx(np.array([0, 0, 0]), abs=0.1)  # Not anomalous given upstream nodes.
    assert scores["X3"][:2] == approx(np.array([0, 0]), abs=0.1)  # Not anomalous given upstream nodes.
    assert scores["X3"][2] > 0.2  # Anomalous together with X0.

    np.testing.assert_allclose(np.sum(list(scores.values()), axis=0), expected_total, rtol=0, atol=1e-10)


@flaky(max_runs=3)
def test_given_non_trivial_graph_with_nonlinear_relationships_when_attribute_anomaly_scores_with_it_score_then_returns_qualitatively_correct_results():
    num_training_samples = 10000

    # Defining ground truth model to avoid SCM learning issues and, hence, to focus on the anomaly attribution
    # algorithm.
    class MyNonLinearModel(PredictionModel):
        def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
            pass

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (np.sum(X, axis=1) ** 2).reshape(-1, 1)

        def clone(self):
            pass

    X0 = np.random.normal(0, 1, num_training_samples)
    X1 = MyNonLinearModel().predict(X0.reshape(-1, 1)) + np.random.normal(0, 1, (num_training_samples, 1))  # X0**2 + N
    X2 = np.random.uniform(-5, 5, num_training_samples)
    X3 = MyNonLinearModel().predict(np.column_stack([X1, X2])) + np.random.normal(
        0, 1, (num_training_samples, 1)
    )  # (X1 + X2)**2 + N

    training_data = pd.DataFrame({"X0": X0, "X1": X1.reshape(-1), "X2": X2, "X3": X3.reshape(-1)})
    causal_model = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X3"), ("X2", "X3")]))
    causal_model.set_causal_mechanism("X1", AdditiveNoiseModel(MyNonLinearModel()))
    causal_model.set_causal_mechanism("X3", AdditiveNoiseModel(MyNonLinearModel()))
    auto.assign_causal_mechanisms(causal_model, training_data, auto.AssignmentQuality.GOOD, override_models=False)

    fit(causal_model, training_data)

    anomaly_data = pd.DataFrame(
        {
            "X0": np.array([10, 10, 0]),
            "X1": np.array([100, 110, 0]),
            "X2": np.array([0, 0, 10]),
            "X3": np.array([10000, 12100, 110]),
        }
    )

    scores, expected_total = _attribute_anomalies_with_expected_total(
        causal_model, anomaly_data, num_training_samples, attribute_mean_deviation=False
    )

    # 1. X0 is the root cause (+ 10 to the noise)
    # 2. X0 and X1 are the root causes (+ 10 to both noise)
    # 3. X2 and X3 are both root causes (+ 10 to both noises)
    assert scores["X0"][0] > 8  # Anomalous, but the only root cause.
    assert scores["X0"][1] > 4  # Anomalous, but together with X1.
    assert scores["X0"][2] < 1  # Not anomalous.

    assert scores["X1"][0] < 1  # Not anomalous given upstream nodes.
    assert scores["X1"][1] > 1  # Anomalous, but together with X0.
    assert scores["X1"][2] < 1  # Not anomalous given upstream nodes.

    assert scores["X2"][0] < 1  # Not anomalous.
    assert scores["X2"][1] < 1  # Not anomalous.
    assert scores["X2"][2] > 1  # Anomalous, but together with X3.

    assert scores["X3"][0] < 1  # Not anomalous given upstream nodes.
    assert scores["X3"][1] < 1  # Not anomalous given upstream nodes.
    assert scores["X3"][2] > 1  # Anomalous, but together with X1.

    np.testing.assert_allclose(np.sum(list(scores.values()), axis=0), expected_total, rtol=0, atol=1e-10)


def test_relative_frequency():
    assert np.abs(_relative_frequency(np.array([True, True, False, True])) - 4 / 5) < 0.1
