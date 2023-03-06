import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx
from scipy import stats

from dowhy.gcm import (
    AdditiveNoiseModel,
    ClassifierFCM,
    ScipyDistribution,
    StructuralCausalModel,
    auto,
    confidence_intervals,
    fit,
)
from dowhy.gcm.feature_relevance import feature_relevance_distribution, feature_relevance_sample, parent_relevance
from dowhy.gcm.ml import create_linear_regressor, create_logistic_regression_classifier
from dowhy.gcm.shapley import ShapleyApproximationMethods, ShapleyConfig
from dowhy.gcm.uncertainty import estimate_entropy_of_probabilities
from dowhy.gcm.util.general import means_difference


@flaky(max_runs=5)
def test_when_using_parent_relevance_with_continous_data_then_returns_correct_results():
    causal_model = StructuralCausalModel(nx.DiGraph([("X1", "X2"), ("X0", "X2")]))
    causal_model.set_causal_mechanism("X1", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.norm, loc=0, scale=1))
    causal_model.set_causal_mechanism("X2", AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    X0 = np.random.normal(0, 1, 1000)
    X1 = np.random.normal(0, 1, 1000)
    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": 3 * X0 + X1})

    fit(causal_model, training_data)
    relevance, noise = parent_relevance(causal_model, "X2")

    # Contributions should add up to Var(X2)
    assert relevance[("X0", "X2")] + relevance[("X1", "X2")] + noise == approx(
        np.var(training_data["X2"].to_numpy()), abs=1.5
    )
    assert relevance[("X0", "X2")] == approx(9, abs=1)
    assert relevance[("X1", "X2")] == approx(1, abs=0.3)
    assert noise == approx(0, abs=0.5)


@flaky(max_runs=5)
def test_when_using_parent_relevance_with_categorical_data_then_returns_correct_results():
    causal_model = StructuralCausalModel(nx.DiGraph([("X0", "Y"), ("X1", "Y"), ("X2", "Y"), ("X3", "Y"), ("X4", "Y")]))
    causal_model.set_causal_mechanism("X0", ScipyDistribution(stats.uniform, loc=0, scale=1))
    causal_model.set_causal_mechanism("X1", ScipyDistribution(stats.uniform, loc=0, scale=1))
    causal_model.set_causal_mechanism("X2", ScipyDistribution(stats.uniform, loc=0, scale=1))
    causal_model.set_causal_mechanism("X3", ScipyDistribution(stats.uniform, loc=0, scale=1))
    causal_model.set_causal_mechanism("X4", ScipyDistribution(stats.uniform, loc=0, scale=1))
    causal_model.set_causal_mechanism("Y", ClassifierFCM())

    X = np.random.uniform(0, 1, (1000, 5))
    Y = []

    for n in X:
        if n[0] + n[1] > 1:
            Y.append(0)
        else:
            Y.append(1)

    X0 = X[:, 0]
    X1 = X[:, 1]
    X2 = X[:, 2]
    X3 = X[:, 3]
    X4 = X[:, 4]
    Y = np.array(Y).astype(str)

    fit(causal_model, data=pd.DataFrame({"X0": X0, "X1": X1, "X2": X2, "X3": X3, "X4": X4, "Y": Y}))

    relevance, noise = parent_relevance(causal_model, "Y", num_samples_randomization=1000, num_samples_baseline=100)

    assert relevance[("X0", "Y")] == approx(0.125, abs=0.05)
    assert relevance[("X1", "Y")] == approx(0.125, abs=0.05)
    assert relevance[("X2", "Y")] == approx(0, abs=0.05)
    assert relevance[("X3", "Y")] == approx(0, abs=0.05)
    assert relevance[("X4", "Y")] == approx(0, abs=0.05)
    assert noise == approx(0, abs=0.05)


@flaky(max_runs=3)
def test_when_given_linear_data_when_estimate_feature_relevance_per_sample_with_mean_diff_then_returns_expected_values():
    num_vars = 15
    X = np.random.normal(0, 1, (1000, num_vars))
    coefficients = np.random.choice(20, num_vars) - 10

    def model(x) -> np.ndarray:
        return np.sum(coefficients * x, axis=1)

    shapley_values = feature_relevance_sample(
        model, feature_samples=X, baseline_samples=X[:20], subset_scoring_func=means_difference
    )

    for i in range(20):
        assert coefficients * (X[i, :] - np.mean(X, axis=0)) == approx(shapley_values[i], abs=0.001)


@flaky(max_runs=3)
def test_given_baseline_values_when_estimating_feature_relevance_sample_with_mean_diff_then_returns_expected_result():
    num_vars = 5
    X = np.random.normal(0, 1, (1000, num_vars))
    coefficients = np.random.choice(20, num_vars) - 10

    def model(x) -> np.ndarray:
        return np.sum(coefficients * x, axis=1)

    shapley_values_1 = feature_relevance_sample(
        model,
        feature_samples=X,
        baseline_samples=X[:20],
        subset_scoring_func=lambda x, y: np.mean(x) * y,
        baseline_target_values=np.zeros(20),
        shapley_config=ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT),
    )

    shapley_values_2 = feature_relevance_sample(
        model,
        feature_samples=X,
        baseline_samples=X[:20],
        subset_scoring_func=lambda x, y: np.mean(x) * y,
        baseline_target_values=np.zeros(20) + 1,
        shapley_config=ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT),
    )

    shapley_values_3 = feature_relevance_sample(
        model,
        feature_samples=X,
        baseline_samples=X[:20],
        subset_scoring_func=lambda x, y: np.mean(x) * y,
        baseline_target_values=np.zeros(20) + 2,
        shapley_config=ShapleyConfig(approximation_method=ShapleyApproximationMethods.EXACT),
    )

    for i in range(20):
        assert shapley_values_1[i] == approx(np.zeros(5))
        assert shapley_values_2[i] * 2 == approx(shapley_values_3[i])


@flaky(max_runs=5)
def test_given_specific_batch_size_when_estimate_feature_relevance_per_sample_then_returns_expected_results():
    X = np.random.normal(0, 1, (1000, 3))
    coefficients = np.random.choice(20, 3) - 10

    def model(x) -> np.ndarray:
        return np.sum(coefficients * x, axis=1)

    shapley_values = feature_relevance_sample(
        model, feature_samples=X, baseline_samples=X, subset_scoring_func=means_difference, max_batch_size=123
    )

    for i in range(1000):
        assert coefficients * (X[i, :] - np.mean(X, axis=0)) == approx(shapley_values[i], abs=0.001)


@flaky(max_runs=3)
def test_when_using_feature_relevance_distribution_with_entropy_set_function_then_returns_correct_results():
    X = np.random.uniform(0, 1, (3000, 5))
    Y = []

    for n in X:
        if n[0] + n[1] > 1:
            Y.append(0)
        else:
            Y.append(1)

    Y = np.array(Y).astype(str)

    classifier_mdl = create_logistic_regression_classifier()
    classifier_mdl.fit(X, Y)

    # H(P(Y)) -- Can be precomputed
    h_p_Y = estimate_entropy_of_probabilities(np.mean(classifier_mdl.predict_probabilities(X), axis=0).reshape(1, -1))

    # -(H(P(Y | do(x_S)) - H(P(Y))) = H(P(Y)) - H(P(Y | do(x_S))
    def my_entropy_feature_attribution_function(
        randomized_predictions: np.ndarray, baseline_predictions: np.ndarray
    ) -> float:
        # H(P(Y | do(x_S)) = H(E[P(Y | x_S, X'_\S)])

        # E[P(Y | x_S, X'_\S)]
        p_y_do_xs = np.mean(randomized_predictions, axis=0).reshape(1, -1)

        # H(E[P(Y | x_S, X'_\S)])
        h_p_Y_do_xs = estimate_entropy_of_probabilities(p_y_do_xs)

        # Using H(P(Y)) based on the origina data, i.e. ignoring baseline_predictions.
        return h_p_Y - h_p_Y_do_xs

    feature_contributions = feature_relevance_distribution(
        classifier_mdl.predict_probabilities, X, my_entropy_feature_attribution_function
    )

    # E[H(P(Y)) - H(P(Y | do(X_U))] = H(P(Y)) - E[H(P(Y | X))]
    expected_sum_shapley_values = h_p_Y - estimate_entropy_of_probabilities(classifier_mdl.predict_probabilities(X))

    assert feature_contributions[0] == approx(expected_sum_shapley_values / 2, abs=0.075)
    assert feature_contributions[1] == approx(expected_sum_shapley_values / 2, abs=0.075)
    assert feature_contributions[2] == approx(0, abs=0.01)
    assert feature_contributions[3] == approx(0, abs=0.01)
    assert feature_contributions[4] == approx(0, abs=0.01)

    assert np.sum(feature_contributions) == approx(expected_sum_shapley_values, abs=0.04)


@flaky(max_runs=3)
def test_given_misspecified_graph_when_estimating_parent_relevance_with_observed_data_then_returns_correct_result():
    causal_model_without = StructuralCausalModel(nx.DiGraph([("X0", "X2"), ("X1", "X2")]))
    causal_model_with = StructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X0", "X2"), ("X1", "X2")]))

    X0 = np.random.normal(0, 1, 1000)
    X1 = X0 + np.random.normal(0, 1, 1000)

    training_data = pd.DataFrame({"X0": X0, "X1": X1, "X2": X0 + X1})
    auto.assign_causal_mechanisms(causal_model_without, training_data, auto.AssignmentQuality.GOOD)
    auto.assign_causal_mechanisms(causal_model_with, training_data, auto.AssignmentQuality.GOOD)

    fit(causal_model_without, training_data)
    fit(causal_model_with, training_data)

    feature_relevance_missing_edge, _ = parent_relevance(causal_model_without, "X2", training_data)
    feature_relevance_with_edge, _ = parent_relevance(causal_model_with, "X2")

    assert feature_relevance_missing_edge[("X0", "X2")] == approx(feature_relevance_with_edge[("X0", "X2")], abs=0.15)
    assert feature_relevance_missing_edge[("X1", "X2")] == approx(feature_relevance_with_edge[("X1", "X2")], abs=0.15)
