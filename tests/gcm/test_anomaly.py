import operator

import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky
from pytest import approx

from dowhy.gcm import auto, AdditiveNoiseModel, fit, anomaly_scores, \
    MedianCDFQuantileScorer, ITAnomalyScorer, RescaledMedianCDFQuantileScorer, \
    MeanDeviationScorer, MedianDeviationScorer, InverseDensityScorer, EmpiricalDistribution, \
    ProbabilisticCausalModel
from dowhy.gcm.anomaly import conditional_anomaly_score
from dowhy.gcm.constant import EPS
from dowhy.gcm.distribution_change import estimate_distribution_change_scores
from dowhy.gcm.ml import create_linear_regressor


def _create_scm_for_distribution_change():
    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    return causal_model


def test_estimate_anomaly_scores_using_median_cdf_quantile_anomaly_scoring():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})
    outlier_observations = pd.DataFrame({'X0': X0[:1],
                                         'X1': X1[:1] + np.std(X1.data) * 3,
                                         'X2': X2[:1],
                                         'X3': X3[:1]})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(causal_model, original_observations)

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=MedianCDFQuantileScorer).items(),
               key=operator.itemgetter(1))[0] == 'X1'

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=
                              lambda: ITAnomalyScorer(MedianCDFQuantileScorer())).items(),
               key=operator.itemgetter(1))[0] == 'X1'


def test_estimate_anomaly_scores_using_rescaled_median_cdf_quantile_scores():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})
    outlier_observations = pd.DataFrame({'X0': X0[:1],
                                         'X1': X1[:1] + np.std(X1.data) * 3,
                                         'X2': X2[:1],
                                         'X3': X3[:1]})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(causal_model, original_observations)

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=RescaledMedianCDFQuantileScorer).items(),
               key=operator.itemgetter(1))[0] == 'X1'

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=
                              lambda: ITAnomalyScorer(RescaledMedianCDFQuantileScorer())).items(),
               key=operator.itemgetter(1))[0] == 'X1'


def test_estimate_anomaly_scores_using_mean_deviation_anomaly_scoring_function():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})
    outlier_observations = pd.DataFrame({'X0': X0[:1],
                                         'X1': X1[:1] + np.std(X1.data) * 3,
                                         'X2': X2[:1],
                                         'X3': X3[:1]})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(causal_model, original_observations)

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=lambda: MeanDeviationScorer())
               .items(),
               key=operator.itemgetter(1))[0] == 'X1'

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=lambda: ITAnomalyScorer(MeanDeviationScorer())).items(),
               key=operator.itemgetter(1))[0] == 'X1'


def test_estimate_anomaly_scores_using_median_deviation_anomaly_scoring_function():
    print(np.random.get_state())

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})
    outlier_observations = pd.DataFrame({'X0': X0[:1],
                                         'X1': X1[:1] + np.std(X1.data) * 3,
                                         'X2': X2[:1],
                                         'X3': X3[:1]})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(causal_model, original_observations)

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=lambda: MedianDeviationScorer())
               .items(),
               key=operator.itemgetter(1))[0] == 'X1'

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=lambda: ITAnomalyScorer(MedianDeviationScorer())).items(),
               key=operator.itemgetter(1))[0] == 'X1'


def test_when_estimate_inverse_density_score_then_returns_expected_results():
    print(np.random.get_state())

    X0 = np.random.uniform(-1, 1, 500)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 500)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 500)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 500)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})
    outlier_observations = pd.DataFrame({'X0': X0[:1],
                                         'X1': X1[:1] + np.std(X1.data) * 3,
                                         'X2': X2[:1],
                                         'X3': X3[:1]})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X1'), ('X0', 'X2'), ('X2', 'X3')]))
    causal_model.set_causal_mechanism('X0', EmpiricalDistribution())
    causal_model.set_causal_mechanism('X1', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X2', AdditiveNoiseModel(prediction_model=create_linear_regressor()))
    causal_model.set_causal_mechanism('X3', AdditiveNoiseModel(prediction_model=create_linear_regressor()))

    fit(causal_model, original_observations)

    assert max(anomaly_scores(causal_model,
                              outlier_observations,
                              anomaly_scorer_factory=lambda: InverseDensityScorer()).items(),
               key=operator.itemgetter(1))[0] == 'X1'


def test_estimate_distribution_change_scores():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0,
                                         'X1': X1,
                                         'X2': X2,
                                         'X3': X3})

    causal_model = _create_scm_for_distribution_change()
    fit(causal_model, original_observations)

    assert \
        max(estimate_distribution_change_scores(causal_model, original_observations, outlier_observations).items(),
            key=operator.itemgetter(1))[0] == 'X2'


def test_estimate_distribution_change_scores_with_expectation():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 2 + 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0,
                                         'X1': X1,
                                         'X2': X2,
                                         'X3': X3})

    causal_model = _create_scm_for_distribution_change()
    fit(causal_model, original_observations)

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.mean(x) - np.mean(y)))

    assert scores['X0'] == approx(0, abs=0.1)
    assert scores['X1'] == approx(0, abs=0.1)
    assert scores['X2'] == approx(2, abs=0.1)
    assert scores['X3'] == approx(0, abs=0.1)

    assert max(scores.items(), key=operator.itemgetter(1))[0] == 'X2'


def test_estimate_distribution_change_scores_with_expectation_root_node():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})

    # Here, changing the mechanism.
    X0 = np.random.uniform(-1.5, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0,
                                         'X1': X1,
                                         'X2': X2,
                                         'X3': X3})

    causal_model = _create_scm_for_distribution_change()
    fit(causal_model, original_observations)

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.mean(x) - np.mean(y)))

    assert scores['X0'] == approx(0.25, abs=0.1)
    assert scores['X1'] == approx(0, abs=0.1)
    assert scores['X2'] == approx(0, abs=0.1)
    assert scores['X3'] == approx(0, abs=0.1)

    assert max(scores.items(), key=operator.itemgetter(1))[0] == 'X0'


@flaky(max_runs=3)
def test_given_graph_with_multiple_parents_when_estimate_distribution_change_scores_then_returns_expected_result():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = np.random.uniform(-1, 1, 1000)
    X2 = X0 + X1

    original_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = np.random.uniform(-1, 1, 1000)
    X2 = X0 + X1 + 1
    outlier_observations = pd.DataFrame({'X0': X0, 'X1': X1, 'X2': X2})

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X0', 'X2'), ('X1', 'X2')]))
    auto.assign_causal_mechanisms(causal_model, original_observations, auto.AssignmentQuality.GOOD)

    fit(causal_model, original_observations)

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.mean(x) - np.mean(y)))

    assert scores['X0'] == 0
    assert scores['X1'] == 0
    assert scores['X2'] == approx(1, abs=0.005)


@flaky(max_runs=5)
def test_estimate_distribution_change_scores_with_variance():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 0.5 * X0 + np.random.normal(0, 2, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0,
                                         'X1': X1,
                                         'X2': X2,
                                         'X3': X3})

    causal_model = _create_scm_for_distribution_change()
    fit(causal_model, original_observations)

    scores = estimate_distribution_change_scores(
        causal_model,
        original_observations,
        outlier_observations,
        difference_estimation_func=lambda x, y: abs(np.var(x) - np.var(y)))

    assert scores['X0'] == approx(0, abs=0.1)
    assert scores['X1'] == approx(0, abs=0.1)
    assert scores['X2'] == approx(4, abs=0.5)
    assert scores['X3'] == approx(0, abs=0.1)

    assert max(scores.items(), key=operator.itemgetter(1))[0] == 'X2'


@flaky(max_runs=5)
def test_when_using_estimate_distribution_change_scores_without_fdrc_then_returns_valid_results():
    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X2 = 0.5 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)

    original_observations = pd.DataFrame({'X0': X0,
                                          'X1': X1,
                                          'X2': X2,
                                          'X3': X3})

    X0 = np.random.uniform(-1, 1, 1000)
    X1 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    # Here, changing the mechanism.
    X2 = 2 * X0 + np.random.normal(0, 0.1, 1000)
    X3 = 0.5 * X2 + np.random.normal(0, 0.1, 1000)
    outlier_observations = pd.DataFrame({'X0': X0,
                                         'X1': X1,
                                         'X2': X2,
                                         'X3': X3})

    causal_model = _create_scm_for_distribution_change()
    fit(causal_model, original_observations)

    assert max(estimate_distribution_change_scores(causal_model,
                                                   original_observations,
                                                   outlier_observations,
                                                   mechanism_change_test_fdr_control_method=None).items(),
               key=operator.itemgetter(1))[0] == 'X2'


@flaky(max_runs=5)
def test_estimate_anomaly_scores_does_not_raise_an_error_with_multivariate_inputs():
    """ This test verifies that estimate_anomaly_scores correctly handles multivariate input features, which
     caused problems in an earlier version. """

    causal_model = ProbabilisticCausalModel(nx.DiGraph([('X1', 'X0'), ('X2', 'X0'), ('X3', 'X0'), ('X4', 'X0')]))

    data = np.random.normal(0, 1, (10000, 4))
    data = pd.DataFrame({'X0': (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]).reshape(-1),
                         'X1': data[:, 0].reshape(-1),
                         'X2': data[:, 1].reshape(-1),
                         'X3': data[:, 2].reshape(-1),
                         'X4': data[:, 3].reshape(-1)})
    data_anomaly = pd.DataFrame({'X0': data.iloc[0:1, 0] * 10 + data.iloc[0:1, 1] + data.iloc[0:1, 2]
                                       + data.iloc[0:1, 3],
                                 'X1': data.iloc[0:1, 0] * 10,
                                 'X2': data.iloc[0:1, 1],
                                 'X3': data.iloc[0:1, 2],
                                 'X4': data.iloc[0:1, 3]})

    auto.assign_causal_mechanisms(causal_model, data, auto.AssignmentQuality.GOOD)

    fit(causal_model, data)
    scores = anomaly_scores(causal_model, data_anomaly)
    assert scores['X1'][0] == approx(-np.log(EPS), abs=3)


@flaky(max_runs=3)
def test_estimate_conditional_anomaly_score():
    X = np.random.normal(0, 1, 1000)
    N = np.random.normal(0, 1, 1000)

    Y = 2 * X + N

    causal_model = AdditiveNoiseModel(prediction_model=create_linear_regressor())
    causal_model.fit(X, Y)

    anomaly_scorer = MeanDeviationScorer()
    anomaly_scorer.fit(N)

    assert conditional_anomaly_score(X[:5], Y[:5], causal_model, MeanDeviationScorer).reshape(-1) \
           == approx(anomaly_scorer.score(Y[:5] - 2 * X[:5]), abs=0.1)
