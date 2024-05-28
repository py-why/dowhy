import numpy as np
from pytest import approx

from dowhy.gcm import MedianCDFQuantileScorer, MedianDeviationScorer, RescaledMedianCDFQuantileScorer
from dowhy.gcm.anomaly_scorers import RankBasedAnomalyScorer


def test_given_simple_toy_data_when_using_MedianCDFQuantileScorer_then_returns_expected_scores():
    anomaly_scorer = MedianCDFQuantileScorer()
    anomaly_scorer.fit(np.array(range(0, 20)))
    assert anomaly_scorer.score(np.array([8, 17]))[0] == approx(1 - 18 / 21, abs=0.01)
    assert anomaly_scorer.score(np.array([0])) == approx(1 - 2 / 21, abs=0.01)

    anomaly_scorer.fit(np.array([4.0 for i in range(200)]))
    assert anomaly_scorer.score(np.array([4])) == approx(0, abs=0.01)


def test_given_simple_toy_data_when_using_MedianDeviationScorer_then_returns_expected_scores():
    anomaly_scorer = MedianDeviationScorer()
    anomaly_scorer.fit(np.array(range(0, 20)) / 10)
    assert anomaly_scorer.score(np.array([0.8, 1.7])).reshape(-1) == approx(np.array([0.2, 1]), abs=0.1)


def test_given_data_with_nans_when_using_median_quantile_scorer_with_nan_support_then_returns_expected_scores():
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan])

    scorer = RescaledMedianCDFQuantileScorer()
    scorer.fit(training_data)

    assert scorer.score(np.array([1, 4, 8, np.nan])) == approx(
        [-np.log(2 * 1 / 11), -np.log(2 * 4 / 11), -np.log(2 * 1 / 11), -np.log(2 * 1.5 / 11)]
    )


def test_given_numpy_arrays_with_object_type_when_using_median_quantile_scorer_then_does_not_raise_error():
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan], dtype=object)

    scorer = RescaledMedianCDFQuantileScorer()
    scorer.fit(training_data)

    assert scorer.score(np.array([1, 4, 8, np.nan], dtype=object)) == approx(
        [-np.log(2 * 1 / 11), -np.log(2 * 4 / 11), -np.log(2 * 1 / 11), -np.log(2 * 1.5 / 11)]
    )


def test_when_using_ranked_based_anomaly_scorer_then_returns_expected_scores():
    training_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan])

    scorer = RankBasedAnomalyScorer()
    scorer.fit(training_data)

    assert scorer.score(np.array([1, 4, 9, np.nan])) == approx(
        [-np.log(2 * 2 / 11), -np.log(2 * 5 / 11), -np.log(2 * 1 / 11), -np.log(2 * 3 / 11)]
    )

    training_data = np.array([np.nan, np.nan, np.nan, np.nan])
    scorer.fit(training_data)

    assert scorer.score(np.array([1, np.nan])) == approx([-np.log(2 * 1 / 5), -np.log(1)])
