import numpy as np
from pytest import approx

from dowhy.gcm import MedianCDFQuantileScorer, MedianDeviationScorer


def test_given_simple_toy_data_when_using_MedianCDFQuantileScorer_then_returns_expected_scores():
    anomaly_scorer = MedianCDFQuantileScorer()
    anomaly_scorer.fit(np.array(range(0, 20)))
    assert anomaly_scorer.score(np.array([8, 17]))[0] == approx(1 - 17 / 20, abs=0.01)
    assert anomaly_scorer.score(np.array([0])) == approx(1 - 1 / 20, abs=0.01)

    anomaly_scorer.fit(np.array([4.0 for i in range(200)]))
    assert anomaly_scorer.score(np.array([4])) == approx(0, abs=0.01)


def test_given_simple_toy_data_when_using_MedianDeviationScorer_then_returns_expected_scores():
    anomaly_scorer = MedianDeviationScorer()
    anomaly_scorer.fit(np.array(range(0, 20)) / 10)
    assert anomaly_scorer.score(np.array([0.8, 1.7])).reshape(-1) == approx(np.array([0.2, 1]), abs=0.1)
