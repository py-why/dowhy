import pytest

from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from .base import TestEstimator


class TestPropensityScoreMatchingEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.01, PropensityScoreWeightingEstimator),
                              (0.05, PropensityScoreWeightingEstimator)])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_test()
