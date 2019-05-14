from dowhy.causal_estimators.propensity_score_matching_estimator import PropensityScoreMatchingEstimator
import pytest

from .base import TestEstimator


class TestPropensityScoreMatchingEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.05, PropensityScoreMatchingEstimator),])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_testsuite()
