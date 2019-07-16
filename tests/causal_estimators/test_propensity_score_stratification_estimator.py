import pytest

from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator
from .base import TestEstimator


class TestPropensityScoreStratificationEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.01, PropensityScoreStratificationEstimator),
                              (0.05, PropensityScoreStratificationEstimator)])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_testsuite(tests_to_run="atleast-one-common-cause")
