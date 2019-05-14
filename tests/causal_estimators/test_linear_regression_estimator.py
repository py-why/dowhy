import pytest

from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from .base import TestEstimator


class TestLinearRegressionEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.05, LinearRegressionEstimator),])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_testsuite()
