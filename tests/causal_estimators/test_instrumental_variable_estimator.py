import pytest

from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator
from .base import TestEstimator


class TestPropensityScoreMatchingEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.05, InstrumentalVariableEstimator),
                              (0.1, InstrumentalVariableEstimator)])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_test()
