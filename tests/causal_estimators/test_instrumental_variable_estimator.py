import pytest

from dowhy.causal_estimators.instrumental_variable_estimator import InstrumentalVariableEstimator
from .base import TestEstimator


class TestInstrumentalVariableEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.1, InstrumentalVariableEstimator),
                              (0.2, InstrumentalVariableEstimator)])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_test_binary()
        estimator_tester.average_treatment_effect_test_continuous()
