import pytest
import numpy as np

import dowhy.api
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from .base import TestEstimator

from sklearn.linear_model import LinearRegression

@pytest.mark.usefixtures("fixed_seed")
class TestPropensityScoreWeightingEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator",
        "num_common_causes", "num_instruments",
        "num_effect_modifiers", "num_treatments",
        "treatment_is_binary", "outcome_is_binary"],
                             [(0.4, PropensityScoreWeightingEstimator, [1,2], [0], [0,], [1,], [True,], [False,]),])
    def test_average_treatment_effect(self, error_tolerance, Estimator,
            num_common_causes, num_instruments, num_effect_modifiers,
            num_treatments, treatment_is_binary, outcome_is_binary
            ):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_testsuite(
                num_common_causes=num_common_causes,
                num_instruments = num_instruments,
                num_effect_modifiers = num_effect_modifiers,
                num_treatments=num_treatments,
                treatment_is_binary=treatment_is_binary,
                outcome_is_binary=outcome_is_binary,
                confidence_intervals=[True,],
                test_significance=[True,],
                method_params={
                    'num_ci_simulations': 10,
                    'num_null_simulations': 10
                    }
                )
