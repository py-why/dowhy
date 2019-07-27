import pytest
import numpy as np

import dowhy.api
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from .base import TestEstimator

from sklearn.linear_model import LinearRegression


class TestPropensityScoreWeightingEstimator(object):
    @pytest.mark.parametrize(["error_tolerance", "Estimator"],
                             [(0.4, PropensityScoreWeightingEstimator),
                              (0.5, PropensityScoreWeightingEstimator)])
    def test_average_treatment_effect(self, error_tolerance, Estimator):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_testsuite(tests_to_run="atleast-one-common-cause")
