import pytest
from pytest import mark

from dowhy.causal_estimators.propensity_score_matching_estimator import PropensityScoreMatchingEstimator

from .base import TestEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreMatchingEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "outcome_is_binary",
        ],
        [
            (
                0.3,
                PropensityScoreMatchingEstimator,
                [1, 2],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
            ),
        ],
    )
    def test_average_treatment_effect(
        self,
        error_tolerance,
        Estimator,
        num_common_causes,
        num_instruments,
        num_effect_modifiers,
        num_treatments,
        treatment_is_binary,
        outcome_is_binary,
    ):
        estimator_tester = TestEstimator(error_tolerance, Estimator)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                False,
            ],
            test_significance=[
                False,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )
