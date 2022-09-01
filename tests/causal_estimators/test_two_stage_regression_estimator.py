import pytest
from pytest import mark

from dowhy.causal_estimators.two_stage_regression_estimator import TwoStageRegressionEstimator

from .base import TestEstimator


@mark.usefixtures("fixed_seed")
class TestTwoStageRegressionEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "num_frontdoor_variables",
            "treatment_is_binary",
            "outcome_is_binary",
        ],
        [
            (
                0.1,
                TwoStageRegressionEstimator,
                [0],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    1,
                ],
                [False],
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
        num_frontdoor_variables,
        treatment_is_binary,
        outcome_is_binary,
    ):
        estimator_tester = TestEstimator(error_tolerance, Estimator, identifier_method="frontdoor")
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            num_frontdoor_variables=num_frontdoor_variables,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                False,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )
