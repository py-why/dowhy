import pytest
import statsmodels.api as sm
from pytest import mark

from dowhy.causal_estimators.generalized_linear_model_estimator import GeneralizedLinearModelEstimator

from .base import TestEstimator


@mark.usefixtures("fixed_seed")
class TestGeneralizedLinearModelEstimator(object):
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
            "identifier_method",
        ],
        [
            (
                0.1,
                GeneralizedLinearModelEstimator,
                [
                    0,
                ],
                [
                    0,
                ],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    False,
                ],
                [
                    True,
                ],
                "backdoor",
            )
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
        identifier_method,
    ):
        estimator_tester = TestEstimator(error_tolerance, Estimator, identifier_method)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={
                "num_simulations": 10,
                "num_null_simulations": 10,
                "glm_family": sm.families.Binomial(),
                "predict_score": True,
            },
        )
