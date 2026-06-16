import pytest
from pytest import mark

from dowhy.causal_estimators.doubly_robust_estimator import DoublyRobustEstimator

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestDoublyRobustEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "treatment_is_category",
            "outcome_is_binary",
            "identifier_method",
        ],
        [
            (
                0.1,
                DoublyRobustEstimator,
                [1, 2],
                [0, 1],
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
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.2,
                DoublyRobustEstimator,
                [1, 2],
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
                    True,
                ],
                [
                    False,
                ],
                [
                    True,
                ],
                "general_adjustment",
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
        treatment_is_category,
        outcome_is_binary,
        identifier_method,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            treatment_is_category=treatment_is_category,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )

    def test_multiple_treatments_raises_value_error(self):
        estimator_tester = SimpleEstimator(error_tolerance=0.5, Estimator=DoublyRobustEstimator)
        with pytest.raises(ValueError, match="cannot handle more than one treatment variable"):
            estimator_tester.average_treatment_effect_testsuite(
                num_common_causes=[1],
                num_instruments=[0],
                num_effect_modifiers=[0],
                num_treatments=[2],
                treatment_is_binary=[True],
                outcome_is_binary=[False],
                confidence_intervals=[False],
                test_significance=[False],
                method_params={"num_simulations": 10, "num_null_simulations": 10},
            )
