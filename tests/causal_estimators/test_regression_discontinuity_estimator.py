from pytest import mark

from dowhy.causal_estimators.regression_discontinuity_estimator import RegressionDiscontinuityEstimator

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestRegressionDiscontinuityEstimator(object):
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
                0.2,
                RegressionDiscontinuityEstimator,
                [1],
                [
                    1,
                ],
                [0],
                [
                    1,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "iv",
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
        identifier_method,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            dataset="simple-iv",
            method_params={"rd_variable_name": "Z0", "rd_threshold_value": 0.5, "rd_bandwidth": 0.2},
        )
