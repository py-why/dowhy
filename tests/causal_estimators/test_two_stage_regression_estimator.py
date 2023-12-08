import numpy as np
import pandas as pd
import pytest
from pytest import mark

from dowhy import CausalModel
from dowhy.causal_estimators.two_stage_regression_estimator import TwoStageRegressionEstimator

from .base import SimpleEstimator


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
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method="frontdoor")
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

    def test_frontdoor_estimator(self):
        """
        Test for frontdoor estimation, from @AlxndrMlk
        See issue #616 https://github.com/py-why/dowhy/issues/616
        """

        # Create the graph describing the causal structure
        graph = """
        graph [
            directed 1
            
            node [
                id "X" 
                label "X"
            ]    
            node [
                id "Z"
                label "Z"
            ]
            node [
                id "Y"
                label "Y"
            ]
            node [
                id "U"
                label "U"
            ]
            
            edge [
                source "X"
                target "Z"
            ]
            
            edge [
                source "Z"
                target "Y"
            ]
            
            edge [
                source "U"
                target "Y"
            ]
            
            edge [
                source "U"
                target "X"
            ]
        ]
        """.replace(
            "\n", ""
        )

        N_SAMPLES = 10000
        # Generate the data
        U = np.random.randn(N_SAMPLES)
        X = np.random.randn(N_SAMPLES) + 0.3 * U
        Z = 0.7 * X + 0.3 * np.random.randn(N_SAMPLES)
        Y = 0.65 * Z + 0.2 * U

        # Data to df
        df = pd.DataFrame(np.vstack([X, Z, Y]).T, columns=["X", "Z", "Y"])

        # Create a model
        model = CausalModel(data=df, treatment="X", outcome="Y", graph=graph)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Estimate the effect with front-door
        estimate = model.estimate_effect(identified_estimand=estimand, method_name="frontdoor.two_stage_regression")
        assert estimate.value == pytest.approx(0.45, 0.025)

    @mark.parametrize(
        [
            "Estimator",
            "num_treatments",
            "num_frontdoor_variables",
        ],
        [
            (
                TwoStageRegressionEstimator,
                [2, 1],
                [1, 2],
            )
        ],
    )
    def test_frontdoor_num_variables_error(self, Estimator, num_treatments, num_frontdoor_variables):
        estimator_tester = SimpleEstimator(error_tolerance=0, Estimator=Estimator, identifier_method="frontdoor")
        with pytest.raises((ValueError, Exception)):
            estimator_tester.average_treatment_effect_testsuite(
                num_common_causes=[1, 1],
                num_instruments=[0, 0],
                num_effect_modifiers=[0, 0],
                num_treatments=num_treatments,
                num_frontdoor_variables=num_frontdoor_variables,
                treatment_is_binary=[True],
                outcome_is_binary=[False],
                confidence_intervals=[
                    True,
                ],
                test_significance=[
                    False,
                ],
                method_params={"num_simulations": 10, "num_null_simulations": 10},
            )
