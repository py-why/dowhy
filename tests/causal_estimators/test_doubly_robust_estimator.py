import numpy as np
import pandas as pd
import pytest
from pytest import mark

from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.doubly_robust_estimator import DoublyRobustEstimator
from dowhy.graph import build_graph_from_str

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

    def test_conditional_treatment_effect_with_effect_modifiers(self):
        """DoublyRobustEstimator should produce a per-group CATE when effect modifiers are present."""
        rng = np.random.default_rng(42)
        n = 2000

        # X: confounder, V: continuous effect modifier
        X = rng.standard_normal(n)
        V = rng.standard_normal(n)
        T = (rng.standard_normal(n) + X > 0).astype(int)
        # True effect heterogeneity: CATE = 5 + 2*V
        Y = 5 * T + 2 * V * T + X + rng.standard_normal(n)

        df = pd.DataFrame({"X": X, "V": V, "T": T, "Y": Y})

        gml_graph = """
        graph [
          directed 1
          node [id "X" label "X"]
          node [id "V" label "V"]
          node [id "T" label "T"]
          node [id "Y" label "Y"]
          edge [source "X" target "T"]
          edge [source "X" target "Y"]
          edge [source "T" target "Y"]
          edge [source "V" target "Y"]
        ]
        """
        target_estimand = identify_effect_auto(
            build_graph_from_str(gml_graph),
            observed_nodes=list(df.columns),
            action_nodes=["T"],
            outcome_nodes=["Y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")

        estimator = DoublyRobustEstimator(identified_estimand=target_estimand)
        estimator.fit(df, effect_modifier_names=["V"])

        result = estimator.estimate_effect(df, control_value=0, treatment_value=1, target_units="ate")

        # ATE should be close to E[5 + 2*V] = 5 (since E[V] = 0)
        assert abs(result.value - 5.0) < 0.5, f"ATE estimate {result.value:.3f} too far from 5.0"

        # Conditional estimates should be a non-empty Series
        assert result.conditional_estimates is not None
        assert len(result.conditional_estimates) > 0
