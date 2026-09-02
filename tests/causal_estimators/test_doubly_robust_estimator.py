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


_GML_GRAPH = """
graph [
  directed 1
  node [id "X" label "X"]
  node [id "T" label "T"]
  node [id "Y" label "Y"]
  edge [source "X" target "T"]
  edge [source "X" target "Y"]
  edge [source "T" target "Y"]
]
"""


def _make_estimand(df):
    return identify_effect_auto(
        build_graph_from_str(_GML_GRAPH),
        observed_nodes=list(df.columns),
        action_nodes=["T"],
        outcome_nodes=["Y"],
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    )


def test_doubly_robust_treatment_value_zero():
    """Regression test: ATE should be symmetric whether coded as (T=1 treated, T=0 control)
    or (T=0 treated, T=1 control).  Before the fix, the propensity column index was wrong
    for treatment_value=0, yielding a grossly incorrect estimate."""
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal(n)
    # T=1 means "treated" in the natural coding
    T_natural = (rng.standard_normal(n) + X > 0).astype(int)
    # Flip: T=0 means "treated" in the flipped coding
    T_flipped = 1 - T_natural
    true_ate = 3.0
    Y_natural = true_ate * T_natural + X + rng.standard_normal(n)
    Y_flipped = true_ate * T_flipped + X + rng.standard_normal(n)

    df_natural = pd.DataFrame({"X": X, "T": T_natural, "Y": Y_natural})
    df_flipped = pd.DataFrame({"X": X, "T": T_flipped, "Y": Y_flipped})

    estimand_natural = _make_estimand(df_natural)
    estimand_natural.set_identifier_method("backdoor")
    est_natural = DoublyRobustEstimator(identified_estimand=estimand_natural)
    est_natural.fit(df_natural)
    result_natural = est_natural.estimate_effect(df_natural, control_value=0, treatment_value=1)

    estimand_flipped = _make_estimand(df_flipped)
    estimand_flipped.set_identifier_method("backdoor")
    est_flipped = DoublyRobustEstimator(identified_estimand=estimand_flipped)
    est_flipped.fit(df_flipped)
    # treatment_value=0 is the treated group; control_value=1 is control
    result_flipped = est_flipped.estimate_effect(df_flipped, control_value=1, treatment_value=0)

    # Both should recover ≈ true_ate (up to sign: natural gives +3, flipped gives -3)
    assert abs(result_natural.value - true_ate) < 0.5, f"natural ATE {result_natural.value:.3f} far from {true_ate}"
    assert abs(result_flipped.value + true_ate) < 0.5, f"flipped ATE {result_flipped.value:.3f} far from {-true_ate}"
