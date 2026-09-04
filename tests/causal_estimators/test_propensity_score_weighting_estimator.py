from pytest import mark

import numpy as np
import pandas as pd

from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator
from dowhy.graph import build_graph_from_str

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreWeightingEstimator(object):
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
                0.4,
                PropensityScoreWeightingEstimator,
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
                "backdoor",
            ),
            (
                0.4,
                PropensityScoreWeightingEstimator,
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
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={"num_simulations": 1, "num_null_simulations": 1},
        )


def test_psw_non_zero_one_treatment_encoding():
    """Regression test: PSW must use treatment_value/control_value indicators, not raw data[treatment].

    When treatment is encoded as {1, 2} instead of {0, 1}, the old implementation used
    ``data[T] / ps`` and ``1 - data[T]`` directly in weight formulas, which is only
    correct for {0, 1} encodings and gives wrong results for any other binary encoding.
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal(n)
    # Treatment encoded as {1, 2} (control=1, treated=2)
    ps = 1 / (1 + np.exp(-X))
    T = (rng.random(n) < ps).astype(int) + 1  # 1 = control, 2 = treated
    Y = 3.0 * (T == 2) + 0.5 * X + rng.standard_normal(n)

    df = pd.DataFrame({"X": X, "T": T, "Y": Y})

    gml = """graph [
        directed 1
        node [id "T" label "T"]
        node [id "X" label "X"]
        node [id "Y" label "Y"]
        edge [source "X" target "T"]
        edge [source "X" target "Y"]
        edge [source "T" target "Y"]
    ]"""
    graph = build_graph_from_str(gml)
    estimand = identify_effect_auto(
        graph,
        observed_nodes=["T", "X", "Y"],
        action_nodes=["T"],
        outcome_nodes=["Y"],
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    )

    estimator = PropensityScoreWeightingEstimator(identified_estimand=estimand)
    estimator.fit(df)
    estimate = estimator.estimate_effect(df, treatment_value=2, control_value=1, target_units="ate")

    assert np.isfinite(estimate.value), f"Expected a finite estimate, got {estimate.value}"
    assert abs(estimate.value - 3.0) < 1.0, f"ATE estimate {estimate.value} too far from true value 3.0"
