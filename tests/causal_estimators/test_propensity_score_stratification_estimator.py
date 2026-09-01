from pytest import mark

import numpy as np
import pandas as pd

from dowhy import CausalModel
from dowhy.causal_estimators.propensity_score_stratification_estimator import PropensityScoreStratificationEstimator

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreStratificationEstimator(object):
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
                PropensityScoreStratificationEstimator,
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
                0.1,
                PropensityScoreStratificationEstimator,
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
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )


def test_estimate_effect_does_not_mutate_input_dataframe():
    """_get_strata must not add internal columns (strata, dbar, d_y, dbar_y) to the
    caller's DataFrame.  Regression test for the mutation bug."""
    rng = np.random.default_rng(42)
    n = 500
    W = rng.standard_normal(n)
    T = (rng.standard_normal(n) + W > 0).astype(float)
    Y = 2.0 * T + W + rng.standard_normal(n)
    data = pd.DataFrame({"W": W, "T": T, "Y": Y})
    original_columns = set(data.columns)

    model = CausalModel(data=data, treatment="T", outcome="Y", common_causes=["W"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    model.estimate_effect(
        estimand,
        method_name="backdoor.propensity_score_stratification",
        target_units="ate",
    )

    internal_cols = {"strata", "dbar", "d_y", "dbar_y"}
    leaked = internal_cols & set(data.columns)
    assert not leaked, f"estimate_effect must not add internal columns to the caller's DataFrame; found: {leaked}"
    # Also verify no other unexpected columns were added (propensity_score is allowed as it is
    # the score computed in-place for efficiency and stored in CausalEstimate.propensity_scores).
    extra = set(data.columns) - original_columns - {"propensity_score"}
    assert not extra, f"Unexpected columns were added to the caller's DataFrame: {extra}"
