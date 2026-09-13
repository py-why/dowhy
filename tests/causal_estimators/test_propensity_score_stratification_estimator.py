from pytest import mark

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


def test_pss_non_zero_one_treatment_encoding():
    """Regression test: PSS must use treatment_value/control_value, not hardcoded 0/1.

    When treatment is encoded as {1, 2}, the old code computed ``1 - treatment``
    (which gives -1 for treated units) and used ``== 1`` / ``== 0`` to count
    treated/control per stratum, producing wrong results.
    """
    import numpy as np
    import pandas as pd

    from dowhy import CausalModel

    rng = np.random.default_rng(42)
    n = 1000
    X = rng.standard_normal(n)
    T = rng.choice([1, 2], size=n)
    Y = 3.0 * (T == 2) + 0.5 * X + rng.standard_normal(n)
    df = pd.DataFrame({"X": X, "T": T, "Y": Y})

    model = CausalModel(data=df, treatment=["T"], outcome="Y", common_causes=["X"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.propensity_score_stratification", treatment_value=2, control_value=1
    )

    assert np.isfinite(estimate.value)
    assert abs(estimate.value - 3.0) < 1.5, f"Expected estimate near 3.0, got {estimate.value}"


def test_pss_does_not_mutate_input_dataframe():
    """_get_strata must not add internal __dowhy_*__ columns to the caller's DataFrame."""
    import numpy as np
    import pandas as pd

    from dowhy import CausalModel

    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal(n)
    T = rng.choice([0, 1], size=n)
    Y = 2.0 * T + 0.5 * X + rng.standard_normal(n)
    df = pd.DataFrame({"X": X, "T": T, "Y": Y})
    cols_before = set(df.columns)

    model = CausalModel(data=df, treatment=["T"], outcome="Y", common_causes=["X"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    model.estimate_effect(estimand, method_name="backdoor.propensity_score_stratification")

    internal_cols = {"strata", "dbar", "d_y", "dbar_y"} | {c for c in df.columns if c.startswith("__dowhy_")}
    leaked = internal_cols & set(df.columns)
    assert not leaked, f"_get_strata leaked internal columns: {leaked}"
