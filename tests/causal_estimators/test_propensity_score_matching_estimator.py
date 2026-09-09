from pytest import mark

from dowhy.causal_estimators.propensity_score_matching_estimator import PropensityScoreMatchingEstimator

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreMatchingEstimator(object):
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
                0.3,
                PropensityScoreMatchingEstimator,
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
                0.3,
                PropensityScoreMatchingEstimator,
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
                False,
            ],
            test_significance=[
                False,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )


def test_psm_non_zero_one_treatment_encoding():
    """Regression test: PSM must use treatment_value/control_value, not hardcoded 0/1.

    When treatment is encoded as {1, 2} instead of {0, 1}, the old code misidentified
    treated/control groups (``data[T] == 1`` selected the control arm).
    """
    import numpy as np
    import pandas as pd

    from dowhy import CausalModel

    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal(n)
    T = rng.choice([1, 2], size=n)
    Y = 3.0 * (T == 2) + 0.5 * X + rng.standard_normal(n)
    df = pd.DataFrame({"X": X, "T": T, "Y": Y})

    model = CausalModel(data=df, treatment=["T"], outcome="Y", common_causes=["X"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.propensity_score_matching", treatment_value=2, control_value=1
    )

    assert np.isfinite(estimate.value)
    assert abs(estimate.value - 3.0) < 1.5, f"Expected estimate near 3.0, got {estimate.value}"


def test_psm_string_treatment_encoding():
    """PSM must accept string-valued binary treatment columns."""
    import numpy as np
    import pandas as pd

    from dowhy import CausalModel

    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal(n)
    T_int = rng.choice([0, 1], size=n)
    T = np.where(T_int == 1, "treated", "control")
    Y = 3.0 * T_int + 0.5 * X + rng.standard_normal(n)
    df = pd.DataFrame({"X": X, "T": T, "Y": Y})

    model = CausalModel(data=df, treatment=["T"], outcome="Y", common_causes=["X"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.propensity_score_matching", treatment_value="treated", control_value="control"
    )

    assert np.isfinite(estimate.value)
    assert abs(estimate.value - 3.0) < 1.5, f"Expected estimate near 3.0, got {estimate.value}"
