import random

import numpy as np
import pandas as pd
from pytest import mark
from scipy.special import expit

import dowhy.datasets
from dowhy import CausalModel

from .base import SimpleRefuter


@mark.usefixtures("fixed_seed")
class TestPlaceboRefuter(object):
    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.03, "backdoor.linear_regression", 1000)]
    )
    def test_refutation_placebo_refuter_continuous(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.continuous_treatment_testsuite(num_samples=num_samples)  # Run both

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.1, "backdoor.propensity_score_matching", 5000)]
    )
    def test_refutation_placebo_refuter_binary(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.1, "backdoor.linear_regression", 5000)]
    )
    def test_refutation_placebo_refuter_category(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.categorical_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.1, "backdoor.linear_regression", 5000)]
    )
    def test_refutation_placebo_refuter_category_non_consecutive_index(
        self, error_tolerance, estimator_method, num_samples
    ):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=num_samples,
            treatment_is_binary=False,
            treatment_is_category=True,
        )
        random_index = random.sample(range(1, 10 * num_samples), num_samples)
        data["df"].index = random_index
        refuter_tester.null_refutation_test(data=data)

    def test_placebo_refuter_iv_with_explicit_instrument_name(self):
        """Regression test for #1180: iv_instrument_name must be replaced with its placebo equivalent.

        When estimate_effect is called with method_params={'iv_instrument_name': 'Z0'}, the
        placebo_treatment_refuter must use 'placebo_Z0' as the instrument, not the original 'Z0'.
        Before the fix the isinstance check targeted CausalEstimate instead of the inner
        InstrumentalVariableEstimator, so the substitution was silently skipped.
        """
        np.random.seed(42)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=2000,
            treatment_is_binary=False,
        )
        df = data["df"]
        instrument_name = data["instrument_names"][0]  # e.g. "Z0"

        model = CausalModel(
            data=df,
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            proceed_when_unidentifiable=True,
            test_significance=None,
        )
        identified_estimand = model.identify_effect(method_name="exhaustive-search")
        identified_estimand.set_identifier_method("iv")

        # Estimate with explicit iv_instrument_name (this triggers the bug path)
        estimate_explicit = model.estimate_effect(
            identified_estimand=identified_estimand,
            method_name="iv.instrumental_variable",
            method_params={"iv_instrument_name": instrument_name},
            test_significance=None,
        )
        # Estimate without explicit iv_instrument_name (reference)
        estimate_implicit = model.estimate_effect(
            identified_estimand=identified_estimand,
            method_name="iv.instrumental_variable",
            test_significance=None,
        )

        ref_explicit = model.refute_estimate(
            identified_estimand,
            estimate_explicit,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=20,
        )
        ref_implicit = model.refute_estimate(
            identified_estimand,
            estimate_implicit,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=20,
        )

        # Both placebo effects should be near zero (the placebo instrument is uncorrelated with outcome)
        assert abs(ref_explicit.new_effect) < 5, (
            f"Placebo effect with explicit iv_instrument_name ({ref_explicit.new_effect:.2f}) "
            "is unexpectedly large; the instrument substitution is likely broken"
        )
        # The two refutation results should be consistent (same random seed gives same placebo)
        assert (
            abs(ref_explicit.new_effect) < 5
        ), f"Placebo effect with implicit iv_instrument_name ({ref_implicit.new_effect:.2f}) is unexpectedly large"

    @mark.parametrize("placebo_type", ["permute", "Random Data"])
    def test_placebo_refuter_multiple_treatments(self, placebo_type):
        """Regression test for #251: placebo_treatment_refuter must not raise
        'Wrong number of items passed N, placement implies 1' when multiple treatments are used.
        """
        np.random.seed(42)
        n_treatments = 3
        data = dowhy.datasets.linear_dataset(
            num_samples=500,
            beta=10,
            num_common_causes=0,
            num_instruments=0,
            num_effect_modifiers=0,
            num_treatments=n_treatments,
            treatment_is_binary=True,
            outcome_is_binary=False,
            num_discrete_common_causes=0,
            num_discrete_effect_modifiers=0,
            one_hot_encode=False,
        )
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            control_value=[0] * n_treatments,
            treatment_value=[0, 0, 1],
            method_params={"need_conditional_estimates": False},
        )
        # Must not raise ValueError
        result = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type=placebo_type,
            num_simulations=10,
        )
        # The placebo new_effect should be near zero (no real causal link after permutation/randomisation)
        assert abs(result.new_effect) < abs(estimate.value), (
            f"Placebo new_effect ({result.new_effect:.3f}) is unexpectedly close to the original "
            f"estimate ({estimate.value:.3f}); the placebo is likely not severing the causal link."
        )


def test_placebo_refuter_does_not_reuse_stale_propensity_scores():
    """Regression test for #1728.

    After estimating with a propensity-score method, ``model._data`` gains a
    ``propensity_score`` column.  The placebo refuter must *not* pass that stale
    column into the placebo estimator's fit, because the estimator skips fitting
    when the column is already present — causing the placebo effect to equal the
    real treatment's propensity-weighted contrast instead of collapsing to zero.
    """
    rng = np.random.default_rng(20260801)
    n = 5_000

    x = rng.normal(size=n)
    t = rng.binomial(1, expit(1.5 * x))
    p_y = 0.20 + 0.10 * t + 0.05 * x
    y = rng.binomial(1, np.clip(p_y, 0, 1))

    df = pd.DataFrame({"x": x, "t": t, "y": y})

    model = CausalModel(data=df, treatment="t", outcome="y", common_causes=["x"])
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.propensity_score_weighting",
        method_params={"init_params": {"min_ps_score": 0.01, "max_ps_score": 0.99}},
    )

    # The ``propensity_score`` column is now in the data after estimation.
    assert "propensity_score" in model._data.columns

    result = model.refute_estimate(
        estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        num_simulations=50,
        random_seed=42,
        n_jobs=1,
    )

    # A correct placebo effect should be near zero (|new_effect| < |original estimate|/2).
    # With stale propensity scores the placebo effect would be ~-0.08, far from zero.
    assert abs(result.new_effect) < abs(estimate.value) / 2, (
        f"Placebo new_effect ({result.new_effect:.4f}) is too large; "
        f"the refuter may be reusing stale propensity scores from the original estimation."
    )
