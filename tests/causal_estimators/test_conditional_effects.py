"""
Regression tests for conditional effect estimation across effect modifiers.

`_estimate_conditional_effects` previously called
`groupby(...).apply(fn, include_groups=True)`, which raises
`ValueError: include_groups=True is no longer allowed` on pandas >= 3.0.
These tests confirm conditional effects are computed for single and multiple
effect modifiers, and that the estimators can still access the effect-modifier
(grouping) columns during feature construction.

Constant numeric effect modifiers are also covered. Quantile discretization
cannot create bins for a constant column, but it should still produce the one
observed conditional-effect group instead of crashing while constructing the
result index.

Categorical common causes (confounders) are also covered. When effect modifier
strata are formed by groupby, some strata may not contain all categorical levels
of a confounder. The `Encoders` class must reuse the encoder fitted on the full
dataset so that feature dimensions remain consistent across strata (regression
test for https://github.com/py-why/dowhy/issues/401).
"""

import numpy as np
import pandas as pd

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimator
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand


def _conditional_estimates(num_effect_modifiers, constant_effect_modifiers=False):
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=3,
        num_effect_modifiers=num_effect_modifiers,
        num_samples=2000,
        treatment_is_binary=True,
    )
    if constant_effect_modifiers:
        data["df"].loc[:, data["effect_modifier_names"]] = 0.0
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    return estimate.conditional_estimates


def test_conditional_effects_single_effect_modifier():
    conditional_estimates = _conditional_estimates(num_effect_modifiers=1)
    assert isinstance(conditional_estimates, pd.Series)
    assert len(conditional_estimates) > 0
    assert np.all(np.isfinite(conditional_estimates.values))


def test_conditional_effects_multiple_effect_modifiers():
    conditional_estimates = _conditional_estimates(num_effect_modifiers=2)
    assert isinstance(conditional_estimates, pd.Series)
    assert isinstance(conditional_estimates.index, pd.MultiIndex)
    assert len(conditional_estimates) > 0
    assert np.all(np.isfinite(conditional_estimates.values))


def test_conditional_effects_constant_multiple_effect_modifiers():
    conditional_estimates = _conditional_estimates(num_effect_modifiers=2, constant_effect_modifiers=True)
    assert isinstance(conditional_estimates, pd.Series)
    assert isinstance(conditional_estimates.index, pd.MultiIndex)
    assert conditional_estimates.index.nlevels == 2
    assert len(conditional_estimates) == 1
    assert np.all(np.isfinite(conditional_estimates.values))


def test_conditional_effects_all_missing_multiple_effect_modifiers_returns_empty_multiindex():
    estimator = CausalEstimator(IdentifiedEstimand(None, "treatment", "outcome"))
    estimator._effect_modifier_names = ["modifier_a", "modifier_b"]
    data = pd.DataFrame({"modifier_a": [np.nan], "modifier_b": [np.nan]})

    conditional_estimates = estimator._estimate_conditional_effects(data, lambda _: 0.0)

    assert conditional_estimates.empty
    assert isinstance(conditional_estimates.index, pd.MultiIndex)
    prefix = CausalEstimator.TEMP_CAT_COLUMN_PREFIX
    assert conditional_estimates.index.names == [f"{prefix}modifier_a", f"{prefix}modifier_b"]


def test_categorical_common_cause_consistent_encoding():
    """Regression test for https://github.com/py-why/dowhy/issues/401.

    When effect modifier strata are formed by groupby, some strata may not
    contain all categorical levels of a confounder. The encoder must be fitted
    on the full dataset and reused for each stratum so that feature dimensions
    are consistent with the fitted regression model, avoiding a shape mismatch.

    The categorical confounder has three levels (A, B, C), level C is rare
    (4 % of rows) so it will be absent from many effect-modifier strata.
    """
    rng = np.random.default_rng(0)
    n = 2000
    # C is deliberately rare so it will be absent from some strata
    cat_cause = rng.choice(["A", "B", "C"], size=n, p=[0.48, 0.48, 0.04])
    treatment = rng.integers(0, 2, size=n)
    effect_modifier = rng.standard_normal(n)
    outcome = 2.0 * treatment + (cat_cause == "B").astype(float) + 0.5 * effect_modifier + rng.standard_normal(n) * 0.1

    df = pd.DataFrame({"W0": cat_cause, "v0": treatment, "y": outcome, "X0": effect_modifier})

    model = CausalModel(
        data=df,
        treatment=["v0"],
        outcome="y",
        common_causes=["W0"],
        effect_modifiers=["X0"],
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    # The estimate must succeed without raising a shape-mismatch ValueError
    assert estimate.conditional_estimates is not None
    assert isinstance(estimate.conditional_estimates, pd.Series)
    assert len(estimate.conditional_estimates) > 0
    assert np.all(np.isfinite(estimate.conditional_estimates.values))
    # ATE should be near the true effect of 2.0 (with some tolerance)
    assert abs(estimate.value - 2.0) < 0.5

    # The caller's DataFrame must not be mutated (no __categorical__ columns)
    assert list(df.columns) == ["W0", "v0", "y", "X0"]
