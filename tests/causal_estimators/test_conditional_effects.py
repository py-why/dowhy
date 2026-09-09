"""
Regression test for conditional effect estimation across effect modifiers.

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


def test_conditional_effects_does_not_mutate_input_dataframe():
    """_estimate_conditional_effects must not add/remove columns on the caller's DataFrame.

    The function discretizes numeric effect modifiers into temporary categorical
    columns before grouping. Without ``data = data.copy()`` those columns would
    be added in-place and the manual cleanup at the end would permanently corrupt
    the caller's DataFrame if any exception occurred mid-way.
    """
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=3,
        num_effect_modifiers=2,
        num_samples=500,
        treatment_is_binary=True,
    )
    df = data["df"]
    cols_before = set(df.columns)

    model = CausalModel(
        data=df,
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    assert set(df.columns) == cols_before, (
        f"_estimate_conditional_effects mutated the input DataFrame: "
        f"added columns {set(df.columns) - cols_before}, "
        f"removed columns {cols_before - set(df.columns)}"
    )
