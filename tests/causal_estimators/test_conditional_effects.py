"""
Regression test for conditional effect estimation across effect modifiers.

`_estimate_conditional_effects` previously called
`groupby(...).apply(fn, include_groups=True)`, which raises
`ValueError: include_groups=True is no longer allowed` on pandas >= 3.0.
These tests confirm conditional effects are computed for single and multiple
effect modifiers, and that the estimators can still access the effect-modifier
(grouping) columns during feature construction.
"""

import numpy as np
import pandas as pd

import dowhy.datasets
from dowhy import CausalModel


def _conditional_estimates(num_effect_modifiers):
    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=3,
        num_effect_modifiers=num_effect_modifiers,
        num_samples=2000,
        treatment_is_binary=True,
    )
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
