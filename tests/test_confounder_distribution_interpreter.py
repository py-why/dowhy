"""
Regression tests for ConfounderDistributionInterpreter.

Previously the interpreter cast the confounder column to float
(`aggregated[...].astype("float")`) and used the values directly as bar
positions, so it crashed with `ValueError: Cannot cast str dtype to float64`
for categorical/string confounders — exactly the variables one wants to inspect
for balance. It also misaligned the treated/untreated bars when a category was
absent from one group. These tests verify it works for both categorical and
integer-valued confounders.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel


def _build_estimate(confounder_name, confounder_values):
    np.random.seed(42)
    n = 500
    w = np.random.normal(size=n)
    treatment = (np.random.uniform(size=n) < 1 / (1 + np.exp(-w))).astype(int)
    outcome = 2 * treatment + w + np.random.normal(size=n)
    df = pd.DataFrame({"v0": treatment, "W0": w, confounder_name: confounder_values, "y": outcome})
    model = CausalModel(data=df, treatment="v0", outcome="y", common_causes=["W0", confounder_name])
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_weighting")
    return estimate, df


@patch("matplotlib.pyplot.show")
def test_confounder_distribution_interpreter_with_categorical_confounder(mock_show):
    values = pd.Categorical(np.random.RandomState(0).choice(["a", "b", "c"], size=500))
    estimate, df = _build_estimate("Wc", values)
    estimate.interpret(
        method_name="confounder_distribution_interpreter",
        var_name="Wc",
        var_type="discrete",
        fig_size=(8, 4),
        font_size=10,
    )


@patch("matplotlib.pyplot.show")
def test_confounder_distribution_interpreter_with_integer_confounder(mock_show):
    values = np.random.RandomState(0).randint(0, 3, size=500)
    estimate, df = _build_estimate("Wi", values)
    estimate.interpret(
        method_name="confounder_distribution_interpreter",
        var_name="Wi",
        var_type="discrete",
        fig_size=(8, 4),
        font_size=10,
    )
