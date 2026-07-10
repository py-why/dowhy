"""Tests for CausalRefuter base-class initialisation edge cases."""

import pandas as pd

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefuter


def _make_estimand():
    return IdentifiedEstimand(
        identifier=None,
        treatment_variable=["T"],
        outcome_variable=["Y"],
        backdoor_variables={"backdoor0": ["W"]},
        instrumental_variables=[],
        default_backdoor_id="backdoor0",
    )


def _make_estimate_without_estimator():
    """CausalEstimate that has never had add_estimator() called."""
    estimand = _make_estimand()
    est = CausalEstimate(
        data=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        treatment_name=["T"],
        outcome_name=["Y"],
        estimate=1.0,
        target_estimand=estimand,
        realized_estimand_expr="T -> Y",
        control_value=0,
        treatment_value=1,
    )
    # Deliberately do NOT call est.add_estimator(...)
    return estimand, est


def test_variables_of_interest_initialized_when_estimator_missing():
    """
    If estimate.estimator is not set, CausalRefuter.__init__ should still
    initialise _variables_of_interest to [] rather than leaving it unset,
    so that subsequent calls to choose_variables() do not raise AttributeError.
    """
    estimand, est = _make_estimate_without_estimator()

    refuter = CausalRefuter(
        data=pd.DataFrame({"T": [0, 1], "Y": [0, 1], "W": [0, 1]}),
        identified_estimand=estimand,
        estimate=est,
    )

    assert hasattr(
        refuter, "_variables_of_interest"
    ), "_variables_of_interest must be set even when estimate.estimator is missing"
    assert (
        refuter._variables_of_interest == []
    ), "_variables_of_interest should default to [] when estimator is not available"


def test_choose_variables_with_empty_list():
    """choose_variables() works correctly when _variables_of_interest is empty."""
    estimand, est = _make_estimate_without_estimator()

    refuter = CausalRefuter(
        data=pd.DataFrame({"T": [0, 1], "Y": [0, 1], "W": [0, 1]}),
        identified_estimand=estimand,
        estimate=est,
    )

    # With required_variables=False, should return None (all variables, but none exist)
    result = refuter.choose_variables(required_variables=False)
    assert result is None

    # With required_variables=True, should return empty list
    result = refuter.choose_variables(required_variables=True)
    assert result == []
