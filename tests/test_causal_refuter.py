import numpy as np
import pytest
from flaky import flaky

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefuter, choose_variables, perform_normal_distribution_test


class MockRefuter(CausalRefuter):
    pass


def test_causal_refuter_placeholder_method():
    refuter = MockRefuter(None, IdentifiedEstimand(None, None, None), None)
    with pytest.raises(NotImplementedError):
        refuter.refute_estimate()


@flaky(max_runs=3)
def test_causal_refuter_bootstrap_test():
    estimator = CausalEstimate(None, None, None, 0, None, None, None, None)
    refuter = MockRefuter(None, IdentifiedEstimand(None, None, None), None)
    simulations = np.random.normal(0, 1, 5000)
    pvalue = refuter.perform_bootstrap_test(estimator, simulations)
    assert pvalue > 0.95


def test_normal_distribution_test_zero_std_returns_one():
    """Regression test for issue #807: when all simulations have the same value (std=0),
    perform_normal_distribution_test must return 1.0 instead of NaN."""
    estimate = CausalEstimate(None, None, None, 5.0, None, None, None, None)
    # All simulations return the same value as the estimate → std=0, p-value should be 1.0
    simulations_identical = [5.0] * 20
    p_value = perform_normal_distribution_test(estimate, simulations_identical)
    assert not np.isnan(p_value), "p-value must not be NaN when std of simulations is 0"
    assert p_value == 1.0, f"Expected p_value=1.0 for zero-variance simulations, got {p_value}"


# ---------------------------------------------------------------------------
# Unit tests for choose_variables()
# ---------------------------------------------------------------------------

_VARS = ["W0", "W1", "W2", "Z0"]


def test_choose_variables_true_returns_all():
    """True → all variables_of_interest are returned."""
    result = choose_variables(True, _VARS)
    assert result == _VARS


def test_choose_variables_false_returns_none():
    """False → None (no variables selected, no noise applied)."""
    result = choose_variables(False, _VARS)
    assert result is None


def test_choose_variables_int_returns_correct_count():
    """Integer n → exactly n variables drawn from variables_of_interest."""
    result = choose_variables(2, _VARS)
    assert len(result) == 2
    assert all(v in _VARS for v in result)


def test_choose_variables_int_zero_returns_empty():
    """Integer 0 → empty list (no variables selected)."""
    result = choose_variables(0, _VARS)
    assert result == []


def test_choose_variables_int_too_large_raises():
    """Integer larger than pool size → ValueError."""
    with pytest.raises(ValueError, match="greater than the number of confounders"):
        choose_variables(len(_VARS) + 1, _VARS)


def test_choose_variables_list_select_returns_subset():
    """Explicit list (no `-` prefix) → exactly those variables returned."""
    result = choose_variables(["W0", "Z0"], _VARS)
    assert result == ["W0", "Z0"]


def test_choose_variables_list_deselect_returns_complement():
    """List with `-` prefix → complement of named variables."""
    result = choose_variables(["-W0", "-Z0"], _VARS)
    assert sorted(result) == sorted(["W1", "W2"])


def test_choose_variables_list_deselect_all_returns_empty():
    """Deselecting all variables → empty set returned."""
    result = choose_variables(["-W0", "-W1", "-W2", "-Z0"], _VARS)
    assert set(result) == set()


def test_choose_variables_list_mixed_select_deselect_raises():
    """Mix of select and deselect entries → ValueError."""
    with pytest.raises(ValueError, match="select and deselect"):
        choose_variables(["W0", "-W1"], _VARS)


def test_choose_variables_list_invalid_variable_raises():
    """Variable name not in variables_of_interest → ValueError."""
    with pytest.raises(ValueError, match="not a valid variable"):
        choose_variables(["W0", "NOT_A_VAR"], _VARS)


def test_choose_variables_empty_pool_true_returns_empty():
    """True with empty pool → empty list (edge case)."""
    result = choose_variables(True, [])
    assert result == []


def test_choose_variables_empty_pool_false_returns_none():
    """False with empty pool → None."""
    result = choose_variables(False, [])
    assert result is None
