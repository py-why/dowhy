"""
Tests for the human-readable significance reporting in CausalEstimator.

Covers the formatting of bootstrap significance results (resolving the
ambiguity reported in issue #879) and the dedicated ``significance_level``
parameter, which is independent of the confidence-interval ``confidence_level``.
"""

import numpy as np

from dowhy.causal_estimator import CausalEstimator


class _Estimator:
    """Minimal stand-in exposing only what signif_results_tostr needs."""

    significance_level = CausalEstimator.DEFAULT_SIGNIFICANCE_LEVEL


def _fmt(p_value, significance_level=None):
    est = _Estimator()
    if significance_level is not None:
        est.significance_level = significance_level
    return CausalEstimator.signif_results_tostr(est, {"p_value": p_value})


def test_scalar_pvalue_significant():
    result = _fmt(0.03)
    assert "0.03" in result
    assert "(significant at" in result
    assert "alpha=0.05" in result
    assert "H0" in result


def test_scalar_pvalue_not_significant():
    result = _fmt(0.2)
    assert "not significant" in result


def test_lower_bound_pvalue_uses_less_than():
    # (0, hi) means the estimate is more extreme than every null sample.
    result = _fmt((0, 0.001))
    assert "p < 0.001" in result
    assert "[" not in result  # the old ambiguous "[0, 0.001]" form is gone
    assert "(significant at" in result


def test_upper_bound_pvalue_uses_greater_than():
    result = _fmt((0.99, 1))
    assert "p > 0.99" in result
    assert "not significant" in result


def test_array_pvalue_reports_each_treatment():
    # 0.01 is below alpha=0.05 (significant); 0.3 is above (not significant).
    result = _fmt(np.array([0.01, 0.3]))
    assert "0.01" in result and "0.3" in result
    assert "[significant, not significant]" in result


def test_significance_level_is_independent_of_confidence_level():
    # The same p-value bound is significant at alpha=0.05 but inconclusive at
    # the stricter alpha=0.01 -- the verdict must follow significance_level only.
    assert "(significant at" in _fmt((0, 0.02), significance_level=0.05)
    assert "inconclusive" in _fmt((0, 0.02), significance_level=0.01)


def test_default_significance_level():
    assert CausalEstimator.DEFAULT_SIGNIFICANCE_LEVEL == 0.05
