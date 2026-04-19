import numpy as np
import pytest
from flaky import flaky

from dowhy.causal_estimator import CausalEstimate
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand
from dowhy.causal_refuter import CausalRefuter, perform_normal_distribution_test


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
