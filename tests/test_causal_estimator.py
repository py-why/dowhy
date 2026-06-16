import unittest

import numpy as np
import pytest

from dowhy.causal_estimator import CausalEstimator


class MockEstimator(CausalEstimator):
    pass


def test_causal_estimator_placeholder_methods():
    estimator = MockEstimator(None)
    with pytest.raises(NotImplementedError):
        estimator._do(None)
    with pytest.raises(NotImplementedError):
        estimator.construct_symbolic_estimator(None)


class _FakeEstimator:
    """Minimal stand-in for CausalEstimator to test signif_results_tostr."""

    confidence_level = 0.95


def test_signif_results_tostr_scalar_significant():
    """A scalar p-value below alpha should be reported as Significant."""
    est = _FakeEstimator()
    result = CausalEstimator.signif_results_tostr(est, {"p_value": 0.03})
    assert "0.03" in result
    assert "Significant" in result
    assert "Not significant" not in result
    assert "alpha=0.05" in result


def test_signif_results_tostr_scalar_not_significant():
    """A scalar p-value above alpha should be reported as Not significant."""
    est = _FakeEstimator()
    result = CausalEstimator.signif_results_tostr(est, {"p_value": 0.2})
    assert "0.2" in result
    assert "Not significant" in result


def test_signif_results_tostr_tuple_lower_bound_significant():
    """A (0, hi) tuple where hi <= alpha means definitely significant."""
    est = _FakeEstimator()
    result = CausalEstimator.signif_results_tostr(est, {"p_value": (0, 0.01)})
    assert "p <" in result
    assert "0.01" in result
    assert "Significant" in result
    assert "Not significant" not in result


def test_signif_results_tostr_tuple_upper_bound_not_significant():
    """A (lo, 1) tuple means definitely not significant."""
    est = _FakeEstimator()
    result = CausalEstimator.signif_results_tostr(est, {"p_value": (0.99, 1)})
    assert "p >" in result
    assert "Not significant" in result


def test_signif_results_tostr_tuple_inconclusive():
    """A tuple whose bounds straddle alpha cannot be conclusively classified."""
    est = _FakeEstimator()
    # alpha=0.05; (0.03, 0.1) straddles 0.05 → Inconclusive
    result = CausalEstimator.signif_results_tostr(est, {"p_value": (0.03, 0.1)})
    assert "Inconclusive" in result


def test_signif_results_tostr_array_pvalue():
    """An np.ndarray p-value (multi-treatment) formats each element separately."""
    est = _FakeEstimator()
    # alpha=0.05; first treatment significant, second not
    result = CausalEstimator.signif_results_tostr(est, {"p_value": np.array([0.03, 0.2])})
    assert "0.03" in result
    assert "0.2" in result
    assert "Significant" in result
    assert "Not significant" in result


def test_signif_results_tostr_custom_confidence_level():
    """Custom confidence_level (e.g. 0.99) should compute alpha as 0.01."""
    est = _FakeEstimator()
    est.confidence_level = 0.99
    result = CausalEstimator.signif_results_tostr(est, {"p_value": 0.03})
    assert "alpha=0.01" in result
    assert "Not significant" in result


class TestCausalEstimator(unittest.TestCase):
    def setUp(self):
        # self.df = pd.read_csv(os.path.join(DATA_PATH,'dgp_1/acic_1_1_data.csv'))
        # self.ate = np.mean(self.df['y1'] - self.df['y0'])
        # treated = self.df[self.df['z']==1]
        # self.att = np.mean(treated['y1'] - treated['y0'])
        self.df = None
        self.ate = 1

    # def test_average_treatment_effect(self):
    #     est_ate = 1
    #     bias = est_ate - self.ate
    #     print(bias)
    #     self.assertAlmostEqual(self.ate, est_ate)

    # def test_average_treatment_effect_on_treated(self):
    #    est_att = 1
    #    self.att=1
    #    bias = est_att - self.att
    #    print(bias)
    #    self.assertAlmostEqual(self.att, est_att)


if __name__ == "__main__":
    unittest.main()


def test_estimate_effect_raises_valueerror_for_missing_estimand():
    """estimate_effect() must raise ValueError when no valid estimand is identified.

    Previously, it silently returned None, giving users no indication of failure.
    See issue #1551 https://github.com/py-why/dowhy/issues/1551
    """
    import dowhy.datasets
    from dowhy import CausalModel

    data = dowhy.datasets.linear_dataset(
        beta=10,
        num_common_causes=3,
        num_instruments=0,
        num_samples=500,
        treatment_is_binary=True,
    )
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["dot_graph"],
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    with pytest.raises(ValueError, match=r"No valid identified estimand for 'iv'"):
        model.estimate_effect(
            identified_estimand=estimand,
            method_name="iv.instrumental_variable",
        )
