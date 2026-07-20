from pytest import mark
import numpy as np
import pandas as pd

from dowhy.causal_estimators.propensity_score_weighting_estimator import PropensityScoreWeightingEstimator

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestPropensityScoreWeightingEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "outcome_is_binary",
            "identifier_method",
        ],
        [
            (
                0.4,
                PropensityScoreWeightingEstimator,
                [1, 2],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.4,
                PropensityScoreWeightingEstimator,
                [1, 2],
                [0],
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "general_adjustment",
            ),
        ],
    )
    def test_average_treatment_effect(
        self,
        error_tolerance,
        Estimator,
        num_common_causes,
        num_instruments,
        num_effect_modifiers,
        num_treatments,
        treatment_is_binary,
        outcome_is_binary,
        identifier_method,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={"num_simulations": 1, "num_null_simulations": 1},
        )


def _make_ipw_data(n=3000, seed=42, true_ate=2.0):
    """Helper to generate a simple confounded dataset for IPW tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    ps_true = 1 / (1 + np.exp(-0.5 * X))
    T = rng.binomial(1, ps_true)
    Y = 3 + true_ate * T + X + rng.normal(0, 1, n)
    return pd.DataFrame({"X": X, "T": T, "Y": Y})


def _fit_ipw(data, target_units="ate", weighting_scheme="ips_weight"):
    """Helper to fit the IPW estimator and return the estimate object."""
    from dowhy import CausalModel

    model = CausalModel(data=data, treatment="T", outcome="Y", common_causes=["X"])
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.propensity_score_weighting",
        target_units=target_units,
        method_params={"weighting_scheme": weighting_scheme},
        confidence_intervals=True,
    )
    return estimate


class TestIPWAnalyticSE:
    """Tests for the influence-function-based analytic SE and CI."""

    def test_se_is_positive_and_finite(self):
        data = _make_ipw_data()
        est = _fit_ipw(data)
        se = est.get_standard_error()
        assert se > 0, "SE must be positive"
        assert np.isfinite(se), "SE must be finite"

    def test_ci_covers_true_ate(self):
        """With a well-specified model and large n, the CI should cover."""
        data = _make_ipw_data(n=5000, true_ate=2.0)
        est = _fit_ipw(data)
        ci = est.get_confidence_intervals()
        assert ci[0] <= 2.0 <= ci[1], (
            f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}] does not cover true ATE=2.0"
        )

    def test_ci_width_is_reasonable(self):
        """CI width should shrink with sample size."""
        data_small = _make_ipw_data(n=500)
        data_large = _make_ipw_data(n=5000)
        est_small = _fit_ipw(data_small)
        est_large = _fit_ipw(data_large)
        ci_small = est_small.get_confidence_intervals()
        ci_large = est_large.get_confidence_intervals()
        width_small = ci_small[1] - ci_small[0]
        width_large = ci_large[1] - ci_large[0]
        assert width_small > width_large, (
            f"CI should be narrower with more data: "
            f"n=500 width={width_small:.4f}, n=5000 width={width_large:.4f}"
        )

    def test_se_scales_with_sqrt_n(self):
        """SE should decrease roughly as 1/sqrt(n)."""
        se_500 = _fit_ipw(_make_ipw_data(n=500)).get_standard_error()
        se_2000 = _fit_ipw(_make_ipw_data(n=2000)).get_standard_error()
        # Expected ratio: sqrt(2000/500) = 2.0
        ratio = se_500 / se_2000
        assert 1.5 < ratio < 2.8, f"SE ratio {ratio:.2f} deviates from expected ~2.0"

    def test_analytic_se_close_to_bootstrap(self):
        """Analytic SE should be in the same ballpark as bootstrap."""
        data = _make_ipw_data(n=2000, seed=99)
        est = _fit_ipw(data)
        analytic_se = est.get_standard_error()
        bootstrap_se = est.estimator._estimate_std_error_with_bootstrap(
            data, num_simulations=200, sample_size_fraction=1
        )
        ratio = analytic_se / bootstrap_se
        # Allow some slack: the analytic SE can be slightly conservative
        # because it ignores propensity-score estimation uncertainty.
        assert 0.5 < ratio < 2.0, (
            f"Analytic/bootstrap SE ratio {ratio:.2f} is too far from 1"
        )

    @mark.parametrize("target_units", ["ate", "att", "atc"])
    def test_all_target_units(self, target_units):
        """All target unit types should produce valid SE."""
        data = _make_ipw_data(n=3000)
        est = _fit_ipw(data, target_units=target_units)
        se = est.get_standard_error()
        assert se > 0 and np.isfinite(se), (
            f"SE for target_units={target_units} is invalid: {se}"
        )

    @mark.parametrize(
        "weighting_scheme",
        ["ips_weight", "ips_normalized_weight", "ips_stabilized_weight"],
    )
    def test_all_weighting_schemes(self, weighting_scheme):
        """All weighting schemes should produce valid SE."""
        data = _make_ipw_data(n=3000)
        est = _fit_ipw(data, weighting_scheme=weighting_scheme)
        se = est.get_standard_error()
        assert se > 0 and np.isfinite(se), (
            f"SE for scheme={weighting_scheme} is invalid: {se}"
        )

    def test_influence_function_mean_near_zero(self):
        """The IF should have mean approximately zero (unbiased score)."""
        data = _make_ipw_data(n=5000)
        est = _fit_ipw(data)
        influence = est.estimator._compute_influence_function()
        # Mean should be close to zero relative to the standard deviation
        assert abs(np.mean(influence)) < 0.5 * np.std(influence), (
            f"IF mean {np.mean(influence):.4f} is too far from zero "
            f"(std={np.std(influence):.4f})"
        )

    def test_heterogeneous_effects_att_differs_from_ate(self):
        """When treatment effect varies with X, ATT should differ from ATE."""
        rng = np.random.default_rng(77)
        n = 5000
        X = rng.normal(0, 1, n)
        # Positive selection: higher X -> more likely treated AND higher effect
        ps = 1 / (1 + np.exp(-0.8 * X))
        T = rng.binomial(1, ps)
        # Heterogeneous effect: tau(X) = 2 + 0.5*X
        Y = 1 + (2 + 0.5 * X) * T + X + rng.normal(0, 1, n)
        data = pd.DataFrame({"X": X, "T": T, "Y": Y})

        est_ate = _fit_ipw(data, target_units="ate")
        est_att = _fit_ipw(data, target_units="att")
        # With positive selection, E[X|T=1] > 0, so ATT > ATE
        assert est_att.value > est_ate.value, (
            f"Expected ATT > ATE with positive selection, "
            f"got ATT={est_att.value:.4f}, ATE={est_ate.value:.4f}"
        )
        # Both should have valid SEs
        assert est_ate.get_standard_error() > 0
        assert est_att.get_standard_error() > 0
