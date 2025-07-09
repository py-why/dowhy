import sys
from unittest.mock import patch

import numpy as np
import pytest
import statsmodels.api as sm
from pytest import mark
from sklearn.ensemble import GradientBoostingRegressor

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_refuters.evalue_sensitivity_analyzer import EValueSensitivityAnalyzer

from .base import SimpleRefuter


@mark.usefixtures("fixed_seed")
class TestAddUnobservedCommonCauseRefuter(object):
    @mark.parametrize(
        ["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
        [
            (0.01, "backdoor.propensity_score_matching", 0.01, 0.02),
        ],
    )
    def test_refutation_binary_treatment(
        self, error_tolerance, estimator_method, effect_strength_on_t, effect_strength_on_y
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance,
            estimator_method,
            "add_unobserved_common_cause",
            confounders_effect_on_t="binary_flip",
            confounders_effect_on_y="linear",
            effect_strength_on_t=effect_strength_on_t,
            effect_strength_on_y=effect_strength_on_y,
        )
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
        [
            (0.01, "iv.instrumental_variable", 0.01, 0.02),
        ],
    )
    def test_refutation_continuous_treatment(
        self, error_tolerance, estimator_method, effect_strength_on_t, effect_strength_on_y
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance,
            estimator_method,
            "add_unobserved_common_cause",
            confounders_effect_on_t="linear",
            confounders_effect_on_y="linear",
            effect_strength_on_t=effect_strength_on_t,
            effect_strength_on_y=effect_strength_on_y,
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
        [
            (0.01, "iv.instrumental_variable", np.arange(0.01, 0.02, 0.001), np.arange(0.02, 0.03, 0.001)),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_refutation_continuous_treatment_range_both_treatment_outcome(
        self, mock_fig, error_tolerance, estimator_method, effect_strength_on_t, effect_strength_on_y
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance,
            estimator_method,
            "add_unobserved_common_cause",
            confounders_effect_on_t="linear",
            confounders_effect_on_y="linear",
            effect_strength_on_t=effect_strength_on_t,
            effect_strength_on_y=effect_strength_on_y,
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
        [
            (0.01, "iv.instrumental_variable", np.arange(0.01, 0.02, 0.001), 0.02),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_refutation_continuous_treatment_range_treatment(
        self, mock_fig, error_tolerance, estimator_method, effect_strength_on_t, effect_strength_on_y
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance,
            estimator_method,
            "add_unobserved_common_cause",
            confounders_effect_on_t="linear",
            confounders_effect_on_y="linear",
            effect_strength_on_t=effect_strength_on_t,
            effect_strength_on_y=effect_strength_on_y,
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
        [
            (0.01, "iv.instrumental_variable", 0.01, np.arange(0.02, 0.03, 0.001)),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_refutation_continuous_treatment_range_outcome(
        self, mock_fig, error_tolerance, estimator_method, effect_strength_on_t, effect_strength_on_y
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance,
            estimator_method,
            "add_unobserved_common_cause",
            confounders_effect_on_t="linear",
            confounders_effect_on_y="linear",
            effect_strength_on_t=effect_strength_on_t,
            effect_strength_on_y=effect_strength_on_y,
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(
        ["estimator_method", "effect_fraction_on_treatment", "benchmark_common_causes", "simulation_method"],
        [
            ("backdoor.linear_regression", [1, 2, 3], ["W3"], "linear-partial-R2"),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_linear_sensitivity_with_confounders(
        self, mock_fig, estimator_method, effect_fraction_on_treatment, benchmark_common_causes, simulation_method
    ):
        np.random.seed(100)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=7,
            num_samples=500,
            num_treatments=1,
            stddev_treatment_noise=10,
            stddev_outcome_noise=1,
        )
        data["df"] = data["df"].drop("W4", axis=1)
        graph_str = 'graph[directed 1node[ id "y" label "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"]  node[ id "W5" label "W5"] node[ id "W6" label "W6"]node[ id "v0" label "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W5" target "v0"] edge[ source "W6" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W5" target "y"] edge[ source "W6" target "y"]]'
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=graph_str,
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(target_estimand, method_name=estimator_method)
        ate_estimate = data["ate"]
        refute = model.refute_estimate(
            target_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method=simulation_method,
            benchmark_common_causes=benchmark_common_causes,
            effect_fraction_on_treatment=effect_fraction_on_treatment,
        )
        if refute.confounder_increases_estimate == True:
            bias_adjusted_estimate = refute.benchmarking_results["bias_adjusted_estimate"]
            assert all(
                estimate <= refute.estimate for estimate in bias_adjusted_estimate
            )  # if confounder_increases_estimate is True, adjusted estimate should be lower than original estimate
        else:
            bias_adjusted_estimate = refute.benchmarking_results["bias_adjusted_estimate"]
            assert all(estimate >= refute.estimate for estimate in bias_adjusted_estimate)

        # check if all partial R^2 values are between 0 and 1
        assert all((val >= 0 and val <= 1) for val in refute.benchmarking_results["r2tu_w"])
        assert all((val >= 0 and val <= 1) for val in refute.benchmarking_results["r2yu_tw"])
        assert refute.stats["r2yt_w"] >= 0 and refute.stats["r2yt_w"] <= 1

        assert refute.stats["robustness_value"] >= 0 and refute.stats["robustness_value"] <= 1
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(
        ["estimator_method", "effect_fraction_on_treatment", "benchmark_common_causes", "simulation_method"],
        [
            ("backdoor.linear_regression", [1, 2, 3], ["W3"], "linear-partial-R2"),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_linear_sensitivity_given_strength_of_confounding(
        self, mock_fig, estimator_method, effect_fraction_on_treatment, benchmark_common_causes, simulation_method
    ):
        np.random.seed(100)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=7,
            num_samples=500,
            num_treatments=1,
            stddev_treatment_noise=10,
            stddev_outcome_noise=1,
        )
        data["df"] = data["df"].drop("W4", axis=1)
        graph_str = 'graph[directed 1node[ id "y" label "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"]  node[ id "W5" label "W5"] node[ id "W6" label "W6"]node[ id "v0" label "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W5" target "v0"] edge[ source "W6" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W5" target "y"] edge[ source "W6" target "y"]]'
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=graph_str,
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(target_estimand, method_name=estimator_method)
        ate_estimate = data["ate"]
        refute = model.refute_estimate(
            target_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method=simulation_method,
            benchmark_common_causes=benchmark_common_causes,
            effect_fraction_on_treatment=effect_fraction_on_treatment,
        )

        # We calculate adjusted estimates for two sets of partial R^2 values.
        bias_adjusted_terms = refute.compute_bias_adjusted(r2tu_w=np.array([0.7, 0.2]), r2yu_tw=np.array([0.9, 0.3]))
        estimate1 = bias_adjusted_terms["bias_adjusted_estimate"][
            0
        ]  # adjusted estimate for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        estimate2 = bias_adjusted_terms["bias_adjusted_estimate"][
            1
        ]  # adjusted estimate for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        print(estimate1, estimate2)
        original_estimate = refute.estimate
        # Test if hypothetical confounding by unobserved confounder u1 leads to an adjusted effect that is farther from the original estimate as compared to u2
        assert abs(original_estimate - estimate1) > abs(original_estimate - estimate2)
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @mark.parametrize(
        [
            "estimator_method",
            "effect_fraction_on_treatment",
            "benchmark_common_causes",
            "simulation_method",
            "rvalue_threshold",
        ],
        [
            ("backdoor.linear_regression", [1, 2, 3], ["W3"], "linear-partial-R2", 0.95),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_linear_sensitivity_dataset_without_confounders(
        self,
        mock_fig,
        estimator_method,
        effect_fraction_on_treatment,
        benchmark_common_causes,
        simulation_method,
        rvalue_threshold,
    ):
        np.random.seed(100)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=7,
            num_samples=500,
            num_treatments=1,
            stddev_treatment_noise=10,
            stddev_outcome_noise=1,
        )
        # Creating a model with no unobserved confounders
        model2 = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )
        target_estimand2 = model2.identify_effect(proceed_when_unidentifiable=True)
        estimate2 = model2.estimate_effect(target_estimand2, method_name=estimator_method)
        ate_estimate = data["ate"]
        refute2 = model2.refute_estimate(
            target_estimand2,
            estimate2,
            method_name="add_unobserved_common_cause",
            simulation_method=simulation_method,
            benchmark_common_causes=benchmark_common_causes,
            effect_fraction_on_treatment=effect_fraction_on_treatment,
        )

        if refute2.confounder_increases_estimate == True:
            bias_adjusted_estimate = refute2.benchmarking_results["bias_adjusted_estimate"]
            assert all(
                estimate <= refute2.estimate for estimate in bias_adjusted_estimate
            )  # if confounder_increases_estimate is True, adjusted estimate should be lower than original estimate
        else:
            bias_adjusted_estimate = refute2.benchmarking_results["bias_adjusted_estimate"]
            assert all(estimate >= refute2.estimate for estimate in bias_adjusted_estimate)

        # check if all partial R^2 values are between 0 and 1
        assert all((val >= 0 and val <= 1) for val in refute2.benchmarking_results["r2tu_w"])
        assert all((val >= 0 and val <= 1) for val in refute2.benchmarking_results["r2yu_tw"])
        assert refute2.stats["r2yt_w"] >= 0 and refute2.stats["r2yt_w"] <= 1

        print(refute2.stats["robustness_value"])
        # for a dataset with no confounders, the robustness value should be higher than a given threshold (0.95 in our case)
        assert refute2.stats["robustness_value"] >= rvalue_threshold and refute2.stats["robustness_value"] <= 1
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.econml
    @pytest.mark.parametrize(
        ["estimator_method", "effect_fraction_on_treatment", "benchmark_common_causes", "simulation_method"],
        [
            ("backdoor.econml.dml.KernelDML", 2, ["W3"], "non-parametric-partial-R2"),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_non_parametric_sensitivity_given_strength_of_confounding(
        self, mock_fig, estimator_method, effect_fraction_on_treatment, benchmark_common_causes, simulation_method
    ):
        np.random.seed(100)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=7,
            num_samples=500,
            num_treatments=1,
            stddev_treatment_noise=5,
            stddev_outcome_noise=5,
        )
        data["df"] = data["df"].drop("W4", axis=1)
        graph_str = 'graph[directed 1node[ id "y" label "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"]  node[ id "W5" label "W5"] node[ id "W6" label "W6"]node[ id "v0" label "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W5" target "v0"] edge[ source "W6" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W5" target "y"] edge[ source "W6" target "y"]]'
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=graph_str,
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # Non Parametric estimator
        estimate = model.estimate_effect(
            target_estimand,
            method_name=estimator_method,
            method_params={
                "init_params": {
                    "model_y": GradientBoostingRegressor(),
                    "model_t": GradientBoostingRegressor(),
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )
        ate_estimate = data["ate"]
        refute = model.refute_estimate(
            target_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method=simulation_method,
            benchmark_common_causes=benchmark_common_causes,
            effect_fraction_on_treatment=effect_fraction_on_treatment,
        )

        assert refute.r2yu_tw >= 0 and refute.r2yu_tw <= 1
        assert refute.r2tu_w >= 0 and refute.r2tu_w <= 1

        # We calculate adjusted estimates for two sets of partial R^2 values.
        benchmarking_results_u1 = refute.perform_benchmarking(r2yu_tw=0.9, r2tu_w=0.7, significance_level=0.05)
        benchmarking_results_u2 = refute.perform_benchmarking(r2yu_tw=0.3, r2tu_w=0.2, significance_level=0.05)
        # adjusted lower ate bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        lower_ate_bound_u1 = benchmarking_results_u1["lower_ate_bound"]
        # adjusted lower ate bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        lower_ate_bound_u2 = benchmarking_results_u2["lower_ate_bound"]

        # adjusted upper ate bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        upper_ate_bound_u1 = benchmarking_results_u1["upper_ate_bound"]
        # adjusted upper ate bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        upper_ate_bound_u2 = benchmarking_results_u2["upper_ate_bound"]

        # adjusted lower confidence bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        lower_confidence_bound_u1 = benchmarking_results_u1["lower_confidence_bound"]
        # adjusted lower confidence bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        lower_confidence_bound_u2 = benchmarking_results_u2["lower_confidence_bound"]

        # adjusted upper confidence bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        upper_confidence_bound_u1 = benchmarking_results_u1["upper_confidence_bound"]
        # adjusted upper confidence bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        upper_confidence_bound_u2 = benchmarking_results_u2["upper_confidence_bound"]

        original_estimate = refute.theta_s
        # Test if hypothetical confounding by unobserved confounder u1 leads to an adjusted effect that is farther from the original estimate as compared to u2
        assert abs(original_estimate - lower_ate_bound_u1) > abs(original_estimate - lower_ate_bound_u2)
        assert abs(original_estimate - upper_ate_bound_u1) > abs(original_estimate - upper_ate_bound_u2)
        assert abs(original_estimate - lower_confidence_bound_u1) > abs(original_estimate - lower_confidence_bound_u2)
        assert abs(original_estimate - upper_confidence_bound_u1) > abs(original_estimate - upper_confidence_bound_u2)
        # we patched figure plotting call to avoid drawing plots during tests
        assert mock_fig.call_count > 0

    @pytest.mark.econml
    @pytest.mark.parametrize(
        ["estimator_method", "effect_fraction_on_outcome", "benchmark_common_causes", "simulation_method"],
        [
            ("backdoor.econml.dml.dml.LinearDML", 2, ["W3"], "non-parametric-partial-R2"),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_partially_linear_sensitivity_given_strength_of_confounding(
        self, mock_fig, estimator_method, effect_fraction_on_outcome, benchmark_common_causes, simulation_method
    ):
        np.random.seed(100)
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=7,
            num_samples=500,
            num_treatments=1,
            stddev_treatment_noise=5,
            stddev_outcome_noise=5,
        )
        data["df"] = data["df"].drop("W4", axis=1)
        graph_str = 'graph[directed 1node[ id "y" label "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"]  node[ id "W5" label "W5"] node[ id "W6" label "W6"]node[ id "v0" label "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W5" target "v0"] edge[ source "W6" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W5" target "y"] edge[ source "W6" target "y"]]'
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=graph_str,
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            target_estimand,
            method_name=estimator_method,
            method_params={
                "init_params": {
                    "model_y": GradientBoostingRegressor(),
                    "model_t": GradientBoostingRegressor(),
                    "linear_first_stages": False,
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )
        ate_estimate = data["ate"]
        refute = model.refute_estimate(
            target_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method=simulation_method,
            benchmark_common_causes=benchmark_common_causes,
            effect_fraction_on_outcome=effect_fraction_on_outcome,
        )

        assert refute.r2yu_tw >= 0 and refute.r2yu_tw <= 1
        assert refute.r2tu_w >= 0 and refute.r2tu_w <= 1

        # We calculate adjusted estimates for two sets of partial R^2 values.
        benchmarking_results_u1 = refute.perform_benchmarking(r2yu_tw=0.9, r2tu_w=0.7, significance_level=0.05)
        benchmarking_results_u2 = refute.perform_benchmarking(r2yu_tw=0.3, r2tu_w=0.2, significance_level=0.05)
        # adjusted lower ate bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        lower_ate_bound_u1 = benchmarking_results_u1["lower_ate_bound"]
        # adjusted lower ate bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        lower_ate_bound_u2 = benchmarking_results_u2["lower_ate_bound"]

        # adjusted upper ate bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        upper_ate_bound_u1 = benchmarking_results_u1["upper_ate_bound"]
        # adjusted upper ate bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        upper_ate_bound_u2 = benchmarking_results_u2["upper_ate_bound"]

        # adjusted lower confidence bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        lower_confidence_bound_u1 = benchmarking_results_u1["lower_confidence_bound"]
        # adjusted lower confidence bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        lower_confidence_bound_u2 = benchmarking_results_u2["lower_confidence_bound"]

        # adjusted upper confidence bound for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        upper_confidence_bound_u1 = benchmarking_results_u1["upper_confidence_bound"]
        # adjusted upper confidence bound for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        upper_confidence_bound_u2 = benchmarking_results_u2["upper_confidence_bound"]

        original_estimate = refute.theta_s
        # Test if hypothetical confounding by unobserved confounder u1 leads to an adjusted effect that is farther from the original estimate as compared to u2
        assert abs(original_estimate - lower_ate_bound_u1) > abs(original_estimate - lower_ate_bound_u2)
        assert abs(original_estimate - upper_ate_bound_u1) > abs(original_estimate - upper_ate_bound_u2)
        assert abs(original_estimate - lower_confidence_bound_u1) > abs(original_estimate - lower_confidence_bound_u2)
        assert abs(original_estimate - upper_confidence_bound_u1) > abs(original_estimate - upper_confidence_bound_u2)
        # we patched figure plotting call to avoid drawing plots during tests
        assert mock_fig.call_count > 0

    def test_evalue(self):
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=5,
            num_samples=1000,
            treatment_is_binary=True,
        )
        model = CausalModel(
            data=data["df"], treatment=data["treatment_name"], outcome=data["outcome_name"], graph=data["gml_graph"]
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        outcome_vals = data["df"][data["outcome_name"]]
        analyzer = EValueSensitivityAnalyzer(
            estimate, identified_estimand, data["df"], data["treatment_name"], data["outcome_name"]
        )

        # comparing test examples from R E-Value package
        stats = analyzer._evalue_OLS(0.293, 0.196, 17.405)
        assert stats["converted_estimate"].round(3) == 1.015
        assert stats["converted_lower_ci"].round(3) == 0.995
        assert stats["converted_upper_ci"].round(3) == 1.036
        assert stats["evalue_estimate"].round(3) == 1.141
        assert stats["evalue_lower_ci"].round(3) == 1.000
        assert stats["evalue_upper_ci"] is None

        stats = analyzer._evalue_MD(0.5, 0.25)
        assert stats["converted_estimate"].round(3) == 1.576
        assert stats["converted_lower_ci"].round(3) == 1.010
        assert stats["converted_upper_ci"].round(3) == 2.460
        assert stats["evalue_estimate"].round(3) == 2.529
        assert stats["evalue_lower_ci"].round(3) == 1.111
        assert stats["evalue_upper_ci"] is None

        stats = analyzer._evalue_OR(0.86, 0.75, 0.99, rare=False)
        assert stats["converted_estimate"].round(3) == 0.927
        assert stats["converted_lower_ci"].round(3) == 0.866
        assert stats["converted_upper_ci"].round(3) == 0.995
        assert stats["evalue_estimate"].round(3) == 1.369
        assert stats["evalue_lower_ci"] == None
        assert stats["evalue_upper_ci"].round(3) == 1.076

        stats = analyzer._evalue_RR(0.8, 0.71, 0.91)
        assert stats["converted_estimate"] == 0.8
        assert stats["converted_lower_ci"] == 0.71
        assert stats["converted_upper_ci"] == 0.91
        assert stats["evalue_estimate"].round(3) == 1.809
        assert stats["evalue_lower_ci"] == None
        assert stats["evalue_upper_ci"].round(3) == 1.429

        # check implementation of Observed Covariate E-value against R package
        assert analyzer._observed_covariate_e_value(2, 4).round(3) == 3.414
        assert analyzer._observed_covariate_e_value(4, 2).round(3) == 3.414
        assert analyzer._observed_covariate_e_value(0.8, 0.9).round(3) == 1.5
        assert analyzer._observed_covariate_e_value(0.9, 0.8).round(3) == 1.5

    @pytest.mark.parametrize(
        "estimator_method",
        [
            ("backdoor.linear_regression"),
            ("general_adjustment.linear_regression"),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_evalue_linear_regression(self, mock_fig, estimator_method):
        # generalized adjustment identification requires python >=3.10
        if estimator_method.startswith("general_adjustment") and sys.version_info < (3, 10):
            return
        data = dowhy.datasets.linear_dataset(
            beta=10, num_common_causes=5, num_samples=1000, treatment_is_binary=True, stddev_outcome_noise=5
        )
        model = CausalModel(
            data=data["df"], treatment=data["treatment_name"], outcome=data["outcome_name"], graph=data["gml_graph"]
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name=estimator_method)
        refute = model.refute_estimate(
            identified_estimand, estimate, method_name="add_unobserved_common_cause", simulation_method="e-value"
        )

        assert refute.stats["evalue_upper_ci"] is None
        assert refute.stats["evalue_lower_ci"] < refute.stats["evalue_estimate"]
        assert mock_fig.call_count > 0

    @pytest.mark.parametrize(
        "estimator_method",
        [
            ("backdoor.generalized_linear_model"),
            ("general_adjustment.generalized_linear_model"),
        ],
    )
    @patch("matplotlib.pyplot.figure")
    def test_evalue_logistic_regression(self, mock_fig, estimator_method):
        # generalized adjustment identification requires python >=3.10
        if estimator_method.startswith("general_adjustment") and sys.version_info < (3, 10):
            return
        data = dowhy.datasets.linear_dataset(
            beta=10,
            outcome_is_binary=True,
            num_common_causes=5,
            num_samples=1000,
            treatment_is_binary=True,
            stddev_outcome_noise=5,
        )
        model = CausalModel(
            data=data["df"], treatment=data["treatment_name"], outcome=data["outcome_name"], graph=data["gml_graph"]
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(
            identified_estimand, method_name=estimator_method, method_params={"glm_family": sm.families.Binomial()}
        )
        refute = model.refute_estimate(
            identified_estimand, estimate, method_name="add_unobserved_common_cause", simulation_method="e-value"
        )

        assert refute.stats["evalue_upper_ci"] is None
        assert refute.stats["evalue_lower_ci"] < refute.stats["evalue_estimate"]
        assert mock_fig.call_count > 0
