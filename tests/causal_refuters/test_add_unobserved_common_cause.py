import pytest
import numpy as np
from unittest.mock import patch
import dowhy.datasets
from dowhy import CausalModel
from .base import TestRefuter

@pytest.mark.usefixtures("fixed_seed")
class TestAddUnobservedCommonCauseRefuter(object):
    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
                             [(0.01, "backdoor.propensity_score_matching", 0.01, 0.02),])
    def test_refutation_binary_treatment(self, error_tolerance, estimator_method,
            effect_strength_on_t, effect_strength_on_y):
        refuter_tester = TestRefuter(error_tolerance, estimator_method,
                "add_unobserved_common_cause",
                confounders_effect_on_t = "binary_flip",
                confounders_effect_on_y = "linear",
                effect_strength_on_t = effect_strength_on_t,
                effect_strength_on_y = effect_strength_on_y)
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
                             [(0.01, "iv.instrumental_variable", 0.01, 0.02),])
    def test_refutation_continuous_treatment(self, error_tolerance, estimator_method,
            effect_strength_on_t, effect_strength_on_y):
        refuter_tester = TestRefuter(error_tolerance, estimator_method,
                "add_unobserved_common_cause",
                confounders_effect_on_t = "linear",
                confounders_effect_on_y = "linear",
                effect_strength_on_t = effect_strength_on_t,
                effect_strength_on_y = effect_strength_on_y)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
                             [(0.01, "iv.instrumental_variable", np.arange(0.01, 0.02, 0.001), np.arange(0.02, 0.03, 0.001)),])
    @patch("matplotlib.pyplot.figure")
    def test_refutation_continuous_treatment_range_both_treatment_outcome(self, mock_fig, error_tolerance, estimator_method,
            effect_strength_on_t, effect_strength_on_y):
        refuter_tester = TestRefuter(error_tolerance, estimator_method,
                "add_unobserved_common_cause",
                confounders_effect_on_t = "linear",
                confounders_effect_on_y = "linear",
                effect_strength_on_t = effect_strength_on_t,
                effect_strength_on_y = effect_strength_on_y)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
                             [(0.01, "iv.instrumental_variable", np.arange(0.01, 0.02, 0.001), 0.02),])
    @patch("matplotlib.pyplot.figure")
    def test_refutation_continuous_treatment_range_treatment(self, mock_fig, error_tolerance, estimator_method,
            effect_strength_on_t, effect_strength_on_y):
        refuter_tester = TestRefuter(error_tolerance, estimator_method,
                "add_unobserved_common_cause",
                confounders_effect_on_t = "linear",
                confounders_effect_on_y = "linear",
                effect_strength_on_t = effect_strength_on_t,
                effect_strength_on_y = effect_strength_on_y)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
                             [(0.01, "iv.instrumental_variable", 0.01, np.arange(0.02, 0.03, 0.001)),])
    @patch("matplotlib.pyplot.figure")
    def test_refutation_continuous_treatment_range_outcome(self, mock_fig, error_tolerance, estimator_method,
            effect_strength_on_t, effect_strength_on_y):
        refuter_tester = TestRefuter(error_tolerance, estimator_method,
                "add_unobserved_common_cause",
                confounders_effect_on_t = "linear",
                confounders_effect_on_y = "linear",
                effect_strength_on_t = effect_strength_on_t,
                effect_strength_on_y = effect_strength_on_y)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(["estimator_method", "effect_strength_on_t", "benchmark_common_causes", "simulated_method_name"],
                             [("backdoor.linear_regression", [1,2,3], ["W3"], "linear-partial-R2"),])
    @patch("matplotlib.pyplot.figure")
    def test_linear_sensitivity_with_confounders(self, mock_fig,estimator_method,
            effect_strength_on_t, benchmark_common_causes, simulated_method_name):
        np.random.seed(100) 
        data = dowhy.datasets.linear_dataset( beta = 10,
                                      num_common_causes = 7,
                                      num_samples = 500,
                                      num_treatments = 1,
                                     stddev_treatment_noise =10,
                                     stddev_outcome_noise = 1
                                    )
        data["df"] = data["df"].drop("W4", axis = 1)
        graph_str = 'graph[directed 1node[ id "y" label "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"]  node[ id "W5" label "W5"] node[ id "W6" label "W6"]node[ id "v0" label "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W5" target "v0"] edge[ source "W6" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W5" target "y"] edge[ source "W6" target "y"]]'
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=graph_str,
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(target_estimand,method_name=estimator_method)
        ate_estimate = data['ate']
        refute = model.refute_estimate(target_estimand, estimate ,
                               method_name = "add_unobserved_common_cause",
                               simulated_method_name = simulated_method_name, 
                               benchmark_common_causes = benchmark_common_causes,
                               effect_fraction_on_treatment = effect_strength_on_t)
        
        if refute.confounder_increases_estimate == True:
                bias_adjusted_estimate = refute.benchmarking_results['bias_adjusted_estimate']
                assert all(estimate <= refute.estimate for estimate in bias_adjusted_estimate) #if confounder_increases_estimate is True, adjusted estimate should be lower than original estimate
        else:
                bias_adjusted_estimate = refute.benchmarking_results['bias_adjusted_estimate']
                assert all(estimate >= refute.estimate for estimate in bias_adjusted_estimate)

        #check if all partial R^2 values are between 0 and 1
        assert all((val >= 0 and val <=1) for val in refute.benchmarking_results['r2tu_w']) 
        assert all((val >= 0 and val <=1) for val in refute.benchmarking_results['r2yu_tw'])
        assert  refute.stats['r2yt_w'] >= 0 and refute.stats['r2yt_w'] <= 1

        assert refute.stats['robustness_value'] >= 0 and refute.stats['robustness_value'] <= 1
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(["estimator_method", "effect_strength_on_t", "benchmark_common_causes", "simulated_method_name"],
                             [("backdoor.linear_regression", [1,2,3], ["W3"], "linear-partial-R2"),])
    @patch("matplotlib.pyplot.figure")
    def test_linear_sensitivity_given_strength_of_confounding(self, mock_fig,estimator_method,
            effect_strength_on_t, benchmark_common_causes, simulated_method_name):
        np.random.seed(100) 
        data = dowhy.datasets.linear_dataset( beta = 10,
                                      num_common_causes = 7,
                                      num_samples = 500,
                                      num_treatments = 1,
                                     stddev_treatment_noise =10,
                                     stddev_outcome_noise = 1
                                    )
        data["df"] = data["df"].drop("W4", axis = 1)
        graph_str = 'graph[directed 1node[ id "y" label "y"]node[ id "W0" label "W0"] node[ id "W1" label "W1"] node[ id "W2" label "W2"] node[ id "W3" label "W3"]  node[ id "W5" label "W5"] node[ id "W6" label "W6"]node[ id "v0" label "v0"]edge[source "v0" target "y"]edge[ source "W0" target "v0"] edge[ source "W1" target "v0"] edge[ source "W2" target "v0"] edge[ source "W3" target "v0"] edge[ source "W5" target "v0"] edge[ source "W6" target "v0"]edge[ source "W0" target "y"] edge[ source "W1" target "y"] edge[ source "W2" target "y"] edge[ source "W3" target "y"] edge[ source "W5" target "y"] edge[ source "W6" target "y"]]'
        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=graph_str,
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(target_estimand,method_name=estimator_method)
        ate_estimate = data['ate']
        refute = model.refute_estimate(target_estimand, estimate ,
                               method_name = "add_unobserved_common_cause",
                               simulated_method_name = simulated_method_name, 
                               benchmark_common_causes = benchmark_common_causes,
                               effect_fraction_on_treatment = effect_strength_on_t)
        
        # We calculate adjusted estimates for two sets of partial R^2 values. 
        bias_adjusted_terms = refute.compute_bias_adjusted(r2tu_w = np.array([0.7, 0.2]) , r2yu_tw = np.array([0.9, 0.3]))
        estimate1 = bias_adjusted_terms['bias_adjusted_estimate'][0] #adjusted estimate for confounder u1 where r2tu_w = 0.7 and r2yu_tw = 0.9
        estimate2 = bias_adjusted_terms['bias_adjusted_estimate'][1] #adjusted estimate for confounder u2 where r2tu_w = 0.2 and r2yu_tw = 0.3
        print(estimate1, estimate2)
        original_estimate = refute.estimate
        #Test if hypothetical confounding by unobserved confounder u1 leads to an adjusted effect that is farther from the original estimate as compared to u2 
        assert abs(original_estimate - estimate1) > abs(original_estimate - estimate2)
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests

    @pytest.mark.parametrize(["estimator_method", "effect_strength_on_t", "benchmark_common_causes", "simulated_method_name", "rvalue_threshold"],
                             [("backdoor.linear_regression", [1,2,3], ["W3"], "linear-partial-R2", 0.95),])
    @patch("matplotlib.pyplot.figure")
    def test_linear_sensitivity_dataset_without_confounders(self, mock_fig,estimator_method,
            effect_strength_on_t, benchmark_common_causes, simulated_method_name, rvalue_threshold):
        np.random.seed(100) 
        data = dowhy.datasets.linear_dataset( beta = 10,
                                      num_common_causes = 7,
                                      num_samples = 500,
                                      num_treatments = 1,
                                     stddev_treatment_noise =10,
                                     stddev_outcome_noise = 1
                                    )
        #Creating a model with no unobserved confounders
        model2 = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )
        target_estimand2 = model2.identify_effect(proceed_when_unidentifiable=True)
        estimate2 = model2.estimate_effect(target_estimand2,method_name=estimator_method)
        ate_estimate = data['ate']
        refute2 = model2.refute_estimate(target_estimand2, estimate2 ,
                               method_name = "add_unobserved_common_cause",
                               simulated_method_name = simulated_method_name, 
                               benchmark_common_causes = benchmark_common_causes,
                               effect_fraction_on_treatment = effect_strength_on_t)
        
        if refute2.confounder_increases_estimate == True:
                bias_adjusted_estimate = refute2.benchmarking_results['bias_adjusted_estimate']
                assert all(estimate <= refute2.estimate for estimate in bias_adjusted_estimate) #if confounder_increases_estimate is True, adjusted estimate should be lower than original estimate
        else:
                bias_adjusted_estimate = refute2.benchmarking_results['bias_adjusted_estimate']
                assert all(estimate >= refute2.estimate for estimate in bias_adjusted_estimate)

        #check if all partial R^2 values are between 0 and 1
        assert all((val >= 0 and val <=1) for val in refute2.benchmarking_results['r2tu_w']) 
        assert all((val >= 0 and val <=1) for val in refute2.benchmarking_results['r2yu_tw'])
        assert  refute2.stats['r2yt_w'] >= 0 and refute2.stats['r2yt_w'] <= 1
        
        print(refute2.stats['robustness_value'])
        #for a dataset with no confounders, the robustness value should be higher than a given threshold (0.95 in our case)
        assert refute2.stats['robustness_value'] >= rvalue_threshold and refute2.stats['robustness_value'] <= 1 
        assert mock_fig.call_count > 0  # we patched figure plotting call to avoid drawing plots during tests
    