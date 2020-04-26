import pytest
import numpy as np
from unittest.mock import patch

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

