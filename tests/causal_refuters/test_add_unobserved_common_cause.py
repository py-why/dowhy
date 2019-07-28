import pytest

from .base import TestRefuter

class TestAddUnobservedCommonCauseRefuter(object):
    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "effect_strength_on_t", "effect_strength_on_y"],
                             [(0.01, "backdoor.propensity_score_matching", 0.01, 0.02),])
    def test_refutation_continuous_treatment(self, error_tolerance, estimator_method,
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
