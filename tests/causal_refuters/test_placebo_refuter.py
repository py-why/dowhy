import pytest
import numpy as np
from .base import TestRefuter

@pytest.mark.usefixtures("fixed_seed")
class TestPlaceboRefuter(object):
    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "num_samples"],
                             [(0.03, "backdoor.linear_regression", 1000)])
    def test_refutation_placebo_refuter_continuous(self, error_tolerance, 
            estimator_method, num_samples):
            refuter_tester = TestRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
            refuter_tester.continuous_treatment_testsuite(num_samples=num_samples) # Run both

    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "num_samples"],
                              [(0.1, "backdoor.propensity_score_matching", 5000)])
    def test_refutation_placebo_refuter_binary(self, error_tolerance, 
            estimator_method, num_samples):
        refuter_tester = TestRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)
