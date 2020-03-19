import pytest
import numpy as np
from .base import TestRefuter

@pytest.mark.usefixtures("fixed_seed")
class TestDataSubsetRefuter(object):
    @pytest.mark.parametrize(["error_tolerance","estimator_method"],
                              [(0.01, "iv.instrumental_variable")])
    def test_refutation_bootstrap_refuter_continuous(self, error_tolerance, estimator_method):
        refuter_tester = TestRefuter(error_tolerance, estimator_method, "bootstrap_refuter")
        refuter_tester.continuous_treatment_testsuite() # Run both

    @pytest.mark.parametrize(["error_tolerance", "estimator_method"],
                             [(0.01, "backdoor.propensity_score_matching")])
    def test_refutation_bootstrap_refuter_binary(self, error_tolerance, estimator_method):
        refuter_tester = TestRefuter(error_tolerance, estimator_method, "bootstrap_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause") 
