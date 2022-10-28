import numpy as np
from pytest import mark

from .base import TestRefuter


@mark.usefixtures("fixed_seed")
class TestDataSubsetRefuter(object):
    @mark.parametrize(["error_tolerance", "estimator_method"], [(0.01, "iv.instrumental_variable")])
    def test_refutation_data_subset_refuter_continuous(self, error_tolerance, estimator_method):
        refuter_tester = TestRefuter(error_tolerance, estimator_method, "data_subset_refuter")
        refuter_tester.continuous_treatment_testsuite()  # Run both

    @mark.parametrize(["error_tolerance", "estimator_method"], [(0.01, "backdoor.propensity_score_matching")])
    def test_refutation_data_subset_refuter_binary(self, error_tolerance, estimator_method):
        refuter_tester = TestRefuter(error_tolerance, estimator_method, "data_subset_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause")
