import pytest
import numpy as np
from .base import TestRefuter

@pytest.mark.usefixtures("fixed_seed")
class TestDataSubsetRefuter(object):
    '''
        The first two tests are for the default behavior, in which we just bootstrap the data
        and obtain the estimate.

    '''

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

    
    @pytest.mark.parametrize(["error_tolerance","estimator_method","num_common_causes","required_variables"],
                              [(0.01, "iv.instrumental_variable",5, 3)])
    def test_refutation_bootstrap_refuter_continuous_integer_argument(self, error_tolerance, estimator_method, num_common_causes, required_variables):
        refuter_tester = TestRefuter(error_tolerance, 
                                     estimator_method, 
                                     "bootstrap_refuter",
                                     required_variables=required_variables,
                                     )
        refuter_tester.continuous_treatment_testsuite(num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause") # Run atleats one common cause

    @pytest.mark.parametrize(["error_tolerance","estimator_method", "num_common_causes", "required_variables"],
                              [(0.01, "iv.instrumental_variable", 5, ["X0","X1"])])
    def test_refutation_bootstrap_refuter_continuous_list_argument(self, error_tolerance, estimator_method, num_common_causes, required_variables):
        refuter_tester = TestRefuter(error_tolerance,
                                     estimator_method,
                                     "bootstrap_refuter",
                                     required_variables=required_variables)
        refuter_tester.continuous_treatment_testsuite(num_common_causes=num_common_causes, tests_to_run="atleat-one-common-cause") # Run atleast one common cause

    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "num_common_causes", "required_variables"],
                             [(0.01, "backdoor.propensity_score_matching", 5, 3)])
    def test_refutation_bootstrap_refuter_binary_integer_argument(self, error_tolerance, estimator_method, num_common_causes, required_variables):
        refuter_tester = TestRefuter(error_tolerance, 
                                     estimator_method,
                                    "bootstrap_refuter",
                                    required_variables=required_variables)
        refuter_tester.binary_treatment_testsuite(num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause")
    
    @pytest.mark.parametrize(["error_tolerance", "estimator_method", "num_common_causes", "required_variables"],
                             [(0.01, "backdoor.propensity_score_matching",5, ["X0", "X1"])])
    def test_refutation_bootstrap_refuter_binary_list_argument(self, error_tolerance, estimator_method, num_common_causes, required_variables):
        refuter_tester = TestRefuter(error_tolerance,
                                     estimator_method, 
                                     "bootstrap_refuter",
                                     required_variables=required_variables)
        refuter_tester.binary_treatment_testsuite(num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause")
    