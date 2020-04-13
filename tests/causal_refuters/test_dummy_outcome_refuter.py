import pytest
import numpy as np
from .base import TestRefuter

def simple_linear_outcome_model(df):
    # The outcome is a linar function of the confounder
    # The slope is 2 and the intercept is 3
    return df['W0'].values * 2 + 3

@pytest.mark.usefixtures("fixed_seed")
class TestDummyOtcomeRefuter(object):
    @pytest.mark.parametrize(["error_tolerence","estimator_method"],
                             [(0.03, "iv.instrumental_variable")])
    def test_refutation_dummy_outcome_refuter_randomly_generated(self, error_tolerence, estimator_method):
        refuter_tester = TestRefuter(error_tolerence, estimator_method, "dummy_outcome_refuter")
        refuter_tester.continuous_treatment_testsuite()

    @pytest.mark.parametrize(["error_tolerence","estimator_method","outcome_function"],
                             [(0.03, "iv.instrumental_variable",simple_linear_outcome_model)])
    def test_refutation_dummy_outcome_refuter_linear_regression(self, error_tolerence, estimator_method, outcome_function):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
