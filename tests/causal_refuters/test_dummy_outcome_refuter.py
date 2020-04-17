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
    def test_refutation_dummy_outcome_refuter_custom_function_linear_regression(self, error_tolerence, estimator_method, outcome_function):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @pytest.mark.parametrize(["error_tolerence","estimator_method","outcome_function"],
                             [(0.1, "iv.instrumental_variable","linear_regression")])
    def test_refutation_dummy_outcome_refuter_internal_linear_regression(self, error_tolerence, estimator_method, outcome_function):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @pytest.mark.parametrize(["error_tolerence","estimator_method","outcome_function","params"],
                             [(0.1, "iv.instrumental_variable","knn",{'n_neighbors':5})])
    def test_refutation_dummy_outcome_refuter_internal_knn(self, error_tolerence, estimator_method, outcome_function,params):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function,
                                    params=params)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")
    
    @pytest.mark.parametrize(["error_tolerence","estimator_method","outcome_function","params","num_samples"],
                             [(0.1, "iv.instrumental_variable","svm",{'C':1,'epsilon':0.2}, 1000)])
    def test_refutation_dummy_outcome_refuter_internal_svm(self, error_tolerence, estimator_method, outcome_function,params, num_samples):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function,
                                    params=params)
        refuter_tester.continuous_treatment_testsuite(num_samples=num_samples, tests_to_run="atleast-one-common-cause")

    @pytest.mark.parametrize(["error_tolerence","estimator_method","outcome_function","params","num_samples"],
                             [(0.1, "iv.instrumental_variable","random_forest",{'max_depth':20}, 1000)])
    def test_refutation_dummy_outcome_refuter_internal_random_forest(self, error_tolerence, estimator_method, outcome_function,params, num_samples):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function,
                                    params=params)
        refuter_tester.continuous_treatment_testsuite(num_samples,tests_to_run="atleast-one-common-cause")

    # As we run with only one common cause and one instrument variable we run with (?, 2)
    @pytest.mark.parametrize(["error_tolerence","estimator_method","outcome_function","params"],
                             [(0.1, "iv.instrumental_variable","neural_network",{'solver':'lbfgs', 'alpha':1e-5, 'hidden_layer_sizes':(5,2)})])
    def test_refutation_dummy_outcome_refuter_internal_neural_network(self, error_tolerence, estimator_method, outcome_function,params):
        refuter_tester = TestRefuter(error_tolerence, 
                                    estimator_method, 
                                    "dummy_outcome_refuter",
                                    outcome_function=outcome_function,
                                    params=params)
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")