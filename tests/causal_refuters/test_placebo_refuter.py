import random

from pytest import mark

import dowhy.datasets

from .base import SimpleRefuter


@mark.usefixtures("fixed_seed")
class TestPlaceboRefuter(object):
    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.03, "backdoor.linear_regression", 1000)]
    )
    def test_refutation_placebo_refuter_continuous(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.continuous_treatment_testsuite(num_samples=num_samples)  # Run both

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.1, "backdoor.propensity_score_matching", 5000)]
    )
    def test_refutation_placebo_refuter_binary(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.1, "backdoor.linear_regression", 5000)]
    )
    def test_refutation_placebo_refuter_category(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        refuter_tester.categorical_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.1, "backdoor.linear_regression", 5000)]
    )
    def test_refutation_placebo_refuter_category_non_consecutive_index(
        self, error_tolerance, estimator_method, num_samples
    ):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "placebo_treatment_refuter")
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_samples=num_samples,
            treatment_is_binary=False,
            treatment_is_category=True,
        )
        random_index = random.sample(range(1, 10 * num_samples), num_samples)
        data["df"].index = random_index
        refuter_tester.null_refutation_test(data=data)
