import sys

from pytest import mark

from .base import SimpleRefuter


@mark.usefixtures("fixed_seed")
class TestDataSubsetRefuter(object):
    """
    The first two tests are for the default behavior, in which we just bootstrap the data
    and obtain the estimate.

    """

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"], [(0.05, "iv.instrumental_variable", 1000)]
    )
    def test_refutation_bootstrap_refuter_continuous(self, error_tolerance, estimator_method, num_samples):
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "bootstrap_refuter")
        refuter_tester.continuous_treatment_testsuite(num_samples=num_samples)  # Run both

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_samples"],
        [
            (0.05, "backdoor.propensity_score_matching", 1000),
            (0.05, "general_adjustment.propensity_score_matching", 1000),
        ],
    )
    def test_refutation_bootstrap_refuter_binary(self, error_tolerance, estimator_method, num_samples):
        # generalized adjustment identification requires python >=3.10
        if estimator_method.startswith("general_adjustment") and sys.version_info < (3, 10):
            return
        refuter_tester = SimpleRefuter(error_tolerance, estimator_method, "bootstrap_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_common_causes", "required_variables", "num_samples"],
        [(0.05, "iv.instrumental_variable", 5, 3, 1000)],
    )
    def test_refutation_bootstrap_refuter_continuous_integer_argument(
        self, error_tolerance, estimator_method, num_common_causes, required_variables, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance,
            estimator_method,
            "bootstrap_refuter",
            required_variables=required_variables,
        )
        refuter_tester.continuous_treatment_testsuite(
            num_samples=num_samples, num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause"
        )  # Run atleast one common cause

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_common_causes", "required_variables", "num_samples"],
        [(0.05, "iv.instrumental_variable", 5, ["W0", "W1"], 1000)],
    )
    def test_refutation_bootstrap_refuter_continuous_list_argument(
        self, error_tolerance, estimator_method, num_common_causes, required_variables, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance, estimator_method, "bootstrap_refuter", required_variables=required_variables
        )
        refuter_tester.continuous_treatment_testsuite(
            num_samples=num_samples, num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause"
        )  # Run atleast one common cause

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_common_causes", "required_variables", "num_samples"],
        [
            (0.1, "backdoor.propensity_score_matching", 5, 3, 5000),
            (0.1, "general_adjustment.propensity_score_matching", 5, 3, 5000),
        ],
    )
    def test_refutation_bootstrap_refuter_binary_integer_argument(
        self, error_tolerance, estimator_method, num_common_causes, required_variables, num_samples
    ):
        # generalized adjustment identification requires python >=3.10
        if estimator_method.startswith("general_adjustment") and sys.version_info < (3, 10):
            return
        refuter_tester = SimpleRefuter(
            error_tolerance, estimator_method, "bootstrap_refuter", required_variables=required_variables
        )
        refuter_tester.binary_treatment_testsuite(
            num_samples=num_samples, num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause"
        )

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_common_causes", "required_variables", "num_samples"],
        [
            (0.1, "backdoor.propensity_score_matching", 5, ["W0", "W1"], 5000),
            (0.1, "general_adjustment.propensity_score_matching", 5, ["W0", "W1"], 5000),
        ],
    )
    def test_refutation_bootstrap_refuter_binary_list_argument(
        self, error_tolerance, estimator_method, num_common_causes, required_variables, num_samples
    ):
        # generalized adjustment identification requires python >=3.10
        if estimator_method.startswith("general_adjustment") and sys.version_info < (3, 10):
            return
        refuter_tester = SimpleRefuter(
            error_tolerance, estimator_method, "bootstrap_refuter", required_variables=required_variables
        )
        refuter_tester.binary_treatment_testsuite(
            num_samples=num_samples, num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause"
        )

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_common_causes", "required_variables", "num_samples"],
        [(0.1, "iv.instrumental_variable", 5, ["-W0", "-W1"], 5000)],
    )
    def test_refutation_bootstrap_refuter_continuous_list_negative_argument(
        self, error_tolerance, estimator_method, num_common_causes, required_variables, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerance, estimator_method, "bootstrap_refuter", required_variables=required_variables
        )
        refuter_tester.continuous_treatment_testsuite(
            num_samples=num_samples, num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause"
        )  # Run atleast one common cause

    @mark.parametrize(
        ["error_tolerance", "estimator_method", "num_common_causes", "required_variables", "num_samples"],
        [
            (0.1, "backdoor.propensity_score_matching", 5, ["-W0", "-W1"], 5000),
            (0.1, "general_adjustment.propensity_score_matching", 5, ["-W0", "-W1"], 5000),
        ],
    )
    def test_refutation_bootstrap_refuter_binary_list_negative_argument(
        self, error_tolerance, estimator_method, num_common_causes, required_variables, num_samples
    ):
        # generalized adjustment identification requires python >=3.10
        if estimator_method.startswith("general_adjustment") and sys.version_info < (3, 10):
            return
        refuter_tester = SimpleRefuter(
            error_tolerance, estimator_method, "bootstrap_refuter", required_variables=required_variables
        )
        refuter_tester.binary_treatment_testsuite(
            num_samples=num_samples, num_common_causes=num_common_causes, tests_to_run="atleast-one-common-cause"
        )
