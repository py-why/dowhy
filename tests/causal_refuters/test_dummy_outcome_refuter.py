from pytest import mark

from .base import SimpleRefuter


def simple_linear_outcome_model(X_train, output_train):
    # The outcome is a linear function of the confounder
    # The slope is 1,2 and the intercept is 3
    return lambda X_train: X_train[:, 0] + 2 * X_train[:, 1] + 3


@mark.usefixtures("fixed_seed")
class TestDummyOutcomeRefuter(object):
    @mark.parametrize(["error_tolerence", "estimator_method"], [(0.03, "iv.instrumental_variable")])
    def test_refutation_dummy_outcome_refuter_default_continuous_treatment(self, error_tolerence, estimator_method):
        refuter_tester = SimpleRefuter(error_tolerence, estimator_method, "dummy_outcome_refuter")
        refuter_tester.continuous_treatment_testsuite(num_dummyoutcome_simulations=100)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "num_samples"], [(0.1, "backdoor.propensity_score_matching", 1000)]
    )
    def test_refutation_dummy_outcome_refuter_default_binary_treatment(
        self, error_tolerence, estimator_method, num_samples
    ):
        refuter_tester = SimpleRefuter(error_tolerence, estimator_method, "dummy_outcome_refuter")
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [(0.05, "iv.instrumental_variable", [("zero", ""), ("noise", {"std_dev": 1})])],
    )
    def test_refutation_dummy_outcome_refuter_randomly_generated_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )

        refuter_tester.continuous_treatment_testsuite()

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [(0.05, "backdoor.propensity_score_matching", [("zero", ""), ("noise", {"std_dev": 1})], 1000)],
    )
    def test_refutation_dummy_outcome_refuter_randomly_generated_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )

        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [(0.03, "iv.instrumental_variable", [("permute", {"permute_fraction": 1})])],
    )
    def test_refutation_dummy_outcome_refuter_permute_data_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )

        refuter_tester.continuous_treatment_testsuite()

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [(0.1, "backdoor.linear_regression", [("permute", {"permute_fraction": 1})], 1000)],
    )
    def test_refutation_dummy_outcome_refuter_permute_data_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )

        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [(0.2, "iv.instrumental_variable", [(simple_linear_outcome_model, {}), ("noise", {"std_dev": 1})])],
    )
    def test_refutation_dummy_outcome_refuter_custom_function_linear_regression_with_noise_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.xfail
    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [(0.2, "backdoor.linear_regression", [(simple_linear_outcome_model, {}), ("noise", {"std_dev": 1})], 1000)],
    )
    def test_refutation_dummy_outcome_refuter_custom_function_linear_regression_with_noise_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [
            (
                0.2,
                "backdoor.linear_regression",
                [("permute", {"permute_fraction": 0.5}), (simple_linear_outcome_model, {}), ("noise", {"std_dev": 1})],
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_custom_function_linear_regression_with_permute_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.xfail
    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.2,
                "backdoor.propensity_score_matching",
                [("permute", {"permute_fraction": 0.5}), (simple_linear_outcome_model, {}), ("noise", {"std_dev": 1})],
                1000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_custom_function_linear_regression_with_permute_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [(0.01, "iv.instrumental_variable", [("linear_regression", {}), ("zero", ""), ("noise", {"std_dev": 1})])],
    )
    def test_refutation_dummy_outcome_refuter_internal_linear_regression_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.2,
                "backdoor.propensity_score_matching",
                [("linear_regression", {}), ("zero", ""), ("noise", {"std_dev": 1})],
                1000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_linear_regression_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [(0.2, "iv.instrumental_variable", [("knn", {"n_neighbors": 5}), ("zero", ""), ("noise", {"std_dev": 1})])],
    )
    def test_refutation_dummy_outcome_refuter_internal_knn_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.2,
                "backdoor.propensity_score_matching",
                [("knn", {"n_neighbors": 5}), ("zero", ""), ("noise", {"std_dev": 1})],
                1000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_knn_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(tests_to_run="atleast-one-common-cause", num_samples=num_samples)

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.01,
                "iv.instrumental_variable",
                [("svm", {"C": 1, "epsilon": 0.2}), ("zero", ""), ("noise", {"std_dev": 1})],
                10000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_svm_continuous_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(num_samples=num_samples, tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.1,
                "backdoor.propensity_score_matching",
                [("svm", {"C": 1, "epsilon": 0.2}), ("zero", ""), ("noise", {"std_dev": 1})],
                1000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_svm_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(num_samples=num_samples, tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.01,
                "iv.instrumental_variable",
                [("random_forest", {"max_depth": 20}), ("zero", ""), ("noise", {"std_dev": 1})],
                10000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_random_forest_continuous_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(num_samples, tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.1,
                "backdoor.propensity_score_matching",
                [("random_forest", {"max_depth": 20}), ("zero", ""), ("noise", {"std_dev": 1})],
                1000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_random_forest_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(num_samples, tests_to_run="atleast-one-common-cause")

    # As we run with only one common cause and one instrument variable we run with (?, 2)
    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations"],
        [
            (
                0.01,
                "iv.instrumental_variable",
                [
                    ("neural_network", {"solver": "lbfgs", "alpha": 1e-5, "hidden_layer_sizes": (5, 2)}),
                    ("zero", ""),
                    ("noise", {"std_dev": 1}),
                ],
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_neural_network_continuous_treatment(
        self, error_tolerence, estimator_method, transformations
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.continuous_treatment_testsuite(tests_to_run="atleast-one-common-cause")

    @mark.parametrize(
        ["error_tolerence", "estimator_method", "transformations", "num_samples"],
        [
            (
                0.1,
                "backdoor.propensity_score_matching",
                [
                    ("neural_network", {"solver": "lbfgs", "alpha": 1e-5, "hidden_layer_sizes": (5, 2)}),
                    ("zero", ""),
                    ("noise", {"std_dev": 1}),
                ],
                1000,
            )
        ],
    )
    def test_refutation_dummy_outcome_refuter_internal_neural_network_binary_treatment(
        self, error_tolerence, estimator_method, transformations, num_samples
    ):
        refuter_tester = SimpleRefuter(
            error_tolerence, estimator_method, "dummy_outcome_refuter", transformations=transformations
        )
        refuter_tester.binary_treatment_testsuite(num_samples=num_samples, tests_to_run="atleast-one-common-cause")
