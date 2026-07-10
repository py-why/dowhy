import numpy as np
import pandas as pd
import pytest
from pytest import mark

from dowhy.causal_refuters.dummy_outcome_refuter import preprocess_data_by_treatment

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


def _build_estimate():
    import numpy as np
    import pandas as pd

    from dowhy import CausalModel

    rng = np.random.RandomState(0)
    n = 500
    w = rng.normal(size=n)
    v = (rng.uniform(size=n) < 1 / (1 + np.exp(-w))).astype(int)
    y = 2 * v + w + rng.normal(size=n)
    data = pd.DataFrame({"v0": v, "W0": w, "y": y})
    model = CausalModel(data=data, treatment="v0", outcome="y", common_causes=["W0"])
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    return model, identified_estimand, estimate


def _run_dummy_outcome_refuter(random_state):
    model, identified_estimand, estimate = _build_estimate()
    refutations = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="dummy_outcome_refuter",
        num_simulations=10,
        transformation_list=[("permute", {"permute_fraction": 1}), ("noise", {"std_dev": 1})],
        random_state=random_state,
    )
    return [refutation.new_effect for refutation in refutations]


def test_dummy_outcome_refuter_is_reproducible_with_random_state():
    assert _run_dummy_outcome_refuter(random_state=123) == _run_dummy_outcome_refuter(random_state=123)


def test_dummy_outcome_refuter_differs_across_random_states():
    assert _run_dummy_outcome_refuter(random_state=123) != _run_dummy_outcome_refuter(random_state=456)


# ---------------------------------------------------------------------------
# Regression tests for preprocess_data_by_treatment
# ---------------------------------------------------------------------------


def _make_df(treatment_series, n=200):
    """Build a minimal DataFrame with a treatment column and outcome."""
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "treatment": treatment_series,
            "outcome": rng.normal(size=n),
            "covariate": rng.normal(size=n),
        }
    )


def test_preprocess_data_by_treatment_categorical_no_keyerror():
    """Regression: categorical treatment branch must NOT raise KeyError.

    Prior to the fix, the categorical branch mistakenly called
    ``data.groupby("bins")`` (overwriting the valid groupby with one that
    referenced a non-existent column), causing a guaranteed KeyError.
    """
    rng = np.random.RandomState(42)
    n = 200
    raw = rng.choice(["A", "B", "C"], size=n)
    treatment = pd.Categorical(raw, categories=["A", "B", "C"])
    df = _make_df(treatment, n=n)

    groups = preprocess_data_by_treatment(
        data=df,
        treatment_name=["treatment"],
        unobserved_confounder_values=None,
        bucket_size_scale_factor=0.5,
        chosen_variables=["covariate"],
    )
    group_keys = [k for k, _ in groups]
    assert set(group_keys) == {"A", "B", "C"}


def test_preprocess_data_by_treatment_continuous_observed_true():
    """Regression: pd.cut creates a Categorical 'bins' column; groupby must
    use observed=True so that empty bin categories are not included."""
    rng = np.random.RandomState(0)
    n = 100
    treatment = pd.Series(rng.uniform(0, 10, size=n))
    df = _make_df(treatment, n=n)

    # Should not raise and should return at least one group
    groups = preprocess_data_by_treatment(
        data=df,
        treatment_name=["treatment"],
        unobserved_confounder_values=None,
        bucket_size_scale_factor=0.5,
        chosen_variables=["covariate"],
    )
    assert len(list(groups)) > 0


def test_preprocess_data_by_treatment_multiple_treatments_raises():
    """preprocess_data_by_treatment must raise ValueError for >1 treatments."""
    rng = np.random.RandomState(0)
    n = 50
    df = pd.DataFrame({"t1": rng.randint(0, 2, size=n), "t2": rng.randint(0, 2, size=n), "outcome": rng.normal(size=n)})
    with pytest.raises(ValueError, match="single treatment"):
        preprocess_data_by_treatment(
            data=df,
            treatment_name=["t1", "t2"],
            unobserved_confounder_values=None,
            bucket_size_scale_factor=0.5,
            chosen_variables=[],
        )
