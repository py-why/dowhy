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


class TestBootstrapRefuterBugFixes:
    """Regression tests for bugs fixed in bootstrap_refuter._refute_once."""

    def test_integer_column_receives_noise(self):
        """Bug 1: (\"float\" or \"int\") in dtype reduced to \"float\" in dtype, so integer
        columns silently received no noise.  After the fix the call must complete."""
        import numpy as np
        import pandas as pd

        from dowhy import CausalModel

        rng = np.random.RandomState(42)
        n = 300
        w = (rng.normal(size=n) * 10).astype("int64")
        v = (rng.uniform(size=n) < 0.5).astype(int)
        y = 2.0 * v + 0.3 * w + rng.normal(size=n)
        df = pd.DataFrame({"v0": v, "W0": w, "y": y})
        assert df["W0"].dtype == np.dtype("int64")

        model = CausalModel(
            data=df, treatment="v0", outcome="y", common_causes=["W0"], proceed_when_unidentifiable=True
        )
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        refutation = model.refute_estimate(
            estimand, estimate, method_name="bootstrap_refuter", required_variables=["W0"], num_simulations=5
        )
        assert np.isfinite(refutation.new_effect)

    def test_categorical_column_no_crash(self):
        """Bugs 2 & 3: the category branch used probs from the bool branch (NameError),
        called np.where with only 2 args (ValueError), and discarded the astype result.
        After the fix the call must complete without error and return a finite effect."""
        import numpy as np
        import pandas as pd

        from dowhy import CausalModel

        rng = np.random.RandomState(7)
        n = 400
        w_raw = rng.choice(["A", "B", "C"], size=n)
        w = pd.Categorical(w_raw)
        v = (rng.uniform(size=n) < 0.5).astype(int)
        y = 2.0 * v + (w_raw == "A").astype(float) + rng.normal(size=n)
        df = pd.DataFrame({"v0": v, "W0": w, "y": y})
        assert df["W0"].dtype.name == "category"

        model = CausalModel(
            data=df, treatment="v0", outcome="y", common_causes=["W0"], proceed_when_unidentifiable=True
        )
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        refutation = model.refute_estimate(
            estimand,
            estimate,
            method_name="bootstrap_refuter",
            required_variables=["W0"],
            num_simulations=5,
            random_state=42,
        )
        assert np.isfinite(refutation.new_effect)
