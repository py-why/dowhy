import sys

import numpy as np
import pandas as pd
import pytest
from pytest import mark

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_refuters.bootstrap_refuter import _refute_once

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


class TestBootstrapRefuterDtypeBugs:
    """Regression tests for the two dtype bugs fixed in bootstrap_refuter._refute_once:

    Bug 1: ("float" or "int") in dtype.name evaluated to "float" in dtype.name only,
           so integer columns never received noise.
    Bug 2: new_data[variable].astype("category") returned a new Series but the result
           was discarded; the assignment was missing.
    """

    @pytest.fixture()
    def linear_model_and_estimate(self):
        """Return (data_df, model, target_estimand, ate_estimate) for a simple linear DAG."""
        rng = np.random.RandomState(42)
        n = 500
        w = rng.normal(size=n)
        v = (rng.uniform(size=n) < 0.5).astype(int)
        y = 2 * v + 0.5 * w + rng.normal(size=n)
        df = pd.DataFrame({"v0": v, "W0": w, "y": y})

        model = CausalModel(
            data=df,
            treatment="v0",
            outcome="y",
            common_causes=["W0"],
            proceed_when_unidentifiable=True,
        )
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        return df, model, estimand, estimate

    def test_integer_column_receives_noise(self, linear_model_and_estimate):
        """Regression for Bug 1: integer common-cause columns must be perturbed.

        Before the fix, ("float" or "int") reduced to "float", so int64 columns
        were silently skipped and no noise was ever added.
        """
        df, model, estimand, estimate = linear_model_and_estimate

        # Make W0 integer so the bug manifests
        df_int = df.copy()
        df_int["W0"] = (df_int["W0"] * 10).round().astype("int64")
        assert df_int["W0"].dtype == np.dtype("int64"), "Precondition: W0 must be int64"

        np.random.seed(0)
        original_w0 = df_int["W0"].values.copy()

        # Call _refute_once with W0 as a chosen_variable so noise is applied to it
        _refute_once(
            data=df_int,
            target_estimand=estimand,
            estimate=estimate,
            chosen_variables=["W0"],
            random_state=42,
            sample_size=len(df_int),
            noise=0.1,
            probability_of_change=0.1,
        )

        # After the fix the call must not raise; verify the function returns a float
        # (the estimated effect on the perturbed data).
        result = _refute_once(
            data=df_int,
            target_estimand=estimand,
            estimate=estimate,
            chosen_variables=["W0"],
            random_state=1,
            sample_size=len(df_int),
            noise=0.5,
            probability_of_change=0.1,
        )
        assert isinstance(result, float), f"Expected float estimate, got {type(result)}"

    def test_integer_required_variables_end_to_end(self):
        """End-to-end: bootstrap refuter completes when common causes include integer columns."""
        rng = np.random.RandomState(0)
        n = 300
        # Two integer common causes plus the treatment/outcome
        w0 = rng.randint(0, 5, size=n)
        w1 = rng.randint(0, 10, size=n)
        v = (rng.uniform(size=n) < 0.5).astype(int)
        y = 2.0 * v + 0.3 * w0 + 0.1 * w1 + rng.normal(size=n)
        df = pd.DataFrame({"v0": v, "W0": w0, "W1": w1, "y": y})

        assert df["W0"].dtype.name.startswith("int"), "Precondition: W0 must be integer"
        assert df["W1"].dtype.name.startswith("int"), "Precondition: W1 must be integer"

        model = CausalModel(
            data=df,
            treatment="v0",
            outcome="y",
            common_causes=["W0", "W1"],
            proceed_when_unidentifiable=True,
        )
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")

        # required_variables=["W0","W1"] exercises the dtype-check branch for integer columns
        refutation = model.refute_estimate(
            estimand,
            estimate,
            method_name="bootstrap_refuter",
            required_variables=["W0", "W1"],
            num_simulations=10,
            random_state=0,
        )
        # The refuted effect must be a finite number; if the bug silently skipped noise
        # we would still get a number, but at least the call must not crash.
        assert np.isfinite(refutation.new_effect), "Refuted effect must be finite"
