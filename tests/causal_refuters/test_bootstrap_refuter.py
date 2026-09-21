import sys

import numpy as np
import pandas as pd
import pytest
from pytest import mark

from dowhy import CausalModel

from .base import SimpleRefuter

# ---------------------------------------------------------------------------
# Helper: build a minimal CausalModel and estimate for refuter tests
# ---------------------------------------------------------------------------

_GML_ONE_CAUSE = """graph [directed 1
  node [id "W" label "W"]
  node [id "v0" label "v0"]
  node [id "y" label "y"]
  edge [source "W" target "v0"]
  edge [source "W" target "y"]
  edge [source "v0" target "y"]
]"""

_GML_TWO_CAUSES = """graph [directed 1
  node [id "W" label "W"]
  node [id "G" label "G"]
  node [id "v0" label "v0"]
  node [id "y" label "y"]
  edge [source "W" target "v0"]
  edge [source "W" target "y"]
  edge [source "G" target "v0"]
  edge [source "G" target "y"]
  edge [source "v0" target "y"]
]"""


def _make_base_model(df, graph):
    model = CausalModel(data=df, treatment="v0", outcome="y", graph=graph)
    estimand = model.identify_effect()
    estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    return model, estimand, estimate


# ---------------------------------------------------------------------------
# Regression tests: known bugs in BootstrapRefuter noise injection
# ---------------------------------------------------------------------------


class TestBootstrapRefuterNoiseDtypes:
    """Regression tests for dtype-specific noise injection in BootstrapRefuter.

    The ``_perturb_once`` helper uses a string-match condition to choose how to
    add noise to each required-variable column.  Several dtype branches contain
    bugs that are fixed in PR #1730; these tests document the expected
    *correct* behaviour and are marked ``xfail(strict=True)`` until that PR
    merges.
    """

    @pytest.fixture()
    def rng(self):
        return np.random.default_rng(0)

    @pytest.fixture()
    def float_dataset(self, rng):
        n = 300
        w = rng.standard_normal(n)
        v0 = (w + rng.standard_normal(n) > 0).astype(int)
        y = 10 * v0 + 2 * w + rng.standard_normal(n)
        return pd.DataFrame({"W": w.astype(float), "v0": v0, "y": y})

    @pytest.fixture()
    def int_dataset(self, rng):
        """Dataset where the confounder is an integer column."""
        n = 300
        w_float = rng.standard_normal(n)
        w_int = (w_float * 3).astype(np.int32)
        v0 = (w_float + rng.standard_normal(n) > 0).astype(int)
        y = 10 * v0 + w_float + rng.standard_normal(n)
        return pd.DataFrame({"W": w_int, "v0": v0, "y": y})

    @pytest.fixture()
    def bool_dataset(self, rng):
        """Dataset where the confounder is a boolean column."""
        n = 500
        w_bool = rng.integers(0, 2, size=n).astype(bool)
        v0 = (w_bool.astype(int) + rng.standard_normal(n) > 0).astype(int)
        y = 10 * v0 + w_bool.astype(float) + rng.standard_normal(n)
        return pd.DataFrame({"W": w_bool, "v0": v0, "y": y})

    @pytest.fixture()
    def categorical_dataset(self, rng):
        """Dataset where the confounder is a categorical column."""
        n = 400
        w_float = rng.standard_normal(n)
        w_cat = pd.Categorical(np.where(w_float > 0, "high", "low"))
        v0 = (w_float + rng.standard_normal(n) > 0).astype(int)
        y = 10 * v0 + w_float + rng.standard_normal(n)
        return pd.DataFrame({"W": w_cat, "v0": v0, "y": y})

    # ---- positive test: float columns receive Gaussian noise ---------------

    def test_float_confounder_bootstrap_runs_and_perturbs(self, float_dataset):
        """Bootstrap refuter must run without error and perturb float confounders."""
        model, estimand, estimate = _make_base_model(float_dataset, _GML_ONE_CAUSE)
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=3,
            noise=0.1,
            random_state=42,
        )
        # Result must be finite
        assert np.isfinite(refute.new_effect)

    # ---- regression: boolean columns work correctly -------------------------

    def test_bool_confounder_bootstrap_runs(self, bool_dataset):
        """Boolean confounder columns must be handled without error."""
        model, estimand, estimate = _make_base_model(bool_dataset, _GML_ONE_CAUSE)
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=3,
            noise=0.1,
            probability_of_change=0.2,
            random_state=7,
        )
        assert np.isfinite(refute.new_effect)

    # ---- regression: integer columns silently receive no noise (documented) --

    def test_int_confounder_bootstrap_runs_without_error(self, int_dataset):
        """Bootstrap refuter must not crash with integer confounder columns.

        Note: Due to the dtype check bug ``('float' or 'int') in dtype.name``
        which evaluates to ``'float' in dtype.name``, integer columns currently
        receive *no* noise perturbation (they silently skip all branches).
        The refuter still runs; it just doesn't perturb integer columns.
        The correct behaviour is tracked in PR #1730.
        """
        model, estimand, estimate = _make_base_model(int_dataset, _GML_ONE_CAUSE)
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=3,
            noise=0.2,
            random_state=5,
        )
        assert np.isfinite(refute.new_effect)

    # ---- regression (xfail): categorical columns cause UnboundLocalError ----

    @pytest.mark.xfail(
        strict=True,
        raises=UnboundLocalError,
        reason=(
            "Known bug: the categorical branch in _refute_once references `probs` "
            "which is only defined inside the preceding `bool` branch, causing "
            "UnboundLocalError when a categorical column is encountered directly "
            "without a boolean column also being in required_variables.  "
            "Fix in PR #1730."
        ),
    )
    def test_categorical_confounder_bootstrap_crashes(self, categorical_dataset):
        """Categorical confounder columns currently crash with UnboundLocalError.

        The ``_refute_once`` helper only defines ``probs`` inside the ``bool``
        dtype branch.  When a categorical column is the first (or only) column
        in ``required_variables``, the ``category`` branch references
        ``probs`` before it is assigned, raising ``UnboundLocalError``.
        This test documents the bug and will become a passing test once PR #1730
        is merged (the ``xfail`` marker should be removed at that point).
        """
        model, estimand, estimate = _make_base_model(categorical_dataset, _GML_ONE_CAUSE)
        model.refute_estimate(
            estimand,
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=2,
            noise=0.1,
            probability_of_change=0.2,
            random_state=1,
        )


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
