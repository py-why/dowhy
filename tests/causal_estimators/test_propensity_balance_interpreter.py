"""Tests for PropensityBalanceInterpreter.

Regression test for https://github.com/py-why/dowhy/issues/802:
pd.wide_to_long with stubnames=["W"] silently returned an empty DataFrame when
covariate column names did not start with "W", causing pd.concat to raise
ValueError: No objects to concatenate.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before any pyplot import

import pytest  # noqa: E402

import dowhy.datasets  # noqa: E402
from dowhy import EstimandType, identify_effect_auto  # noqa: E402
from dowhy.causal_estimators.propensity_score_stratification_estimator import (  # noqa: E402
    PropensityScoreStratificationEstimator,
)
from dowhy.graph import build_graph_from_str  # noqa: E402
from dowhy.interpreters.propensity_balance_interpreter import PropensityBalanceInterpreter  # noqa: E402


def _make_estimate(gml_graph, df, treatment_name, outcome_name, num_strata=5):
    """Helper: identify, fit, and estimate with PropensityScoreStratification."""
    target_estimand = identify_effect_auto(
        build_graph_from_str(gml_graph),
        observed_nodes=list(df.columns),
        action_nodes=treatment_name,
        outcome_nodes=outcome_name,
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
    )
    target_estimand.set_identifier_method("backdoor")
    estimator = PropensityScoreStratificationEstimator(
        identified_estimand=target_estimand,
        num_strata=num_strata,
    )
    estimator.fit(df)
    return estimator.estimate_effect(df, control_value=0, treatment_value=1, target_units="ate")


@pytest.fixture
def estimate_default_names(fixed_seed):
    """Estimate using the default dowhy linear_dataset (covariates named W0, W1, treatment v0)."""
    data = dowhy.datasets.linear_dataset(
        beta=5,
        num_common_causes=2,
        num_instruments=0,
        num_effect_modifiers=0,
        num_treatments=1,
        num_samples=2000,
        treatment_is_binary=True,
    )
    return _make_estimate(data["gml_graph"], data["df"], data["treatment_name"], data["outcome_name"])


@pytest.fixture
def estimate_custom_names(fixed_seed):
    """Estimate using a dataset with realistic column names (age, educ) — the bug scenario."""
    data = dowhy.datasets.linear_dataset(
        beta=5,
        num_common_causes=2,
        num_instruments=0,
        num_effect_modifiers=0,
        num_treatments=1,
        num_samples=2000,
        treatment_is_binary=True,
    )
    df = data["df"].rename(columns={"W0": "age", "W1": "educ", "v0": "treat"})
    gml_graph = data["gml_graph"].replace("W0", "age").replace("W1", "educ").replace("v0", "treat")
    return _make_estimate(gml_graph, df, ["treat"], "y")


@pytest.mark.usefixtures("fixed_seed")
class TestPropensityBalanceInterpreter:
    def _get_plot_df(self, estimate):
        """Call interpreter directly so we can inspect the returned DataFrame."""
        interp = PropensityBalanceInterpreter(estimate)
        return interp.interpret(estimate._data)

    def test_no_exception_with_default_names(self, estimate_default_names):
        """Interpreter should not raise with default W0/W1 covariate names."""
        self._get_plot_df(estimate_default_names)  # must not raise

    def test_no_exception_with_custom_names(self, estimate_custom_names):
        """Regression: interpreter must not raise ValueError with non-W named covariates (#802)."""
        self._get_plot_df(estimate_custom_names)  # must not raise

    def test_returns_nonempty_dataframe(self, estimate_custom_names):
        """Interpreter should return a non-empty DataFrame."""
        plot_df = self._get_plot_df(estimate_custom_names)
        assert plot_df is not None
        assert not plot_df.empty, "PropensityBalanceInterpreter returned an empty DataFrame"

    def test_has_expected_columns(self, estimate_custom_names):
        """Output DataFrame should have common_cause_id, std_mean_diff, and sample columns."""
        plot_df = self._get_plot_df(estimate_custom_names)
        assert "common_cause_id" in plot_df.columns
        assert "std_mean_diff" in plot_df.columns
        assert "sample" in plot_df.columns

    def test_has_both_samples(self, estimate_custom_names):
        """Output DataFrame should include both Unadjusted and PropensityAdjusted rows."""
        plot_df = self._get_plot_df(estimate_custom_names)
        samples = set(plot_df["sample"].unique())
        assert "Unadjusted" in samples, "Missing Unadjusted rows in output"
        assert "PropensityAdjusted" in samples, "Missing PropensityAdjusted rows in output"

    def test_covers_all_covariates(self, estimate_custom_names):
        """Every covariate should appear as a common_cause_id in the output."""
        plot_df = self._get_plot_df(estimate_custom_names)
        cause_names = set(estimate_custom_names.estimator._observed_common_causes_names)
        output_causes = set(plot_df["common_cause_id"].unique())
        assert cause_names == output_causes, f"Expected causes {cause_names}, got {output_causes}"
