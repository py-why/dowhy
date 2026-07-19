"""
Regression tests for PropensityBalanceInterpreter.

Previously the interpreter required covariate columns named "W0", "W1", … (a
restriction from pd.wide_to_long) and hard-coded the treatment column name as
"v0" in a pandas query.  These tests verify the interpreter works correctly
with arbitrary covariate and treatment names.
"""

import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel


@pytest.mark.usefixtures("fixed_seed")
class TestPropensityBalanceInterpreter:
    """Test PropensityBalanceInterpreter with non-default column names."""

    def _build_estimate(self, treatment_name: str, covariate_names: list):
        """Return a fitted PropensityScoreStratification estimate."""
        np.random.seed(42)
        n = 500

        # Build a simple dataset with the requested column names
        common_causes = {col: np.random.normal(size=n) for col in covariate_names}
        df = pd.DataFrame(common_causes)
        propensity = 1 / (1 + np.exp(-(sum(df[c] for c in covariate_names))))
        df[treatment_name] = (np.random.uniform(size=n) < propensity).astype(int)
        df["outcome"] = df[treatment_name] * 2.0 + sum(df[c] for c in covariate_names) + np.random.normal(size=n)

        # Build the graph string
        gml = (
            f'graph [directed 1 node [id "{treatment_name}" label "{treatment_name}"] '
            + " ".join(f'node [id "{c}" label "{c}"]' for c in covariate_names)
            + ' node [id "outcome" label "outcome"] '
            + " ".join(
                f'edge [source "{c}" target "{treatment_name}"] edge [source "{c}" target "outcome"]'
                for c in covariate_names
            )
            + f' edge [source "{treatment_name}" target "outcome"]]'
        )

        model = CausalModel(
            data=df,
            treatment=treatment_name,
            outcome="outcome",
            graph=gml,
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_stratification",
        )
        return estimate, df

    def test_interpreter_with_standard_covariate_names(self):
        """Interpreter should work when covariates happen to be named W0, W1."""
        estimate, df = self._build_estimate("treatment", ["W0", "W1"])
        result = estimate.interpret(method_name="PropensityBalanceInterpreter", data=df)
        assert result is not None
        assert "common_cause_id" in result.columns
        assert set(result["common_cause_id"]).issubset({"W0", "W1"})

    def test_interpreter_with_arbitrary_covariate_names(self):
        """Interpreter must not fail with non-W0/W1 covariate column names.

        This was the root bug: pd.wide_to_long required columns named W0, W1,
        and the treatment filter hard-coded column name 'v0'.
        """
        estimate, df = self._build_estimate("treated", ["age", "income"])
        result = estimate.interpret(method_name="PropensityBalanceInterpreter", data=df)
        assert result is not None
        assert "common_cause_id" in result.columns
        assert set(result["common_cause_id"]).issubset({"age", "income"})

    def test_interpreter_returns_dataframe(self):
        """interpret() should return a DataFrame with std_mean_diff and sample columns."""
        estimate, df = self._build_estimate("T", ["X1", "X2"])
        result = estimate.interpret(method_name="PropensityBalanceInterpreter", data=df)
        assert isinstance(result, pd.DataFrame)
        assert "std_mean_diff" in result.columns
        assert "sample" in result.columns
        assert set(result["sample"]).issuperset({"Unadjusted", "PropensityAdjusted"})

    def test_interpreter_with_snake_case_method_name(self):
        """Interpreter lookup should continue to support snake_case names."""
        estimate, df = self._build_estimate("treatment", ["W0", "W1"])
        result = estimate.interpret(method_name="propensity_balance_interpreter", data=df)
        assert isinstance(result, pd.DataFrame)

    def test_interpreter_with_multiple_method_names_returns_list(self):
        """interpret() should return one result per requested interpreter method."""
        estimate, df = self._build_estimate("treatment", ["W0", "W1"])
        result = estimate.interpret(
            method_name=["PropensityBalanceInterpreter", "propensity_balance_interpreter"], data=df
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, pd.DataFrame) for item in result)
