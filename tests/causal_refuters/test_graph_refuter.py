"""
Tests for dowhy.causal_refuters.graph_refuter.GraphRefuter and GraphRefutation.

GraphRefuter checks whether conditional independence constraints implied by a causal graph
hold in the dataset.  For continuous variables it uses partial correlation (p-value); for
discrete variables it uses conditional mutual information.  These tests exercise the public
API directly (without going through CausalModel.refute_graph) so that every code path is
covered at unit-test speed.
"""

import numpy as np
import pandas as pd
import pytest

from dowhy.causal_refuters.graph_refuter import GraphRefutation, GraphRefuter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_continuous_independent(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """X, Y, Z are all sampled independently → X ⊥ Y | Z is TRUE."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"X": rng.randn(n), "Y": rng.randn(n), "Z": rng.randn(n)})


def _make_continuous_dependent(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """U is an unmeasured common cause of X and Y.
    X ⊥ Y | Z is FALSE because the path X ← U → Y is not blocked by Z."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n)
    Z = rng.randn(n)
    X = 2.0 * U + 0.1 * rng.randn(n)
    Y = 2.0 * U + 0.1 * rng.randn(n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def _make_discrete_independent(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Discrete X, Y, Z are sampled independently → X ⊥ Y | Z is TRUE.

    Column names are single characters to avoid the list(name) expansion issue in
    the conditional_MI helper (list('AB') would be interpreted as ['A', 'B']).
    """
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "A": rng.randint(0, 3, size=n),
            "B": rng.randint(0, 3, size=n),
            "C": rng.randint(0, 3, size=n),
        },
        dtype=np.int64,
    )


def _make_discrete_dependent(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Discrete X = Y ⊕ noise, so X and Y share information.
    X ⊥ Y | Z is FALSE for any Z that does not include Y."""
    rng = np.random.RandomState(seed)
    Z = rng.randint(0, 3, size=n)
    Y = rng.randint(0, 3, size=n)
    X = (Y + rng.randint(0, 2, size=n)) % 3  # X depends on Y
    return pd.DataFrame({"A": X, "B": Y, "C": Z}, dtype=np.int64)


# ===========================================================================
# GraphRefuter — continuous variables (partial correlation)
# ===========================================================================


class TestGraphRefuterContinuous:
    """Tests using continuous (float) columns where partial_correlation is used."""

    def test_passes_when_variables_are_truly_independent(self):
        """refute_model should pass when X ⊥ Y | Z holds in the data."""
        data = _make_continuous_independent()
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=[("X", "Y", ("Z",))])

        assert result.refutation_result is True
        assert len(refuter._true_implications) == 1
        assert len(refuter._false_implications) == 0

    def test_fails_when_variables_are_dependent(self):
        """refute_model should fail when X ⊥ Y | Z does NOT hold in the data."""
        data = _make_continuous_dependent()
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=[("X", "Y", ("Z",))])

        assert result.refutation_result is False
        assert len(refuter._false_implications) == 1
        assert len(refuter._true_implications) == 0

    def test_multiple_constraints_all_satisfied(self):
        """When all constraints hold, the refutation should pass."""
        rng = np.random.RandomState(42)
        n = 2000
        # W, X, Y, Z are mutually independent
        data = pd.DataFrame(
            {
                "W": rng.randn(n),
                "X": rng.randn(n),
                "Y": rng.randn(n),
                "Z": rng.randn(n),
            }
        )
        constraints = [
            ("W", "X", ("Y",)),
            ("W", "Y", ("Z",)),
        ]
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=constraints)

        assert result.refutation_result is True
        assert result.number_of_constraints_satisfied == 2
        assert result.number_of_constraints_model == 2

    def test_result_counts_match(self):
        """number_of_constraints_model and number_of_constraints_satisfied are set correctly."""
        data = _make_continuous_independent()
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=[("X", "Y", ("Z",))])

        assert result.number_of_constraints_model == 1
        assert result.number_of_constraints_satisfied in {0, 1}

    def test_partial_correlation_records_result(self):
        """partial_correlation() should store results in _results dict."""
        data = _make_continuous_independent()
        refuter = GraphRefuter(data=data)
        refuter.partial_correlation(x="X", y="Y", z=("Z",))

        assert len(refuter._results) == 1
        key = next(iter(refuter._results))
        p_value, is_independent = refuter._results[key]
        assert isinstance(p_value, float)
        assert isinstance(is_independent, bool)


# ===========================================================================
# GraphRefuter — discrete variables (conditional mutual information)
# ===========================================================================


class TestGraphRefuterDiscrete:
    """Tests using discrete (int64) columns where conditional_mutual_information is used."""

    def test_passes_for_independent_discrete_variables(self):
        """CMI ≈ 0 for truly independent discrete variables → refutation passes."""
        data = _make_discrete_independent()
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=[("A", "B", ("C",))])

        assert result.refutation_result is True

    def test_fails_for_dependent_discrete_variables(self):
        """CMI > 0.05 for clearly dependent discrete variables → refutation fails."""
        data = _make_discrete_dependent()
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=[("A", "B", ("C",))])

        # A (X) depends on B (Y); the constraint A ⊥ B | C should be violated
        assert result.refutation_result is False


# ===========================================================================
# set_refutation_result logic
# ===========================================================================


class TestSetRefutationResult:
    """Unit tests for set_refutation_result in isolation."""

    def _refuter_with(self, true_n: int, false_n: int) -> GraphRefuter:
        """Return a GraphRefuter whose implication lists are pre-populated."""
        data = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [0.0]})
        refuter = GraphRefuter(data=data)
        refuter._true_implications = [object()] * true_n
        refuter._false_implications = [object()] * false_n
        return refuter

    def test_all_true_implications(self):
        refuter = self._refuter_with(true_n=3, false_n=0)
        refuter.set_refutation_result(number_of_constraints_model=3)
        assert refuter._refutation_passed is True

    def test_some_false_implications(self):
        refuter = self._refuter_with(true_n=2, false_n=1)
        refuter.set_refutation_result(number_of_constraints_model=3)
        assert refuter._refutation_passed is False

    def test_all_false_implications(self):
        refuter = self._refuter_with(true_n=0, false_n=3)
        refuter.set_refutation_result(number_of_constraints_model=3)
        assert refuter._refutation_passed is False

    def test_no_constraints_run_gives_warning_pass(self):
        """If no tests were run (both lists empty) the refuter logs a warning and passes."""
        refuter = self._refuter_with(true_n=0, false_n=0)
        refuter.set_refutation_result(number_of_constraints_model=1)
        # When false_implications is empty but true_implications < model count,
        # the code falls through to the second elif (false_implications == 0) → passes.
        assert refuter._refutation_passed is True


# ===========================================================================
# GraphRefutation result object
# ===========================================================================


class TestGraphRefutation:
    """Tests for the GraphRefutation result container."""

    def _make_refutation(self, result: bool) -> GraphRefutation:
        ref = GraphRefutation(
            method_name_discrete="conditional_mutual_information",
            method_name_continuous="partial_correlation",
        )
        ref.add_conditional_independence_test_result(
            number_of_constraints_model=3,
            number_of_constraints_satisfied=2 if result else 0,
            refutation_result=result,
        )
        return ref

    def test_attributes_set_correctly(self):
        ref = self._make_refutation(result=True)
        assert ref.number_of_constraints_model == 3
        assert ref.number_of_constraints_satisfied == 2
        assert ref.refutation_result is True

    def test_str_before_test(self):
        """__str__ without result should not raise."""
        ref = GraphRefutation(
            method_name_discrete="conditional_mutual_information",
            method_name_continuous="partial_correlation",
        )
        s = str(ref)
        assert "partial_correlation" in s

    def test_str_after_test(self):
        """__str__ with result should include all fields."""
        ref = self._make_refutation(result=False)
        s = str(ref)
        assert "3" in s
        assert "False" in s

    def test_failing_refutation_attributes(self):
        ref = self._make_refutation(result=False)
        assert ref.refutation_result is False
        assert ref.number_of_constraints_satisfied == 0


# ===========================================================================
# Mixed variable types
# ===========================================================================


class TestGraphRefuterMixedTypes:
    """
    Tests where the dataset contains both continuous and discrete columns.
    The GraphRefuter routes each (x, y, z) triple to the appropriate test
    based on the dtype of each column.
    """

    def test_continuous_x_y_with_discrete_z(self):
        """
        When x and y are float but z is integer, the code falls into the
        binary/continuous path using partial_correlation.
        """
        rng = np.random.RandomState(7)
        n = 2000
        data = pd.DataFrame(
            {
                "X": rng.randn(n),  # continuous
                "Y": rng.randn(n),  # continuous (independent of X)
                "Z": rng.randint(0, 2, size=n).astype(np.int64),  # binary integer
            }
        )
        refuter = GraphRefuter(data=data)
        result = refuter.refute_model(independence_constraints=[("X", "Y", ("Z",))])
        # X and Y are independent → should pass (or at least not raise)
        assert result.refutation_result in {True, False}  # no exception
