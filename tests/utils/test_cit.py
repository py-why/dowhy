"""Tests for dowhy.utils.cit — conditional information theory helpers."""
import numpy as np
import pandas as pd
import pytest

from dowhy.utils.cit import conditional_MI, entropy, partial_corr


class TestEntropy:
    def test_uniform_binary(self):
        """Uniform binary distribution has entropy = 1 bit."""
        values = [0, 1, 0, 1, 0, 1, 0, 1]
        assert entropy(values) == pytest.approx(1.0)

    def test_constant_distribution(self):
        """A constant random variable has zero entropy."""
        assert entropy([0, 0, 0, 0]) == pytest.approx(0.0)

    def test_deterministic_tuple_keys(self):
        """entropy() works with tuple keys (as used for joint distributions)."""
        values = [(0, 0), (1, 1), (0, 0), (1, 1)]
        assert entropy(values) == pytest.approx(1.0)


class TestConditionalMI:
    """Tests for conditional mutual information.

    Uses a controlled discrete dataset so results are deterministic without
    relying on random sampling.
    """

    @pytest.fixture()
    def independent_data(self):
        """X and Y are independent given Z (Z=0 for all rows here)."""
        rng = np.random.default_rng(0)
        n = 500
        z = rng.integers(0, 2, n)
        # X and Y generated independently from Z
        x = rng.integers(0, 2, n)
        y = rng.integers(0, 2, n)
        return pd.DataFrame({"x": x, "y": y, "z": z})

    @pytest.fixture()
    def dependent_data(self):
        """X and Y are dependent (Y == X) while Z is independent of both."""
        rng = np.random.default_rng(1)
        n = 500
        x = rng.integers(0, 2, n)
        y = x.copy()  # perfectly correlated
        z = rng.integers(0, 2, n)
        return pd.DataFrame({"x": x, "y": y, "z": z})

    @pytest.fixture()
    def multichar_independent_data(self):
        """Same as independent_data but with multi-character column names."""
        rng = np.random.default_rng(2)
        n = 500
        z0 = rng.integers(0, 2, n)
        x0 = rng.integers(0, 2, n)
        y0 = rng.integers(0, 2, n)
        return pd.DataFrame({"x0": x0, "y0": y0, "z0": z0})

    @pytest.fixture()
    def multichar_dependent_data(self):
        """Perfectly dependent variables with multi-character column names."""
        rng = np.random.default_rng(3)
        n = 500
        x0 = rng.integers(0, 2, n)
        y0 = x0.copy()
        z0 = rng.integers(0, 2, n)
        return pd.DataFrame({"x0": x0, "y0": y0, "z0": z0})

    # --- correctness ---

    def test_independent_cmi_near_zero(self, independent_data):
        """CMI of two independent variables should be close to zero."""
        cmi = conditional_MI(data=independent_data, x="x", y="y", z=("z",))
        assert cmi == pytest.approx(0.0, abs=0.1), f"Expected CMI ≈ 0, got {cmi}"

    def test_dependent_cmi_positive(self, dependent_data):
        """CMI of perfectly dependent variables should be substantially positive."""
        cmi = conditional_MI(data=dependent_data, x="x", y="y", z=("z",))
        assert cmi > 0.5, f"Expected CMI > 0.5, got {cmi}"

    # --- robustness to multi-character column names (regression for the list() expansion bug) ---

    def test_multichar_column_names_independent(self, multichar_independent_data):
        """conditional_MI must work correctly with multi-char column names."""
        cmi = conditional_MI(data=multichar_independent_data, x="x0", y="y0", z=("z0",))
        assert cmi == pytest.approx(0.0, abs=0.1), (
            f"CMI with multi-char column names should be ≈ 0 for independent vars, got {cmi}"
        )

    def test_multichar_column_names_dependent(self, multichar_dependent_data):
        """conditional_MI must detect dependence with multi-char column names."""
        cmi = conditional_MI(data=multichar_dependent_data, x="x0", y="y0", z=("z0",))
        assert cmi > 0.5, (
            f"CMI with multi-char column names should be large for dependent vars, got {cmi}"
        )

    def test_multiple_conditioning_variables(self):
        """conditional_MI handles z with multiple variables."""
        rng = np.random.default_rng(4)
        n = 500
        z1 = rng.integers(0, 2, n)
        z2 = rng.integers(0, 2, n)
        x = rng.integers(0, 2, n)
        y = rng.integers(0, 2, n)
        df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})
        cmi = conditional_MI(data=df, x="x", y="y", z=("z1", "z2"))
        # Independent variables → CMI near zero
        assert cmi == pytest.approx(0.0, abs=0.15), f"Expected CMI ≈ 0, got {cmi}"


class TestPartialCorr:
    """Basic sanity tests for partial_corr (which has correct column indexing)."""

    def test_uncorrelated_partial_corr_near_zero(self):
        """Partial correlation of uncorrelated continuous variables should be near zero."""
        rng = np.random.default_rng(10)
        n = 1000
        z = rng.standard_normal(n)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        result = partial_corr(data=df, x="x", y="y", z=["z"])
        assert abs(result["r"]) < 0.15, f"Expected near-zero partial corr, got {result['r']}"
        assert result["p-val"] > 0.01, f"Expected large p-value, got {result['p-val']}"

    def test_perfectly_correlated_partial_corr(self):
        """Partial correlation of y = x + z should capture residual correlation."""
        rng = np.random.default_rng(11)
        n = 1000
        z = rng.standard_normal(n)
        x = rng.standard_normal(n)
        y = x + 0.5 * z  # y directly depends on x
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        result = partial_corr(data=df, x="x", y="y", z=["z"])
        assert result["r"] > 0.8, f"Expected high positive partial corr, got {result['r']}"
        assert result["p-val"] < 0.001, f"Expected small p-value, got {result['p-val']}"
