import numpy as np
import pandas as pd
import pytest

from dowhy.utils.cit import compute_ci, conditional_MI, partial_corr


class TestConditionalMI:
    """Tests for the conditional_MI function."""

    def test_multi_char_column_names(self):
        """Regression test for #949: column names with >1 character were iterated as chars."""
        rng = np.random.default_rng(42)
        n = 200
        # Use multi-character column names that would be broken by list("Foo") -> ['F','o','o']
        df = pd.DataFrame(
            {
                "Foo": rng.integers(0, 3, size=n),
                "Bar": rng.integers(0, 3, size=n),
                "Baz": rng.integers(0, 3, size=n),
            }
        )
        # Should not raise KeyError
        result = conditional_MI(data=df, x="Foo", y="Bar", z=["Baz"])
        assert isinstance(result, float)

    def test_single_char_column_names(self):
        """Single-character column names should still work."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "X": rng.integers(0, 2, size=n),
                "Y": rng.integers(0, 2, size=n),
                "Z": rng.integers(0, 2, size=n),
            }
        )
        result = conditional_MI(data=df, x="X", y="Y", z=["Z"])
        assert isinstance(result, float)

    def test_independent_variables_low_cmi(self):
        """Independent variables should have low conditional mutual information."""
        rng = np.random.default_rng(7)
        n = 5000
        df = pd.DataFrame(
            {
                "Alpha": rng.integers(0, 2, size=n),  # independent of Beta
                "Beta": rng.integers(0, 2, size=n),
                "Gamma": rng.integers(0, 2, size=n),
            }
        )
        result = conditional_MI(data=df, x="Alpha", y="Beta", z=["Gamma"])
        # Truly independent variables should yield low CMI
        assert result < 0.05

    def test_dependent_variables_high_cmi(self):
        """Fully dependent variables should have high conditional mutual information."""
        rng = np.random.default_rng(42)
        n = 1000
        x_vals = rng.integers(0, 2, size=n)
        df = pd.DataFrame(
            {
                "Foo": x_vals,
                "Bar": x_vals,  # identical to Foo -> fully dependent
                "Baz": rng.integers(0, 2, size=n),
            }
        )
        result = conditional_MI(data=df, x="Foo", y="Bar", z=["Baz"])
        # Fully dependent variables should yield CMI close to 1 bit
        assert result > 0.5


class TestComputeCi:
    """Tests for compute_ci() — verifies proper exception raising."""

    def test_missing_r_raises_value_error(self):
        with pytest.raises(ValueError, match="'r'.*'nx'"):
            compute_ci(r=None, nx=100)

    def test_missing_nx_raises_value_error(self):
        with pytest.raises(ValueError, match="'r'.*'nx'"):
            compute_ci(r=0.5, nx=None)

    def test_non_float_confidence_raises_type_error(self):
        with pytest.raises(TypeError, match="'confidence' must be a float"):
            compute_ci(r=0.5, nx=100, confidence=95)  # int, not float

    def test_confidence_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            compute_ci(r=0.5, nx=100, confidence=0.0)

    def test_confidence_one_raises_value_error(self):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            compute_ci(r=0.5, nx=100, confidence=1.0)

    def test_valid_call_returns_two_element_array(self):
        ci = compute_ci(r=0.5, nx=50, confidence=0.95)
        assert ci.shape == (2,)
        assert ci[0] < ci[1]


class TestPartialCorr:
    """Tests for partial_corr() — verifies proper exception raising."""

    def _make_data(self, n=50):
        rng = np.random.default_rng(42)
        z = rng.normal(size=n)
        x = z + rng.normal(size=n)
        y = z + rng.normal(size=n)
        return pd.DataFrame({"x": x, "y": y, "z": z})

    def test_too_few_samples_raises_value_error(self):
        data = self._make_data(n=50).iloc[:2]
        with pytest.raises(ValueError, match="more than 2 samples"):
            partial_corr(data=data, x="x", y="y", z=["z"])

    def test_x_equals_z_raises_value_error(self):
        data = self._make_data()
        with pytest.raises(ValueError, match="'x' and 'z' must be distinct"):
            partial_corr(data=data, x="x", y="y", z="x")

    def test_y_equals_z_raises_value_error(self):
        data = self._make_data()
        with pytest.raises(ValueError, match="'y' and 'z' must be distinct"):
            partial_corr(data=data, x="x", y="y", z="y")

    def test_x_equals_y_raises_value_error(self):
        data = self._make_data()
        with pytest.raises(ValueError, match="'x' and 'y' must be distinct"):
            partial_corr(data=data, x="x", y="x", z=["z"])

    def test_x_in_z_list_raises_value_error(self):
        data = self._make_data()
        with pytest.raises(ValueError, match="'x'.*must not appear in 'z'"):
            partial_corr(data=data, x="x", y="y", z=["z", "x"])

    def test_y_in_z_list_raises_value_error(self):
        data = self._make_data()
        with pytest.raises(ValueError, match="'y'.*must not appear in 'z'"):
            partial_corr(data=data, x="x", y="y", z=["z", "y"])

    def test_valid_pearson_returns_dict_with_expected_keys(self):
        data = self._make_data()
        result = partial_corr(data=data, x="x", y="y", z=["z"])
        assert set(result.keys()) == {"n", "r", "CI95%", "p-val"}
        assert -1 <= result["r"] <= 1

    def test_valid_spearman_returns_dict(self):
        data = self._make_data()
        result = partial_corr(data=data, x="x", y="y", z=["z"], method="spearman")
        assert "r" in result
        assert -1 <= result["r"] <= 1

    def test_all_missing_after_drop_raises_value_error(self):
        data = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0], "z": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="After dropping missing values"):
            partial_corr(data=data, x="x", y="y", z=["z"])
