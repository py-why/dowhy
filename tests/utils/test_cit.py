"""Unit tests for dowhy/utils/cit.py — validates proper exception raising."""

import numpy as np
import pandas as pd
import pytest

from dowhy.utils.cit import compute_ci, partial_corr


class TestComputeCi:
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
