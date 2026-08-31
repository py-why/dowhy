"""Tests for dowhy/utils/regression.py utility functions."""

import numpy as np
import pandas as pd
import pytest

from dowhy.utils.regression import create_polynomial_function, generate_moment_function, get_numeric_features


class TestGetNumericFeatures:
    def test_all_numeric(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        indices = get_numeric_features(df)
        assert sorted(indices) == [0, 1]

    def test_mixed_types(self):
        df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["x", "y"]})
        indices = get_numeric_features(df)
        assert indices == [0]

    def test_no_numeric(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        indices = get_numeric_features(df)
        assert indices == []

    def test_column_order(self):
        df = pd.DataFrame({"z": [1.0], "a": [2.0], "m": [3.0]})
        indices = get_numeric_features(df)
        assert len(indices) == 3


class TestGenerateMomentFunction:
    def test_shape(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((50, 3))
        g = lambda x: x[:, 0]  # noqa: E731 — simple stub
        result = generate_moment_function(W, g)
        assert result.shape == (50,)

    def test_constant_treatment_function(self):
        """If g ignores x[:,0] (treatment column), moment should be zero."""
        W = np.ones((10, 3))
        g = lambda x: x[:, 1]  # noqa: E731 — depends only on confounder, not treatment
        result = generate_moment_function(W, g)
        np.testing.assert_allclose(result, 0.0)

    def test_linear_treatment_function(self):
        """If g(x) = x[:,0], moment = g(1,W) - g(0,W) = 1 - 0 = 1 for every row."""
        W = np.zeros((5, 2))
        g = lambda x: x[:, 0]  # noqa: E731
        result = generate_moment_function(W, g)
        np.testing.assert_allclose(result, 1.0)


class TestCreatePolynomialFunction:
    def test_returns_correct_number_of_functions(self):
        fns = create_polynomial_function(3)
        assert len(fns) == 4  # degrees 0, 1, 2, 3

    def test_degree_zero_is_constant_one(self):
        """Regression test: closure bug caused degree-0 fn to return x^max_degree."""
        fns = create_polynomial_function(4)
        x = np.array([[2.0], [5.0], [10.0]])
        np.testing.assert_allclose(fns[0](x), np.ones((3, 1)))

    def test_each_function_computes_correct_power(self):
        """Each function fns[k] should return x^k, not x^max_degree."""
        fns = create_polynomial_function(3)
        x = np.array([[2.0], [3.0]])
        for k, fn in enumerate(fns):
            expected = x**k
            np.testing.assert_allclose(fn(x), expected, err_msg=f"degree {k} wrong")

    def test_degree_zero_polynomial(self):
        fns = create_polynomial_function(0)
        assert len(fns) == 1
        x = np.array([[7.0], [8.0]])
        np.testing.assert_allclose(fns[0](x), np.ones((2, 1)))

    def test_functions_are_independent(self):
        """Calling functions in reverse order should give consistent results."""
        fns = create_polynomial_function(5)
        x = np.array([[3.0]])
        for k in reversed(range(6)):
            np.testing.assert_allclose(fns[k](x), x**k, err_msg=f"degree {k} wrong on reverse call")
