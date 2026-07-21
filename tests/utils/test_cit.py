import numpy as np
import pandas as pd

from dowhy.utils.cit import conditional_MI


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
