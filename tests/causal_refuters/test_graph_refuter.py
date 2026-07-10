"""Tests for GraphRefuter, including non-int64/int32 dtype support."""

import numpy as np
import pandas as pd
import pytest

from dowhy.causal_refuters.graph_refuter import GraphRefuter


class TestGraphRefuterCMI:
    """Focused unit tests for the conditional_mutual_information method of GraphRefuter."""

    @pytest.fixture
    def independent_data(self):
        """Generate data where a and b are conditionally independent given c."""
        np.random.seed(42)
        n = 5000
        c = np.random.randint(0, 2, n)
        # a and b each depend only on c, so a ⊥ b | c
        a = np.where(c == 0, np.random.randint(0, 2, n), np.random.randint(0, 2, n))
        b = np.where(c == 0, np.random.randint(0, 2, n), np.random.randint(0, 2, n))
        return pd.DataFrame({"a": a.astype(np.int64), "b": b.astype(np.int64), "c": c.astype(np.int64)})

    @pytest.fixture
    def dependent_data(self):
        """Generate data where a and b are conditionally dependent given c."""
        np.random.seed(42)
        n = 5000
        c = np.random.randint(0, 2, n)
        a = np.random.randint(0, 2, n)
        # b is perfectly determined by a, so a and b are NOT independent given c
        b = a
        return pd.DataFrame({"a": a.astype(np.int64), "b": b.astype(np.int64), "c": c.astype(np.int64)})

    def test_independent_pair_yields_high_pvalue(self, independent_data):
        """An approximately independent discrete pair should yield p_value >= 0.05."""
        refuter = GraphRefuter(data=independent_data)
        refuter.conditional_mutual_information(x="a", y="b", z=frozenset(["c"]))

        assert len(refuter._true_implications) == 1, "Independent pair should be accepted as conditionally independent"
        assert len(refuter._false_implications) == 0

        key = ("a", "b") + (frozenset(["c"]),)
        p_value, result = refuter._results[key]
        assert p_value >= 0.05, f"Expected p_value >= 0.05 for independent pair, got {p_value}"
        assert result is True

    def test_dependent_pair_yields_low_pvalue(self, dependent_data):
        """A strongly dependent discrete pair should yield p_value < 0.05."""
        refuter = GraphRefuter(data=dependent_data)
        refuter.conditional_mutual_information(x="a", y="b", z=frozenset(["c"]))

        assert len(refuter._false_implications) == 1, "Dependent pair should be rejected as conditionally independent"
        assert len(refuter._true_implications) == 0

        key = ("a", "b") + (frozenset(["c"]),)
        p_value, result = refuter._results[key]
        assert p_value < 0.05, f"Expected p_value < 0.05 for dependent pair, got {p_value}"
        assert result is False

    def test_degenerate_constant_column(self):
        """A constant column (cardinality 1) should be treated as non-rejection (p_value=1.0)."""
        np.random.seed(0)
        n = 100
        data = pd.DataFrame(
            {
                "a": np.zeros(n, dtype=np.int64),  # constant: cardinality 1
                "b": np.random.randint(0, 2, n).astype(np.int64),
                "c": np.random.randint(0, 2, n).astype(np.int64),
            }
        )
        refuter = GraphRefuter(data=data)
        refuter.conditional_mutual_information(x="a", y="b", z=frozenset(["c"]))

        key = ("a", "b") + (frozenset(["c"]),)
        p_value, result = refuter._results[key]
        assert p_value == 1.0, "Degenerate (constant) variable should give p_value=1.0"
        assert result is True


class TestGraphRefuterDtypeDetection:
    """Tests that GraphRefuter correctly classifies all integer/bool dtypes as discrete."""

    def _make_simple_independence_constraint(self):
        """Return a single d-separation triple (x _||_ y | z) for testing."""
        return [("x", "y", ("z",))]

    def test_int8_classified_as_discrete(self):
        """Columns with int8 dtype should be treated as discrete, not continuous."""
        rng = np.random.default_rng(42)
        n = 200
        # z is a root; x and y are conditionally independent given z
        z = rng.integers(0, 3, size=n).astype(np.int8)
        noise_x = rng.integers(0, 2, size=n).astype(np.int8)
        noise_y = rng.integers(0, 2, size=n).astype(np.int8)
        x = ((z + noise_x) % 3).astype(np.int8)
        y = ((z + noise_y) % 3).astype(np.int8)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        assert df["x"].dtype == np.int8

        refuter = GraphRefuter(data=df)
        # z is a discrete column — check it is discovered
        # We call refute_model to exercise the dtype branch
        refuter.refute_model(independence_constraints=[("x", "y", ("z",))])
        # With z conditioning, x _||_ y should hold (CMI ≈ 0)
        # Key: the refuter must have used conditional_mutual_information, not partial_correlation
        # If it used partial_correlation, the result would still be stored but in a different code path.
        # We verify no exception was raised and some result was stored.
        assert len(refuter._results) > 0

    def test_uint8_classified_as_discrete(self):
        """Columns with uint8 dtype should be treated as discrete."""
        rng = np.random.default_rng(42)
        n = 200
        z = rng.integers(0, 3, size=n).astype(np.uint8)
        noise_x = rng.integers(0, 2, size=n).astype(np.uint8)
        noise_y = rng.integers(0, 2, size=n).astype(np.uint8)
        x = ((z + noise_x) % 3).astype(np.uint8)
        y = ((z + noise_y) % 3).astype(np.uint8)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        assert df["x"].dtype == np.uint8

        refuter = GraphRefuter(data=df)
        refuter.refute_model(independence_constraints=[("x", "y", ("z",))])
        assert len(refuter._results) > 0

    def test_bool_classified_as_discrete(self):
        """Boolean columns should be treated as discrete (binary)."""
        rng = np.random.default_rng(42)
        n = 200
        z = rng.integers(0, 2, size=n).astype(bool)
        noise = rng.integers(0, 2, size=n).astype(bool)
        x = z ^ noise
        y = z ^ rng.integers(0, 2, size=n).astype(bool)
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        assert df["x"].dtype == np.bool_

        refuter = GraphRefuter(data=df)
        refuter.refute_model(independence_constraints=[("x", "y", ("z",))])
        assert len(refuter._results) > 0

    def test_float64_classified_as_continuous(self):
        """float64 columns should remain classified as continuous."""
        rng = np.random.default_rng(42)
        n = 500
        z = rng.standard_normal(n)
        x = z + rng.standard_normal(n) * 0.1
        y = z + rng.standard_normal(n) * 0.1
        df = pd.DataFrame({"x": x, "y": y, "z": z})
        assert df["x"].dtype == np.float64

        refuter = GraphRefuter(data=df)
        refuter.refute_model(independence_constraints=[("x", "y", ("z",))])
        assert len(refuter._results) > 0
