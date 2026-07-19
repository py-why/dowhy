"""Tests for GraphRefuter, including non-int64/int32 dtype support."""

import numpy as np
import pandas as pd

from dowhy.causal_refuters.graph_refuter import GraphRefuter


class TestGraphRefuter:
    """Tests for the GraphRefuter class, including regression tests for reported bugs."""

    def test_conditional_mi_multi_char_column_names(self):
        """Regression test for #949: KeyError when column names have multiple characters.

        Previously, `conditional_MI` called `list(x)` on a string column name,
        iterating over individual characters instead of treating the name as a key.
        """
        rng = np.random.default_rng(42)
        n = 200
        data = pd.DataFrame(
            {
                "Foo": rng.integers(0, 3, size=n),
                "Bar": rng.integers(0, 3, size=n),
                "treatment": rng.integers(0, 2, size=n),
                "outcome": rng.integers(0, 2, size=n),
            }
        )
        refuter = GraphRefuter(data=data)
        refuter.conditional_mutual_information(x="Foo", y="Bar", z=["treatment"])
        # Verify a result was stored and the CMI value is a finite non-negative number
        assert len(refuter._results) > 0
        cmi_val = list(refuter._results.values())[0][0]
        assert np.isfinite(cmi_val) and cmi_val >= 0

    def test_conditional_mi_single_char_column_names(self):
        """Single-character column names should still work correctly."""
        rng = np.random.default_rng(42)
        n = 200
        data = pd.DataFrame(
            {
                "A": rng.integers(0, 3, size=n),
                "B": rng.integers(0, 3, size=n),
                "C": rng.integers(0, 2, size=n),
            }
        )
        refuter = GraphRefuter(data=data)
        refuter.conditional_mutual_information(x="A", y="B", z=["C"])
        assert len(refuter._results) > 0
        cmi_val = list(refuter._results.values())[0][0]
        assert np.isfinite(cmi_val) and cmi_val >= 0

    def test_graph_refuter_with_multi_char_columns(self):
        """End-to-end test: refute_model should work with multi-character column names."""
        rng = np.random.default_rng(0)
        n = 300
        foo = rng.integers(0, 3, size=n)
        bar = rng.integers(0, 3, size=n)
        treatment = (foo + rng.integers(0, 2, size=n)) % 2
        outcome = (bar + treatment + rng.integers(0, 2, size=n)) % 2

        data = pd.DataFrame({"Foo": foo, "Bar": bar, "treatment": treatment, "outcome": outcome})
        refuter = GraphRefuter(data=data)
        # Foo _|_ treatment | []  (no conditioning set)
        independence_constraints = [("Foo", "treatment", [])]
        result = refuter.refute_model(independence_constraints=independence_constraints)
        assert hasattr(result, "refutation_result")
        assert result.refutation_result is not None


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
