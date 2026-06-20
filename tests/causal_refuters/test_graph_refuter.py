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
