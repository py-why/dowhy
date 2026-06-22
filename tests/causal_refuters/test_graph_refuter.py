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
        # Should not raise KeyError
        refuter.conditional_mutual_information(x="Foo", y="Bar", z=["treatment"])

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
        assert result.__class__.__name__.endswith("Refutation")
        assert hasattr(result, "refutation_result")
        assert result.refutation_result is not None
