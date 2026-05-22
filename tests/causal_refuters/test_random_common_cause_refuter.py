"""Tests for the RandomCommonCause refuter.

The RandomCommonCause refuter adds a randomly generated common cause to the
causal graph and re-estimates the effect.  For a correctly specified model the
new estimate should remain close to the original estimate, because an
independent Gaussian covariate cannot confound the treatment–outcome
relationship.
"""

import numpy as np
import pytest

import dowhy.datasets
from dowhy import CausalModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model_and_estimate(
    beta: float = 10.0,
    num_common_causes: int = 1,
    num_instruments: int = 1,
    num_samples: int = 2000,
    treatment_is_binary: bool = False,
    seed: int = 42,
):
    """Return (model, estimand, estimate) for a simple linear dataset."""
    np.random.seed(seed)
    data = dowhy.datasets.linear_dataset(
        beta=beta,
        num_common_causes=num_common_causes,
        num_instruments=num_instruments,
        num_samples=num_samples,
        treatment_is_binary=treatment_is_binary,
    )
    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"],
        outcome=data["outcome_name"],
        graph=data["gml_graph"],
        proceed_when_unidentifiable=True,
        test_significance=None,
    )
    estimand = model.identify_effect(method_name="exhaustive-search")
    estimand.set_identifier_method("backdoor")
    estimate = model.estimate_effect(
        identified_estimand=estimand,
        method_name="backdoor.linear_regression",
        test_significance=None,
    )
    return model, estimand, estimate


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fixed_seed")
class TestRandomCommonCauseRefuter:
    """Unit tests for the RandomCommonCause refuter."""

    def test_refutation_continuous_treatment_preserves_estimate(self):
        """Adding a random common cause should not materially change the estimate."""
        model, estimand, estimate = _build_model_and_estimate(beta=10.0, num_samples=2000, treatment_is_binary=False)
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=20,
            random_state=42,
        )
        error = abs(ref.new_effect - estimate.value)
        tolerance = abs(estimate.value) * 0.1  # within 10 % of original
        assert error < tolerance, (
            f"Refuted estimate {ref.new_effect:.4f} deviated too far from "
            f"original {estimate.value:.4f} (error={error:.4f}, tol={tolerance:.4f})"
        )

    def test_refutation_binary_treatment_preserves_estimate(self):
        """Refuter should preserve the estimate for binary treatments too."""
        model, estimand, estimate = _build_model_and_estimate(beta=2.0, num_samples=2000, treatment_is_binary=True)
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=20,
            random_state=42,
        )
        error = abs(ref.new_effect - estimate.value)
        tolerance = abs(estimate.value) * 0.1
        assert error < tolerance, (
            f"Refuted estimate {ref.new_effect:.4f} deviated too far from "
            f"original {estimate.value:.4f} (error={error:.4f}, tol={tolerance:.4f})"
        )

    def test_random_state_int_gives_reproducible_results(self):
        """Two calls with the same integer random_state must return identical new_effect."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=1000)
        ref1 = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=10,
            random_state=0,
        )
        ref2 = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=10,
            random_state=0,
        )
        assert ref1.new_effect == ref2.new_effect, (
            f"Results with the same random_state must be identical: " f"{ref1.new_effect} vs {ref2.new_effect}"
        )

    def test_random_state_numpy_gives_reproducible_results(self):
        """A np.random.RandomState seed should also yield reproducible results."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=1000)
        rs = np.random.RandomState(7)
        ref1 = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=10,
            random_state=rs,
        )
        rs2 = np.random.RandomState(7)
        ref2 = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=10,
            random_state=rs2,
        )
        assert ref1.new_effect == ref2.new_effect, (
            f"Results with the same np.random.RandomState seed must be identical: "
            f"{ref1.new_effect} vs {ref2.new_effect}"
        )

    def test_refutation_result_has_required_attributes(self):
        """The returned refutation object must expose expected attributes."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=1000)
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=10,
            random_state=42,
        )
        assert hasattr(ref, "new_effect"), "Refutation result must have 'new_effect'"
        assert hasattr(ref, "refutation_result"), "Refutation result must have 'refutation_result'"
        assert hasattr(ref, "refutation_type"), "Refutation result must have 'refutation_type'"
        assert isinstance(ref.new_effect, float), f"new_effect must be a float, got {type(ref.new_effect)}"

    def test_refutation_type_string(self):
        """The refutation_type should identify this as the RandomCommonCause refuter."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=1000)
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=10,
            random_state=42,
        )
        assert (
            "random common cause" in ref.refutation_type.lower()
        ), f"Unexpected refutation_type: {ref.refutation_type!r}"

    def test_num_simulations_parameter_respected(self):
        """The refuter should accept the num_simulations parameter without error."""
        model, estimand, estimate = _build_model_and_estimate(num_samples=1000)
        # Use a small number of simulations to keep the test fast
        ref = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=5,
            random_state=42,
        )
        assert ref.new_effect is not None
