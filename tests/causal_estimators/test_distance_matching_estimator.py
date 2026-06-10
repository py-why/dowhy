import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel
from dowhy.causal_estimators.distance_matching_estimator import DistanceMatchingEstimator

from .base import SimpleEstimator


@pytest.fixture
def binary_treatment_dataset():
    """Small deterministic dataset for distance matching tests."""
    rng = np.random.default_rng(42)
    n = 500
    w = rng.standard_normal(n)
    treatment = (w + rng.standard_normal(n) > 0).astype(int)
    outcome = 10 * treatment + 2 * w + rng.standard_normal(n)
    return pd.DataFrame({"W": w, "v0": treatment, "y": outcome})


@pytest.fixture
def binary_treatment_dataset_with_exact_col():
    """Dataset with a discrete covariate for exact matching tests."""
    rng = np.random.default_rng(0)
    n = 1000
    w_cont = rng.standard_normal(n)
    w_cat = rng.integers(0, 2, size=n)  # binary exact-match column
    treatment = ((w_cont + w_cat + rng.standard_normal(n)) > 0).astype(int)
    outcome = 10 * treatment + 2 * w_cont + 3 * w_cat + rng.standard_normal(n)
    return pd.DataFrame({"W": w_cont, "W_cat": w_cat, "v0": treatment, "y": outcome})


GML_SINGLE_CAUSE = """graph [directed 1 node [id "W" label "W"] node [id "v0" label "v0"]
node [id "y" label "y"] edge [source "W" target "v0"] edge [source "W" target "y"]
edge [source "v0" target "y"]]"""

GML_TWO_CAUSES = """graph [directed 1 node [id "W" label "W"] node [id "W_cat" label "W_cat"]
node [id "v0" label "v0"] node [id "y" label "y"]
edge [source "W" target "v0"] edge [source "W" target "y"]
edge [source "W_cat" target "v0"] edge [source "W_cat" target "y"]
edge [source "v0" target "y"]]"""


class TestDistanceMatchingEstimator:
    @pytest.mark.parametrize("target_units", ["att", "atc", "ate"])
    def test_estimate_is_close_to_true_effect(self, binary_treatment_dataset, target_units):
        """ATT/ATC/ATE estimate should be within a reasonable range of the true beta=10."""
        data = binary_treatment_dataset
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_SINGLE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units=target_units,
        )
        assert abs(estimate.value - 10) < 3.0, f"Estimate {estimate.value:.2f} too far from true effect 10"

    def test_matched_indices_att_populated(self, binary_treatment_dataset):
        """matched_indices_att should be populated when target_units='att'."""
        data = binary_treatment_dataset
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_SINGLE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name="backdoor.distance_matching", target_units="att")
        estimator = estimate.estimator
        assert estimator.matched_indices_att is not None
        assert len(estimator.matched_indices_att) == data["v0"].sum()

    def test_matched_indices_atc_populated(self, binary_treatment_dataset):
        """matched_indices_atc should be populated when target_units='atc'."""
        data = binary_treatment_dataset
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_SINGLE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name="backdoor.distance_matching", target_units="atc")
        estimator = estimate.estimator
        assert estimator.matched_indices_atc is not None
        assert len(estimator.matched_indices_atc) == (data["v0"] == 0).sum()

    def test_exact_match_restricts_matches_to_same_group(self, binary_treatment_dataset_with_exact_col):
        """With exact_match_cols, every matched control unit must share the same W_cat value."""
        data = binary_treatment_dataset_with_exact_col
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_TWO_CAUSES)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"fit_params": {"exact_match_cols": ["W_cat"]}},
        )
        estimator = estimate.estimator
        treated_indices = data.index[data["v0"] == 1]

        assert estimator.matched_indices_att is not None
        assert len(estimator.matched_indices_att) == len(treated_indices)

        for treated_idx, matched_control_indices in estimator.matched_indices_att.items():
            matched_control_indices = np.atleast_1d(matched_control_indices)
            assert len(matched_control_indices) > 0

            treated_group = data.loc[treated_idx, "W_cat"]
            matched_controls = data.loc[matched_control_indices]

            assert (matched_controls["v0"] == 0).all()
            assert (matched_controls["W_cat"] == treated_group).all()

    @pytest.mark.parametrize("target_units", ["att", "atc", "ate"])
    def test_exact_match_estimate_finite(self, binary_treatment_dataset_with_exact_col, target_units):
        """Estimates with exact_match_cols should be finite for all target_units."""
        data = binary_treatment_dataset_with_exact_col
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_TWO_CAUSES)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units=target_units,
            method_params={"fit_params": {"exact_match_cols": ["W_cat"]}},
        )
        assert np.isfinite(estimate.value), f"Non-finite estimate for target_units={target_units}"

    def test_invalid_target_units_raises(self, binary_treatment_dataset):
        """Passing an unsupported target_units string must raise ValueError."""
        data = binary_treatment_dataset
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_SINGLE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        with pytest.raises(ValueError, match="Target units string value not supported"):
            model.estimate_effect(estimand, method_name="backdoor.distance_matching", target_units="invalid")

    def test_non_binary_treatment_raises(self):
        """DistanceMatchingEstimator must raise when treatment is not binary."""
        rng = np.random.default_rng(7)
        n = 200
        data = pd.DataFrame({"W": rng.standard_normal(n), "v0": rng.integers(0, 4, n), "y": rng.standard_normal(n)})
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_SINGLE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        with pytest.raises(Exception, match="binary"):
            model.estimate_effect(estimand, method_name="backdoor.distance_matching", target_units="att")

    def test_average_treatment_effect_via_simple_estimator(self):
        """Smoke test using the shared SimpleEstimator harness."""
        tester = SimpleEstimator(error_tolerance=0.3, Estimator=DistanceMatchingEstimator)
        tester.average_treatment_effect_testsuite(
            num_common_causes=[1],
            num_instruments=[0],
            num_effect_modifiers=[0],
            num_treatments=[1],
            treatment_is_binary=[True],
            outcome_is_binary=[False],
            confidence_intervals=[False],
            test_significance=[False],
            method_params={"num_simulations": 5, "num_null_simulations": 5},
        )

    def test_custom_minkowski_p_param(self, binary_treatment_dataset):
        """Passing p via distance_metric_params should route through _build_nearest_neighbors."""
        data = binary_treatment_dataset
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_SINGLE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # p=1 gives Manhattan distance; should still produce a finite estimate
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"init_params": {"distance_metric": "minkowski", "p": 1}},
        )
        assert np.isfinite(estimate.value), "Estimate should be finite with custom p=1"
        assert abs(estimate.value - 10) < 5.0, f"Estimate {estimate.value:.2f} too far from true effect 10"
