"""Tests for DistanceMatchingEstimator, especially distance_metric_params handling."""

import numpy as np
import pandas as pd

from dowhy import CausalModel


def _make_binary_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    T = (rng.random(n) < 0.5).astype(int)
    Y = 2 * T + X1 + X2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"X1": X1, "X2": X2, "T": T, "Y": Y})


def _build_model(data):
    return CausalModel(
        data=data,
        treatment="T",
        outcome="Y",
        common_causes=["X1", "X2"],
    )


class TestDistanceMatchingEstimator:
    def test_default_minkowski(self):
        """Default Minkowski metric should work without any metric params."""
        data = _make_binary_dataset()
        model = _build_model(data)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
        )
        assert estimate.value is not None
        assert np.isfinite(estimate.value)

    def test_mahalanobis_with_V_matrix(self):
        """Mahalanobis metric with explicit V covariance matrix (issue #1390)."""
        data = _make_binary_dataset()
        X = data[["X1", "X2"]].values
        V = np.cov(X.T)

        model = _build_model(data)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"distance_metric": "mahalanobis", "V": V},
        )
        assert estimate.value is not None
        assert np.isfinite(estimate.value)

    def test_mahalanobis_with_VI_matrix(self):
        """Mahalanobis metric with VI (inverse covariance) matrix."""
        data = _make_binary_dataset()
        X = data[["X1", "X2"]].values
        VI = np.linalg.inv(np.cov(X.T))

        model = _build_model(data)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"distance_metric": "mahalanobis", "VI": VI},
        )
        assert estimate.value is not None
        assert np.isfinite(estimate.value)

    def test_mahalanobis_atc(self):
        """Mahalanobis with ATC target units."""
        data = _make_binary_dataset()
        X = data[["X1", "X2"]].values
        V = np.cov(X.T)

        model = _build_model(data)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="atc",
            method_params={"distance_metric": "mahalanobis", "V": V},
        )
        assert estimate.value is not None
        assert np.isfinite(estimate.value)

    def test_mahalanobis_ate(self):
        """Mahalanobis with ATE target units."""
        data = _make_binary_dataset()
        X = data[["X1", "X2"]].values
        V = np.cov(X.T)

        model = _build_model(data)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="ate",
            method_params={"distance_metric": "mahalanobis", "V": V},
        )
        assert estimate.value is not None
        assert np.isfinite(estimate.value)
