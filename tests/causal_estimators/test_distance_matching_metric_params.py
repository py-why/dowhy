"""Tests for distance_metric_params handling in DistanceMatchingEstimator.

Regression tests for https://github.com/py-why/dowhy/issues/1390 where
metric-specific parameters (V, VI, w) were spread as top-level kwargs to
NearestNeighbors instead of being passed through metric_params={}.
"""

import numpy as np
import pandas as pd
import pytest

from dowhy import CausalModel
from dowhy.causal_estimators.distance_matching_estimator import DistanceMatchingEstimator

GML_ONE_CAUSE = """graph [directed 1
  node [id "W" label "W"]
  node [id "v0" label "v0"]
  node [id "y" label "y"]
  edge [source "W" target "v0"]
  edge [source "W" target "y"]
  edge [source "v0" target "y"]
]"""


@pytest.fixture
def binary_dataset():
    """Small binary-treatment dataset with two confounders for metric-param tests."""
    rng = np.random.default_rng(7)
    n = 600
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    treatment = (w1 + w2 + rng.standard_normal(n) > 0).astype(int)
    outcome = 10 * treatment + 2 * w1 + w2 + rng.standard_normal(n)
    return pd.DataFrame({"W": w1, "v0": treatment, "y": outcome})


GML_TWO_CAUSES = """graph [directed 1
  node [id "W1" label "W1"]
  node [id "W2" label "W2"]
  node [id "v0" label "v0"]
  node [id "y" label "y"]
  edge [source "W1" target "v0"]
  edge [source "W1" target "y"]
  edge [source "W2" target "v0"]
  edge [source "W2" target "y"]
  edge [source "v0" target "y"]
]"""


@pytest.fixture
def binary_dataset_two_causes():
    """Binary-treatment dataset with two numeric confounders for Mahalanobis tests."""
    rng = np.random.default_rng(42)
    n = 600
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    treatment = (w1 + w2 + rng.standard_normal(n) > 0).astype(int)
    outcome = 10 * treatment + 2 * w1 + w2 + rng.standard_normal(n)
    return pd.DataFrame({"W1": w1, "W2": w2, "v0": treatment, "y": outcome})


class TestDistanceMetricParams:
    """Regression tests for issue #1390: metric-specific params must reach NearestNeighbors."""

    def test_mahalanobis_with_V_matrix(self, binary_dataset_two_causes):
        """Passing V matrix for Mahalanobis distance must not raise TypeError/ValueError."""
        data = binary_dataset_two_causes
        X = data[["W1", "W2"]].values
        V = np.cov(X.T)

        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_TWO_CAUSES)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"distance_metric": "mahalanobis", "V": V},
        )
        assert estimate.value is not None
        assert abs(estimate.value - 10) < 5.0, f"ATT estimate {estimate.value:.2f} unreasonably far from 10"

    def test_mahalanobis_with_VI_matrix(self, binary_dataset_two_causes):
        """Passing VI (inverse covariance) for Mahalanobis distance must not raise."""
        data = binary_dataset_two_causes
        X = data[["W1", "W2"]].values
        VI = np.linalg.inv(np.cov(X.T))

        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_TWO_CAUSES)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"distance_metric": "mahalanobis", "VI": VI},
        )
        assert estimate.value is not None

    def test_minkowski_with_p_param(self, binary_dataset):
        """Passing p=1 (Manhattan distance) via method_params must work correctly."""
        data = binary_dataset
        model = CausalModel(data=data, treatment="v0", outcome="y", graph=GML_ONE_CAUSE)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            estimand,
            method_name="backdoor.distance_matching",
            target_units="att",
            method_params={"distance_metric": "minkowski", "p": 1},
        )
        assert estimate.value is not None

    def test_nn_kwargs_helper_empty(self, binary_dataset):
        """_nn_kwargs() with no metric params should return empty dict."""
        from dowhy import EstimandType, identify_effect_auto
        from dowhy.graph import build_graph_from_str

        data = binary_dataset
        target_estimand = identify_effect_auto(
            build_graph_from_str(GML_ONE_CAUSE),
            observed_nodes=list(data.columns),
            action_nodes=["v0"],
            outcome_nodes=["y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")
        estimator = DistanceMatchingEstimator(identified_estimand=target_estimand)
        assert estimator._nn_kwargs() == {}
        assert estimator.distance_p is None
        assert estimator.distance_metric_params == {}

    def test_nn_kwargs_helper_with_p(self, binary_dataset):
        """_nn_kwargs() with p set should include p as top-level key."""
        from dowhy import EstimandType, identify_effect_auto
        from dowhy.graph import build_graph_from_str

        data = binary_dataset
        target_estimand = identify_effect_auto(
            build_graph_from_str(GML_ONE_CAUSE),
            observed_nodes=list(data.columns),
            action_nodes=["v0"],
            outcome_nodes=["y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")
        estimator = DistanceMatchingEstimator(identified_estimand=target_estimand, p=1)
        nn_kw = estimator._nn_kwargs()
        assert "p" in nn_kw
        assert nn_kw["p"] == 1
        assert "metric_params" not in nn_kw

    def test_nn_kwargs_helper_with_V(self, binary_dataset_two_causes):
        """_nn_kwargs() with V set should put V inside metric_params dict."""
        from dowhy import EstimandType, identify_effect_auto
        from dowhy.graph import build_graph_from_str

        data = binary_dataset_two_causes
        X = data[["W1", "W2"]].values
        V = np.cov(X.T)
        target_estimand = identify_effect_auto(
            build_graph_from_str(GML_TWO_CAUSES),
            observed_nodes=list(data.columns),
            action_nodes=["v0"],
            outcome_nodes=["y"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method("backdoor")
        estimator = DistanceMatchingEstimator(identified_estimand=target_estimand, distance_metric="mahalanobis", V=V)
        nn_kw = estimator._nn_kwargs()
        assert "metric_params" in nn_kw
        assert "V" in nn_kw["metric_params"]
        assert np.array_equal(nn_kw["metric_params"]["V"], V)
        assert "p" not in nn_kw
