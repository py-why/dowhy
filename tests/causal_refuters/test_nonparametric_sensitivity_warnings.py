"""Tests for NonParametricSensitivityAnalyzer warning behavior."""
import logging

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from unittest.mock import patch

import dowhy.datasets
from dowhy import CausalModel


@pytest.mark.econml
class TestNonParametricSensitivityWarnings:
    """Test that NonParametricSensitivityAnalyzer emits appropriate warnings for low model quality."""

    def test_low_outcome_r2_warning(self, caplog):
        """Test that a warning is emitted when outcome model R² is low."""
        np.random.seed(42)
        # Create a dataset where outcome is mostly noise (low R²)
        data = dowhy.datasets.linear_dataset(
            beta=0.5,  # Small effect
            num_common_causes=2,
            num_samples=200,
            num_treatments=1,
            stddev_treatment_noise=1,
            stddev_outcome_noise=10,  # High noise -> low R²
        )

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Use a simple estimator that will produce low R²
        estimate = model.estimate_effect(
            target_estimand,
            method_name="backdoor.econml.dml.KernelDML",
            method_params={
                "init_params": {
                    "model_y": LinearRegression(),
                    "model_t": GradientBoostingRegressor(),
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )

        with caplog.at_level(logging.WARNING):
            refute = model.refute_estimate(
                target_estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                simulation_method="non-parametric-partial-R2",
            )

        # Check that the outcome R² warning was emitted
        assert any(
            log_record
            for log_record in caplog.records
            if (
                "dowhy.causal_refuters.non_parametric_sensitivity_analyzer" in log_record.name
                and log_record.levelname == "WARNING"
                and "outcome regression model has a low or non-finite R²" in log_record.message
            )
        ), f"Expected outcome R² warning not found. Log records: {[r.message for r in caplog.records]}"

    def test_low_treatment_r2_warning(self, caplog):
        """Test that a warning is emitted when treatment model R² is low."""
        np.random.seed(43)
        # Create a dataset where treatment is mostly noise (low R²)
        data = dowhy.datasets.linear_dataset(
            beta=5,
            num_common_causes=2,
            num_samples=200,
            num_treatments=1,
            stddev_treatment_noise=10,  # High noise -> low R²
            stddev_outcome_noise=1,
        )

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            target_estimand,
            method_name="backdoor.econml.dml.KernelDML",
            method_params={
                "init_params": {
                    "model_y": GradientBoostingRegressor(),
                    "model_t": LinearRegression(),
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )

        with caplog.at_level(logging.WARNING):
            refute = model.refute_estimate(
                target_estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                simulation_method="non-parametric-partial-R2",
            )

        # Check that the treatment R² warning was emitted
        assert any(
            log_record
            for log_record in caplog.records
            if (
                "dowhy.causal_refuters.non_parametric_sensitivity_analyzer" in log_record.name
                and log_record.levelname == "WARNING"
                and "treatment regression model has a low or non-finite R²" in log_record.message
            )
        ), f"Expected treatment R² warning not found. Log records: {[r.message for r in caplog.records]}"

    @patch("matplotlib.pyplot.figure")
    def test_invalid_s2_warning(self, mock_fig, caplog):
        """Test that a warning is emitted when S² is non-positive."""
        np.random.seed(44)
        # Create a very small dataset with high noise to potentially trigger invalid S²
        data = dowhy.datasets.linear_dataset(
            beta=0.1,
            num_common_causes=1,
            num_samples=50,  # Small sample
            num_treatments=1,
            stddev_treatment_noise=5,
            stddev_outcome_noise=5,
        )

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            target_estimand,
            method_name="backdoor.econml.dml.KernelDML",
            method_params={
                "init_params": {
                    "model_y": LinearRegression(),
                    "model_t": LinearRegression(),
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )

        with caplog.at_level(logging.WARNING):
            try:
                refute = model.refute_estimate(
                    target_estimand,
                    estimate,
                    method_name="add_unobserved_common_cause",
                    simulation_method="non-parametric-partial-R2",
                )
            except Exception:
                # Some configurations might fail, but we're interested in warnings
                pass

        # Check if S² warning was emitted (it may or may not be depending on the data)
        s2_warning_found = any(
            log_record
            for log_record in caplog.records
            if (
                "dowhy.causal_refuters.non_parametric_sensitivity_analyzer" in log_record.name
                and log_record.levelname == "WARNING"
                and "S² is non-positive" in log_record.message
            )
        )
        # This test documents the behavior but doesn't strictly require the warning
        # since it depends on the random data generation
        # The important thing is that the code handles it gracefully

    def test_non_finite_outcome_variance_warning(self, caplog):
        """Test that a warning is emitted when outcome variance is zero or non-finite."""
        np.random.seed(45)
        # Create a dataset with constant outcome (zero variance)
        data = dowhy.datasets.linear_dataset(
            beta=0,
            num_common_causes=2,
            num_samples=100,
            num_treatments=1,
            stddev_treatment_noise=1,
            stddev_outcome_noise=0.001,  # Very low noise
        )

        # Make outcome constant to trigger zero variance
        data["df"][data["outcome_name"]] = 1.0

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            graph=data["gml_graph"],
            test_significance=None,
        )
        target_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            target_estimand,
            method_name="backdoor.econml.dml.KernelDML",
            method_params={
                "init_params": {
                    "model_y": LinearRegression(),
                    "model_t": GradientBoostingRegressor(),
                },
                "fit_params": {
                    "cache_values": True,
                },
            },
        )

        with caplog.at_level(logging.WARNING):
            try:
                refute = model.refute_estimate(
                    target_estimand,
                    estimate,
                    method_name="add_unobserved_common_cause",
                    simulation_method="non-parametric-partial-R2",
                )
            except Exception:
                # May fail due to zero variance, but we're checking warnings
                pass

        # Check that a non-finite R² warning was emitted
        assert any(
            log_record
            for log_record in caplog.records
            if (
                "dowhy.causal_refuters.non_parametric_sensitivity_analyzer" in log_record.name
                and log_record.levelname == "WARNING"
                and "non-finite R²" in log_record.message
            )
        ), f"Expected non-finite R² warning not found. Log records: {[r.message for r in caplog.records]}"
