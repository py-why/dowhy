import itertools

import numpy as np
import pandas as pd
import pytest
from pytest import mark

import dowhy.datasets
from dowhy import CausalModel
from dowhy.causal_estimators.tabpfn_estimator import TabpfnEstimator

# TabPFN and PyTorch are required for this test
tabpfn = pytest.importorskip("tabpfn")
torch = pytest.importorskip("torch")

from .base import SimpleEstimator


@mark.usefixtures("fixed_seed")
class TestTabpfnEstimator(object):
    """
    Test suite for TabPFN estimator.
    
    Important notes for test configuration:
    - TabPFN is extremely slow on CPU. For CPU environments, use small sample sizes (<=1000).
    - To enable multi-GPU testing, set use_multi_gpu=True in method_params.
    """
    
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "outcome_is_binary",
            "confidence_intervals",
            "test_significance",
            "num_samples",
            "identifier_method",
        ],
        [
            (
                0.4,
                TabpfnEstimator,
                [1, 2],
                [0],
                [0],
                [1],
                [True],
                [False],
                [True, False],
                [True, False],
                200,  # Small sample size for CPU compatibility
                "backdoor",
            ),
            (
                0.4,
                TabpfnEstimator,
                [1],
                [0],
                [0],
                [1],
                [True],
                [True],
                [True, False],
                [True, False],
                200,  # Small sample size for CPU compatibility
                "backdoor",
            ),
        ],
    )
    def test_average_treatment_effect(
        self,
        error_tolerance,
        Estimator,
        num_common_causes,
        num_instruments,
        num_effect_modifiers,
        num_treatments,
        treatment_is_binary,
        outcome_is_binary,
        confidence_intervals,
        test_significance,
        num_samples,
        identifier_method,
    ):
        """
        Test average treatment effect estimation using TabPFN.
        
        Note: We call average_treatment_effect_test directly instead of using
        average_treatment_effect_testsuite because the testsuite does not accept
        num_samples as a parameter. TabPFN requires smaller sample sizes on CPU
        due to performance constraints, so we need explicit control over num_samples.
        """
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        
        # Generate all test configurations
        args_dict = {
            "num_common_causes": num_common_causes,
            "num_instruments": num_instruments,
            "num_effect_modifiers": num_effect_modifiers,
            "num_treatments": num_treatments,
            "treatment_is_binary": treatment_is_binary,
            "outcome_is_binary": outcome_is_binary,
            "confidence_intervals": confidence_intervals,
            "test_significance": test_significance,
        }
        keys, values = zip(*args_dict.items())
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for cfg in configs:
            print(f"\nConfig: {cfg}")
            estimator_tester.average_treatment_effect_test(
                dataset="linear",
                num_samples=num_samples,
                **cfg,
                method_params={
                    "num_simulations": 8,
                    "num_null_simulations": 10,
                    "n_estimators": 4,
                    "model_type": "auto",
                    "use_multi_gpu": False,  # Set to True if testing with multiple GPUs
                    "max_num_classes": 10,
                },
            )

    def test_model_type_auto_detection(self):
        """Test that TabPFN correctly auto-detects classifier vs regressor based on outcome."""
        # Test regression case
        data_regression = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=2,
            num_instruments=0,
            num_effect_modifiers=0,
            num_treatments=1,
            num_samples=200,
            treatment_is_binary=True,
            outcome_is_binary=False,
        )

        model_regression = CausalModel(
            data=data_regression["df"],
            treatment=data_regression["treatment_name"],
            outcome=data_regression["outcome_name"],
            common_causes=data_regression["common_causes_names"],
            graph=data_regression["gml_graph"],
        )

        identified_estimand_regression = model_regression.identify_effect(proceed_when_unidentifiable=True)

        estimate_regression = model_regression.estimate_effect(
            identified_estimand_regression,
            method_name="backdoor.tabpfn",
            method_params={
                "estimator": TabpfnEstimator,
                "model_type": "auto",
                "n_estimators": 4,
                "use_multi_gpu": False,
            },
        )

        assert estimate_regression.estimator.tabpfn_model.resolved_model_type == "Regressor"

        # Test classification case
        data_classification = dowhy.datasets.linear_dataset(
            beta=5,
            num_common_causes=2,
            num_instruments=0,
            num_effect_modifiers=0,
            num_treatments=1,
            num_samples=200,
            treatment_is_binary=True,
            outcome_is_binary=True,
        )

        model_classification = CausalModel(
            data=data_classification["df"],
            treatment=data_classification["treatment_name"],
            outcome=data_classification["outcome_name"],
            common_causes=data_classification["common_causes_names"],
            graph=data_classification["gml_graph"],
        )

        identified_estimand_classification = model_classification.identify_effect(proceed_when_unidentifiable=True)

        estimate_classification = model_classification.estimate_effect(
            identified_estimand_classification,
            method_name="backdoor.tabpfn",
            method_params={
                "estimator": TabpfnEstimator,
                "model_type": "auto",
                "n_estimators": 4,
                "use_multi_gpu": False,
            },
        )

        assert estimate_classification.estimator.tabpfn_model.resolved_model_type == "Classifier"


    def test_tabpfn_predict_proba(self):
        """Test that classifier can predict probabilities correctly."""
        data = dowhy.datasets.linear_dataset(
            beta=5,
            num_common_causes=2,
            num_instruments=0,
            num_effect_modifiers=0,
            num_treatments=1,
            num_samples=200,
            treatment_is_binary=True,
            outcome_is_binary=True,
        )

        model = CausalModel(
            data=data["df"],
            treatment=data["treatment_name"],
            outcome=data["outcome_name"],
            common_causes=data["common_causes_names"],
            graph=data["gml_graph"],
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Fit estimator directly to access underlying model
        est = TabpfnEstimator(
            identified_estimand,
            confidence_intervals=False,
            method_params={"model_type": "classifier", "n_estimators": 4, "use_multi_gpu": False},
        )
        est.fit(data["df"])

        # Build model and check predict_proba
        features, tabpfn_model = est._build_model(data["df"])
        proba = tabpfn_model.predict_proba(features[:, 1:])  # remove intercept

        assert proba is not None
        assert proba.ndim == 2
        assert proba.shape[0] == features.shape[0]
        assert proba.shape[1] in (1, 2)  # binary classification: 1 or 2 columns
        assert np.all(proba >= 0) and np.all(proba <= 1)

        # Test that probabilities sum to 1 for each row (if 2 columns)
        if proba.shape[1] == 2:
            assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
