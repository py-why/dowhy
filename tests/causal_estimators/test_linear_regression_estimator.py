import pytest
from pytest import mark

import dowhy.datasets
from dowhy import EstimandType, identify_effect_auto
from dowhy.causal_estimators.linear_regression_estimator import LinearRegressionEstimator
from dowhy.graph import build_graph_from_str

from .base import SimpleEstimator, TestGraphObject, example_graph


@mark.usefixtures("fixed_seed")
class TestLinearRegressionEstimator(object):
    @mark.parametrize(
        [
            "error_tolerance",
            "Estimator",
            "num_common_causes",
            "num_instruments",
            "num_effect_modifiers",
            "num_treatments",
            "treatment_is_binary",
            "treatment_is_category",
            "outcome_is_binary",
            "identifier_method",
        ],
        [
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    True,
                ],
                [
                    False,
                ],
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
                ],
                [
                    False,
                ],
                [
                    False,
                ],
                "backdoor",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    True,
                ],
                [
                    False,
                ],
                [
                    False,
                ],
                "general_adjustment",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
                ],
                [
                    True,
                ],
                [
                    False,
                ],
                "general_adjustment",
            ),
            (
                0.1,
                LinearRegressionEstimator,
                [0, 1],
                [0, 1],
                [
                    0,
                ],
                [1, 2],
                [
                    False,
                ],
                [
                    False,
                ],
                [
                    False,
                ],
                "general_adjustment",
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
        treatment_is_category,
        outcome_is_binary,
        identifier_method,
    ):
        estimator_tester = SimpleEstimator(error_tolerance, Estimator, identifier_method=identifier_method)
        estimator_tester.average_treatment_effect_testsuite(
            num_common_causes=num_common_causes,
            num_instruments=num_instruments,
            num_effect_modifiers=num_effect_modifiers,
            num_treatments=num_treatments,
            treatment_is_binary=treatment_is_binary,
            treatment_is_category=treatment_is_category,
            outcome_is_binary=outcome_is_binary,
            confidence_intervals=[
                True,
            ],
            test_significance=[
                True,
            ],
            method_params={"num_simulations": 10, "num_null_simulations": 10},
        )

    def test_general_adjustment_estimation_on_example_graphs(self, example_graph: TestGraphObject):
        data = dowhy.datasets.linear_dataset_from_graph(
            example_graph.graph,
            example_graph.action_nodes,
            example_graph.outcome_node,
            treatments_are_binary=True,
            outcome_is_binary=False,
            num_samples=50000,
        )
        data["df"] = data["df"][example_graph.observed_nodes]
        estimator_tester = SimpleEstimator(0.1, LinearRegressionEstimator, identifier_method="general_adjustment")
        estimator_tester.custom_data_average_treatment_effect_test(data)

    def test_none_identifier_method_does_not_raise(self):
        """identifier_method=None (functional API) should not raise ValueError."""
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=0,
            num_treatments=1,
            num_samples=500,
            treatment_is_binary=True,
        )
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        # functional API leaves identifier_method=None; estimator should not raise
        target_estimand.identifier_method = None
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand)
        estimator.fit(data["df"])  # should not raise

    @mark.parametrize("invalid_method", ["frontdoor", "iv", "mediation"])
    def test_invalid_identifier_method_raises(self, invalid_method):
        data = dowhy.datasets.linear_dataset(
            beta=10,
            num_common_causes=1,
            num_instruments=1,
            num_treatments=1,
            num_samples=1000,
            treatment_is_binary=True,
        )
        target_estimand = identify_effect_auto(
            build_graph_from_str(data["gml_graph"]),
            observed_nodes=list(data["df"].columns),
            action_nodes=data["treatment_name"],
            outcome_nodes=data["outcome_name"],
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
        )
        target_estimand.set_identifier_method(invalid_method)
        estimator = LinearRegressionEstimator(identified_estimand=target_estimand)
        with pytest.raises(ValueError, match="only supports backdoor and general_adjustment"):
            estimator.fit(data["df"])
